import numpy as np
import torch
from torch import nn, Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import random
from .log import log_info, log_warn, log_error, log_warn_once, print_table
from .monitor import format_timestamp, format_time
import json
import time
import sys


def seed_everything(seed:int=None):
    '''固定seed
    
    :param seed: int, 随机种子
    '''
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed is None) or not (min_seed_value <= seed <= max_seed_value):
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    log_info(f"Global seed set to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def print_trainable_parameters(module:nn.Module):
    '''打印可训练的参数量'''
    trainable_params = 0
    all_param = 0
    for _, param in module.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    log_info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def get_parameter_device(parameter):
    '''获取device, 从transformers包迁移过来'''
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


class DottableDict(dict):
    '''支持点操作符的字典'''
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()
KwargsConfig = DottableDict


def tran2dottableDict(dict_:dict):
    '''将一个嵌套字典转成DottableDict'''
    def traverse_dict(d): 
        d = DottableDict(d)
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = traverse_dict(value)
        return d
    return traverse_dict(dict_)


class JsonConfig:
    '''读取json配置文件并返回可.操作符的字典'''
    def __new__(self, json_path:str, encoding:str='utf-8', dot:bool=True):
        data = json.load(open(json_path, "r", encoding=encoding))
        if dot:
            return tran2dottableDict(data)
        else:
            return data


class YamlConfig:
    '''读取yaml配置文件并返回可.操作符的字典'''
    def __new__(self, yaml_path:str, encoding:str='utf-8', dot:bool=True):
        import yaml
        data = yaml.load(open(yaml_path, "r", encoding=encoding), Loader=yaml.FullLoader)
        if dot:
            return tran2dottableDict(data)
        else:
            return data

class IniConfig:
    '''读取ini配置文件'''
    def __new__(self, ini_path:str, encoding:str='utf-8', dot:bool=True):
        import configparser
        config = configparser.ConfigParser()
        config.read(ini_path, encoding=encoding)
        if dot:
            return DottableDict({section: DottableDict(config.items(section)) for section in config.sections()})
        else:
            return config


def auto_set_cuda_devices(best_num: Optional[int] = None) -> str:
    '''
    这段代码是一个名为 auto_set_cuda_devices 的函数, 它接受一个可选的整数参数 best_num。该函数用于自动设置环境变量 CUDA_VISIBLE_DEVICES, 以便在多个 GPU 设备中选择最佳的 GPU 设备。
    首先, 该函数检查环境变量 CUDA_VISIBLE_DEVICES 是否已经设置。如果已经设置, 则发出警告并返回当前设置的值。
    如果 best_num 等于 -1, 则将环境变量 CUDA_VISIBLE_DEVICES 设置为 -1 并返回。
    接下来, 该函数尝试使用 nvidia-smi 命令查询 GPU 设备的信息。如果命令不存在, 则发出警告并将环境变量 CUDA_VISIBLE_DEVICES 设置为 -1 并返回。
    如果未指定 best_num, 则从环境变量 LOCAL_WORLD_SIZE 中获取值。然后将其转换为整数。
    接下来, 该函数解析 nvidia-smi 命令的输出, 并计算每个 GPU 设备的得分。得分由 GPU 利用率和可用内存计算得出。
    最后, 该函数选择得分最高的 best_num 个 GPU 设备, 并将其索引作为字符串连接起来。然后将环境变量 CUDA_VISIBLE_DEVICES 设置为该字符串并返回。
    '''
    import subprocess

    # 无需重复设置
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        log_warn("Environment variable `CUDA_VISIBLE_DEVICES` has already been set")
        return os.environ["CUDA_VISIBLE_DEVICES"]

    if best_num == -1:
        log_info(f"SET CUDA_VISIBLE_DEVICES=-1")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "-1"

    try:
        p = subprocess.Popen(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE)
    except FileNotFoundError:
        log_error("`nvidia-smi` not exists")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "-1"

    if best_num is None:
        best_num = os.environ.get("LOCAL_WORLD_SIZE", 1)
    best_num = int(best_num)

    stdout, _ = p.communicate()
    outputs = stdout.decode("utf-8")
    lines = outputs.split(os.linesep)

    scores = []
    for item in lines:
        item_arr = item.split(",")
        if len(item_arr) != 4:
            continue
        gpu_score = 100 - int(item_arr[1].strip())
        memory_score = 100 * (int(item_arr[3].strip()) / int(item_arr[2].strip()))
        score = gpu_score + memory_score
        scores.append(score)

    best_num = min(best_num, len(scores))
    topk_idx = (-np.asarray(scores)).argsort()[:best_num]

    topk_idx_str = ",".join(map(str, topk_idx))
    log_info(f"SET CUDA_VISIBLE_DEVICES={topk_idx_str}")
    os.environ["CUDA_VISIBLE_DEVICES"] = topk_idx_str

    return topk_idx_str


def find_tied_parameters(model: nn.Module, **kwargs):
    """ copyed from accelerate
    Find the tied parameters in a given model.

    <Tip warning={true}>

    The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
    them.

    </Tip>

    Args:
        model (`torch.nn.Module`): The model to inspect.

    Returns:
        List[List[str]]: A list of lists of parameter names being all tied together.

    Example:

    ```py
    >>> from collections import OrderedDict
    >>> import torch.nn as nn

    >>> model = nn.Sequential(OrderedDict([("linear1", nn.Linear(4, 4)), ("linear2", nn.Linear(4, 4))]))
    >>> model.linear2.weight = model.linear1.weight
    >>> find_tied_parameters(model)
    [['linear1.weight', 'linear2.weight']]
    ```
    """
    # Initialize result and named_parameters before recursing.
    named_parameters = kwargs.get("named_parameters", None)
    prefix = kwargs.get("prefix", "")
    result = kwargs.get("result", {})

    if named_parameters is None:
        named_parameters = {n: p for n, p in model.named_parameters()}
    else:
        # A tied parameter will not be in the full `named_parameters` seen above but will be in the `named_parameters`
        # of the submodule it belongs to. So while recursing we track the names that are not in the initial
        # `named_parameters`.
        for name, parameter in model.named_parameters():
            full_name = name if prefix == "" else f"{prefix}.{name}"
            if full_name not in named_parameters:
                # When we find one, it has to be one of the existing parameters.
                for new_name, new_param in named_parameters.items():
                    if new_param is parameter:
                        if new_name not in result:
                            result[new_name] = []
                        result[new_name].append(full_name)

    # Once we have treated direct parameters, we move to the child modules.
    for name, child in model.named_children():
        child_name = name if prefix == "" else f"{prefix}.{name}"
        find_tied_parameters(child, named_parameters=named_parameters, prefix=child_name, result=result)

    # return FindTiedParametersResult([sorted([weight] + list(set(tied))) for weight, tied in result.items()])
    return {weight: list(set(tied)) for weight, tied in result.items()}


def check_cuda_verison():
    '''使用torch查看cuda, cudnn的版本'''
    versions = []
    # 查看PyTorch版本
    versions.append([f'torch', torch.__version__])

    # 查看CUDA版本（如果已安装）
    versions.append(["CUDA version", torch.version.cuda if torch.cuda.is_available() else 'Not available'])
    
    # 查看cuDNN版本（如果已安装）
    versions.append(["cuDNN version", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'])
    
    log_info('Cuda version summary')
    print_table(versions, headers=['name', 'version'])


def check_cuda_capability():
    '''打印各个显卡的算力'''
    # 查看CUDA版本（如果已安装）
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_summay = []
        for device in range(gpu_count):
            gpu_summay.append([device, torch.cuda.get_device_name(device), str(torch.cuda.get_device_capability(device))])
        log_info('CUDA device capability summary')
        print_table(gpu_summay, headers=['device', 'name', 'capability'])
    else:
        log_error('CUDA not available')


  
def check_file_modify_time(file_path:Union[str, List[str]], duration:int=None, verbose=0):
    """  
    判断文件被修改的时间
    :param file_path: 文件路径
    :param duration: 文件的日期如何在duration以内，认为文件刚刚被修改
    :return: 如果文件被修改，返回True；否则返回False
    """
    if isinstance(file_path, str):
        file_path = [file_path]
    
    results = []
    for file_i in file_path:
        # 获取文件的当前修改时间
        file_mtime = os.path.getmtime(file_i)

        # 比较当前修改时间和上次检查的修改时间
        cur_time = time.time()
        res = {'file_name': os.path.basename(file_i)}
        res['file_modify_time'] = format_timestamp(file_mtime)
        if duration is not None and isinstance(duration, int):
            res['current_time'] = format_timestamp(cur_time)
            diff_duration = cur_time - file_mtime
            res['diff_duration'] = format_time(diff_duration)
            res['modified'] = True if diff_duration <= duration else False
        results.append(res)
    
    if verbose > 0:
        print_table(results)

    if (len(results) == 1) and duration is not None:
        return results[0]['modified']
    else:
        return results


def check_file_modified(file_path:Union[str, List[str]], duration:int=1, verbose=0):
    '''判断文件在duration区间是否被修改
    '''
    return check_file_modify_time(file_path, duration, verbose)


def argument_parse(arguments:Union[str, list, dict]=None, description='argument_parse', parse_known_args:bool=True, dot:bool=True):
    ''' 根据传入的参数接受命令行参数，生成argparse.ArgumentParser
    :param arguments: 参数设置，接受str, list, dict输入
    :param description: 描述
    :param parse_known_args: bool, 只解析命令行中认识的参数

    Example
    -----------------------
    >>> args = argument_parse()
    >>> args = argument_parse('deepspeed')
    >>> args = argument_parse(['deepspeed'])
    >>> args = argument_parse({'deepspeed': {'type': str, 'help': 'deepspeed config path'}})
    '''
    import argparse
    parser = argparse.ArgumentParser(description=description)

    if arguments is None:
        # 不预设，直接解析所有命令行参数
        arguments = [arg for arg in sys.argv[1:] if arg.startswith('-') and ('=' not in arg)]
            
    if isinstance(arguments, str):
        parser.add_argument(f'--{arguments}')
    elif isinstance(arguments, list):
        for argument in arguments:
            if argument.startswith('-') and ('=' not in argument):
                parser.add_argument(f"--{argument.lstrip('-')}")
    elif isinstance(arguments, dict):
        for argument, kwargs in arguments.items():
            parser.add_argument(f'--{argument}',  **kwargs)
    else:
        raise TypeError('Args `arguments` only accepts `str,list,dict` format')

    if parse_known_args:
        args, unknown_args = parser.parse_known_args()  # 允许其他参数不传入
    else:
        args = parser.parse_args()
    
    if dot is True:
        args = tran2dottableDict(vars(args))
    return args
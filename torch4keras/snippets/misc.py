import numpy as np
import torch
from torch import nn, Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import random
from .log import log_info, log_warn, log_error
import json


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

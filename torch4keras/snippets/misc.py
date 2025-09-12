import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
import os
import random
from .log import log_info, log_warn, log_error, log_warn_once, print_table
from .monitor import format_timestamp, format_time
from .import_utils import is_torch_available
import json
import time
import sys
import shutil
import re
import requests
import functools
from collections import OrderedDict, defaultdict


if is_torch_available():
    import torch
    from torch import Tensor
    from torch.nn import Module
else:
    class Tensor: pass
    class Module: pass


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


def print_trainable_parameters(module:Module):
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

        def find_tensor_attributes(module: Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


class DottableDict(dict):
    '''支持点操作符的字典，包括自动创建不存在的键和嵌套字典'''  
    use_default_value: bool = False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convert_values_to_dottable()
  
    def _convert_values_to_dottable(self):
        """递归地将所有字典值转换为 DottableDict 实例"""
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, DottableDict):
                self[key] = DottableDict(value)
                self[key]._convert_values_to_dottable()  # 递归转换嵌套字典
  
    def __getattr__(self, item): 
        if self.use_default_value:  
            return self.get(item)
        else:
            try:
                return self[item]
            except KeyError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")
  
    def __setattr__(self, key, value):
        if key.startswith('__') and key.endswith('__'):
            # 允许设置特殊方法名
            super().__setattr__(key, value)
        else:
            if isinstance(value, dict) and not isinstance(value, DottableDict):
                value = DottableDict(value)
            self[key] = value
KwargsConfig = DottableDict


class CachedDictBySize(OrderedDict):
    '''按照size来缓存的字典, 超过maxsize会pop最老的item'''
    def __init__(self, *args, maxsize=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.__maxsize = maxsize
        # 如果初始化时字典已经超过了maxsize，则移除多余的项
        while len(self) > self.__maxsize:
            self.popitem(last=False)
    
    def __setitem__(self, key, value):
        # 检查字典是否已满，如果满了则移除最早的项
        if len(self) >= self.__maxsize:
            self.popitem(last=False)
        # 使用super()来避免无限递归
        super().__setitem__(key, value)


class CachedDictByFreq(dict):
    '''按照freq来缓存的字典, 超过maxsize会先pop频次最低的item'''
    def __init__(self, *args, maxsize=100, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.__maxsize = maxsize  
        self.__frequency = defaultdict(int)  # 用于跟踪每个键的访问频次  
          
        # 如果初始化时字典已经超过了maxsize，则移除多余的项  
        self._trim_to_maxsize(self.__maxsize)  
      
    def __setitem__(self, key, value):  
        if key not in self:
            # 先去掉最低频的，如果先set_item则可能会把当前key,value去掉
            self._trim_to_maxsize(self.__maxsize-1)
            
        super().__setitem__(key, value)  # 这里如果value不一致，则直接替换，仅以key为准
        
        # 更新频次
        self.__frequency[key] += 1
      
    def __getitem__(self, key):  
        # 每当项被访问时，更新其频次, 仅当key存在的时候，如果key不存在，则freq不变
        value = super().__getitem__(key)
        self.__frequency[key] += 1  
        return value
      
    def _trim_to_maxsize(self, maxsize):  
        # 辅助方法，用于在初始化或需要时修剪到maxsize  
        while len(self) > maxsize:  
            min_freq_key = min(self.__frequency, key=self.__frequency.get)  
            del self[min_freq_key]  
            del self.__frequency[min_freq_key]  


class CachedDictByTimeout(dict):
    '''按照time来缓存的字典, 超过最大时长会pop最老的item'''
    def __init__(self, *args, timeout=60, **kwargs):
        super().__init__(*args, **kwargs)
        self.__timeout = timeout  # 超时时间，以秒为单位
  
    def __setitem__(self, key, value):
        super().__setitem__(key, (value, time.time()))  # 存储值和创建时间（或更新时间）
  
    def __getitem__(self, key):
        value, last_access = super().__getitem__(key)
        if time.time() - last_access > self.__timeout:
            # 如果key已过期，则从字典中删除它并抛出KeyError
            del self[key]
            raise KeyError(f"Key `{key}` has expired")
        return value
  
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
  
    # 注意：由于我们重写了__getitem__，所以pop方法不会自动检查过期项
    # 但我们可以提供一个自定义的pop方法，如果需要的话
    def pop(self, key, default=None):
        if key in self:
            value, last_access = super().__getitem__(key)
            if time.time() - last_access > self.__timeout:
                del self[key]
                return default
            else:
                return super().pop(key)[0]  # 只返回值，不返回时间戳
        return default

 
class JsonConfig(DottableDict):
    '''读取json配置文件/字符串/字典并返回可.操作符的字典
    1. json文件路径
    2. json字符串
    3. 字典
    '''
    def __new__(cls, json_path_or_str:Union[str,dict], encoding:str='utf-8', dot:bool=True):
        # 字符串
        if isinstance(json_path_or_str, str):
            # 文件路径
            if os.path.exists(json_path_or_str):
                data = json.load(open(json_path_or_str, "r", encoding=encoding))
            # json字符串
            else:
                data = json.loads(json_path_or_str, "r", encoding=encoding)
        elif isinstance(json_path_or_str, dict):
            pass
        return DottableDict(data) if dot else data


class YamlConfig(DottableDict):
    '''读取yaml配置文件并返回可.操作符的字典'''
    def __new__(cls, yaml_path:str, encoding:str='utf-8', dot:bool=True):
        import yaml
        data = yaml.load(open(yaml_path, "r", encoding=encoding), Loader=yaml.FullLoader)
        return DottableDict(data) if dot else data


class IniConfig(DottableDict):
    '''读取ini配置文件'''
    def __new__(cls, ini_path:str, encoding:str='utf-8', dot:bool=True):
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


def find_tied_parameters(model: Module, **kwargs):
    """ copyed from accelerate
    Find the tied parameters in a given model.

    <Tip warning={true}>

    The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
    them.

    </Tip>

    Args:
        model (`torch.Module`): The model to inspect.

    Returns:
        List[List[str]]: A list of lists of parameter names being all tied together.

    Examples:
    ```python
    >>> from collections import OrderedDict
    >>> import torch.nn as nn

    >>> model = nn.Sequential(OrderedDict([("linear1", nn.Linear(4, 4)), ("linear2", nn.Linear(4, 4))]))
    >>> model.linear2.weight = model.linear1.weight
    >>> find_tied_parameters(model)
    ... # [['linear1.weight', 'linear2.weight']]
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


def check_url_available(url:str, timeout:int=5, verbose:int=0):
    '''检测某个网站是否可以访问'''
    try:  
        # 发送GET请求  
        response = requests.get(url, timeout=timeout)  # 设置超时时间为5秒  
          
        # 检查响应状态码  
        if response.status_code == 200:  
            return True  
        else:
            if verbose > 1:
                log_error(f"url={url}, response.status_code={response.status_code}")  
            return False  
    except requests.exceptions.RequestException as e:  
        # 处理请求异常，例如DNS查询失败、拒绝连接等
        if verbose > 1:
            log_error(f"Access {url} error: {e}")  
        return False


@functools.lru_cache(None)
def check_url_available_cached(url:str, timeout:int=5, verbose:int=0):
    '''检测某个网站是否可以访问, 不重复检测节约时间'''
    return check_url_available(url, timeout, verbose)


def argument_parse(arguments:Union[str, list, dict]=None, description='argument_parse', parse_known_args:bool=True, dot:bool=True):
    ''' 根据传入的参数接受命令行参数，生成argparse.ArgumentParser
    :param arguments: 参数设置，接受str, list, dict输入
    :param description: 描述
    :param parse_known_args: bool, 只解析命令行中认识的参数

    Examples:
    ```python
    >>> args = argument_parse()
    >>> args = argument_parse('deepspeed')
    >>> args = argument_parse(['deepspeed'])
    >>> args = argument_parse({'deepspeed': {'type': str, 'help': 'deepspeed config path'}})
    ```
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
        args = DottableDict(vars(args))
    return args


def cuda_empty_cache(device=None):
    '''清理gpu显存'''
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            return
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    else:
        log_warn_once('torch.cuda.is_available() = False')


class WebServing(object):
    """简单的Web接口，基于bottlepy简单封装，仅作为临时测试使用，不保证性能。

    Examples:
    ```python
    >>> arguments = {'text': (None, True), 'n': (int, False)}
    >>> web = WebServing(port=8864)
    >>> web.route('/gen_synonyms', gen_synonyms, arguments)
    >>> web.start()
    >>> # 然后访问 http://127.0.0.1:8864/gen_synonyms?text=你好
    ```
    
    ```shell
    >>> # 依赖（如果不用 server='paste' 的话，可以不装paste库）:
    >>> pip install bottle
    >>> pip install paste
    ```
    """
    def __init__(self, host='0.0.0.0', port=8000, server='paste'):

        import bottle

        self.host = host
        self.port = port
        self.server = server
        self.bottle = bottle

    def wraps(self, func, arguments, method='GET'):
        """封装为接口函数

        :param func: 要转换为接口的函数，需要保证输出可以json化，即需要保证 json.dumps(func(inputs)) 能被执行成功；
        :param arguments: 声明func所需参数，其中key为参数名，value[0]为对应的转换函数（接口获取到的参数值都是字符串型），value[1]为该参数是否必须；
        :param method: 'GET'或者'POST'。
        """
        def new_func():
            outputs = {'code': 0, 'desc': u'succeeded', 'data': {}}
            kwargs = {}
            for key, value in arguments.items():
                if method == 'GET':
                    result = self.bottle.request.GET.getunicode(key)
                else:
                    result = self.bottle.request.POST.getunicode(key)
                if result is None:
                    if value[1]:
                        outputs['code'] = 1
                        outputs['desc'] = 'lack of "%s" argument' % key
                        return json.dumps(outputs, ensure_ascii=False)
                else:
                    if value[0] is not None:
                        result = value[0](result)
                    kwargs[key] = result
            try:
                outputs['data'] = func(**kwargs)
            except Exception as e:
                outputs['code'] = 2
                outputs['desc'] = str(e)
            return json.dumps(outputs, ensure_ascii=False)

        return new_func

    def route(self, path, func, arguments, method='GET'):
        """添加接口"""
        func = self.wraps(func, arguments, method)
        self.bottle.route(path, method=method)(func)

    def start(self):
        """启动服务"""
        self.bottle.run(host=self.host, port=self.port, server=self.server)


def copytree(src:str, dst:str, ignore_copy_files:str=None, dirs_exist_ok=False):
    '''从一个文件夹copy到另一个文件夹
    
    :param src: str, copy from src
    :param dst: str, copy to dst
    '''
    def _ignore_copy_files(path, content):
        to_ignore = []
        if ignore_copy_files is None:
            return to_ignore
        
        for file_ in content:
            for pattern in ignore_copy_files:
                if re.search(pattern, file_):
                    to_ignore.append(file_)
        return to_ignore

    if src:
        os.makedirs(src, exist_ok=True)
    shutil.copytree(src, dst, ignore=_ignore_copy_files, dirs_exist_ok=dirs_exist_ok)


def add_start_docstrings(*docstr):
    '''装饰器，用于在类docstring前添加统一说明'''
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_model_forward(*docstr):
    '''装饰器，用于在nn.Module.forward前添加统一说明'''
    def docstring_decorator(fn):
        docstring = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        class_name = f"[`{fn.__qualname__.split('.')[0]}`]"
        intro = f"   The {class_name} forward method, overrides the `__call__` special method."
        note = r"""

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>
"""

        fn.__doc__ = intro + note + docstring
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    '''装饰器，用于在类docstring后添加统一说明'''
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator


class NoopContextManager:
    '''无意义的上下文管理器占位'''
    def __enter__(self):
        # 不执行任何操作
        return None
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 不执行任何操作，也不抑制异常
        return False



def cache_text(file_path=None):
    '''把一个字符串缓存到本地txt'''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if file_path is not None:
                if os.path.exists(file_path):
                    # 文件存在则直接读取
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    # 文件不存在则调用原函数提取文本
                    text: str = func(*args, **kwargs) or ""  # 调用原函数提取文本

                    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 确保目录存在
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    return text
            else:
                return func(*args, **kwargs) or ""
        return wrapper
    return decorator


def try_except(
    exception: Type[Exception] = Exception,  # 默认捕获所有 Exception
    logger: Optional[Callable[[str], None]] = None,  # 自定义日志记录函数
    default: Any = None,  # 异常时的返回值
    reraise: bool = False,  # 是否重新抛出异常
):
    """
    通用的 try-except 装饰器
    
    Args:
        exception: 要捕获的异常类型（默认捕获所有 Exception）
        logger: 日志记录函数（如 logging.error 或 print）
        default: 发生异常时的返回值
        reraise: 是否重新抛出异常（True 时忽略 default 参数）
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception as e:
                error_msg = f"Exception in {func.__name__}: {str(e)}"
                if logger:
                    logger(error_msg)
                else:
                    log_error(error_msg)  # 默认打印到控制台
                
                if reraise:
                    raise  # 重新抛出异常
                return default
        return wrapper
    return decorator


class TryExcept:
    """
    通用的 try-except 上下文管理器
    
    Args:
        exception: 要捕获的异常类型（默认捕获所有 Exception）
        logger: 日志记录函数（如 logging.error 或 print）
        default: 发生异常时的返回值（仅对返回值的上下文管理器有用）
        reraise: 是否重新抛出异常
    """
    def __init__( 
        self,
        exception: Type[Exception] = Exception,
        logger: Optional[Callable[[str], None]] = None,
        default: Any = None,
        reraise: bool = False,
    ):
        self.exception = exception
        self.logger = logger
        self.default = default
        self.reraise = reraise
 
    def __enter__(self):
        """进入上下文时无需特殊操作，返回自身以便在 with 块中使用"""
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时处理异常"""
        if exc_type is not None:  # 如果有异常发生
            if issubclass(exc_type, self.exception):  # 检查是否是指定的异常类型
                error_msg = f"{exc_val}"
                if self.logger:
                    self.logger(error_msg)
                else:
                    log_error(error_msg)  # 默认打印到控制台
                
                if self.reraise:
                    return False  # 重新抛出异常（返回 False 会传播异常）
                return True  # 抑制异常（返回 True 表示已处理）
            else:
                return False  # 不是指定的异常类型，重新抛出
        return False  # 没有异常发生
 
    def run(self, func: Callable, *args, **kwargs) -> Any:
        """
        可选：像装饰器一样运行函数（类似原装饰器的功能）
        """
        try:
            return func(*args, **kwargs)
        except self.exception as e:
            error_msg = f"Exception in {func.__name__}: {e}"
            if self.logger:
                self.logger(error_msg)
            else:
                log_error(error_msg)
            
            if self.reraise:
                raise
            return self.default

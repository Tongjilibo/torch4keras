from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import functools
import logging
import datetime


def colorful(obj, color="yellow", display_type="plain"):
    '''
    彩色输出格式：
    设置颜色开始 ：[显示方式;前景色;背景色m
    ### 颜色说明
    #  
    |前景色      |      背景色     |      颜色|
    |------------|-----------------|----------|
    |30          |      40         |     黑色|
    |31          |      41         |     红色|
    |32          |      42         |     绿色|
    |33          |      43         |     黃色|
    |34          |      44         |     蓝色|
    |35          |      45         |     紫红色|
    |36          |      46         |     青蓝色|
    |37          |      47         |     白色|

    ### 显示方式
    |显示方式     |      意义      |
    |-------------|------------   |
    |0            |    终端默认设置|
    |1            |    高亮显示    |
    |4            |    使用下划线  |
    |5            |    闪烁        |
    |7            |    反白显示    |
    |8            |    不可见     |
    '''
    color_dict = {"black":"30", "red":"31", "green":"32", "yellow":"33",
                  "blue":"34", "purple":"35","cyan":"36",  "white":"37"}
    display_type_dict = {"plain":"0","highlight":"1","underline":"4",
                "shine":"5","inverse":"7","invisible":"8"}
    s = str(obj)
    color_code = color_dict.get(color,"")
    display  = display_type_dict.get(display_type,"")
    out = '\033[{};{}m'.format(display,color_code)+s+'\033[0m'
    return out 


def black(obj, display_type="plain"):
    '''黑色'''
    return colorful(obj, color='black', display_type=display_type)


def red(obj, display_type="plain"):
    '''红色'''
    return colorful(obj, color='red', display_type=display_type)


def green(obj, display_type="plain"):
    '''绿色'''
    return colorful(obj, color='green', display_type=display_type)


def yellow(obj, display_type="plain"):
    '''黄色'''
    return colorful(obj, color='yellow', display_type=display_type)


def blue(obj, display_type="plain"):
    '''蓝色'''
    return colorful(obj, color='blue', display_type=display_type)


def purple(obj, display_type="plain"):
    '''紫色'''
    return colorful(obj, color='purple', display_type=display_type)


def cyan(obj, display_type="plain"):
    '''洋红'''
    return colorful(obj, color='cyan', display_type=display_type)


def white(obj, display_type="plain"):
    '''白色'''
    return colorful(obj, color='white', display_type=display_type)


def log_level(string:str, level:Union[int, str]=0, verbose:int=1):
    '''在字符串前面加上有颜色的[INFO][WARNING][ERROR]字样'''
    if level in {0, 'i', 'info', 'INFO'}:
        res = log_info(string, verbose)
    elif level in {1, 'w', 'warn', 'warning', 'WARN', 'WARNING'}:
        res = log_warn(string, verbose)
    elif level == {2, 'e', 'error', 'ERROR'}:
        res = log_error(string, verbose)
    elif level == 1:
        res = string
    return res
info_level_prefix = log_level


def log_free(string:str, prefix:str, string_color:str=None, prefix_color:str='yellow', verbose:int=1):
    '''自由log'''
    if string_color:
        string = colorful(string, color=string_color)
    if prefix:
        prefix = colorful(prefix, color=prefix_color)
        string = prefix + ' ' + string
    
    if verbose != 0:
        print(string)
    return string


def log_info(string:str, verbose:int=1):
    '''[INFO]: message, 绿色前缀'''
    res = colorful('[INFO]', color='green') + ' ' + string.strip()
    if verbose != 0:
        print(res)
    return res


@functools.lru_cache(None)
def log_info_once(string:str, verbose=1):
    ''' 单次warning '''
    return log_info(string, verbose)


def log_warn(string:str, verbose:int=1):
    '''[WARNING]: message, 黄色前缀'''
    res = colorful('[WARNING]', color='yellow') + ' ' + string.strip()
    if verbose != 0:
        print(res)
    return res


@functools.lru_cache(None)
def log_warn_once(string:str, verbose=1):
    ''' 单次warning '''
    return log_warn(string, verbose)


def log_error(string:str, verbose:int=1):
    '''[ERROR]: message, 红色前缀'''
    res = colorful('[ERROR]', color='red') + ' ' + string.strip()
    if verbose != 0:
        print(res)
    return res


@functools.lru_cache(None)
def log_error_once(string:str, verbose=1):
    ''' 单次warning '''
    return log_error(string, verbose)


@functools.lru_cache(None)
def print_once(string:str):
    '''单次打印'''
    print(string)


class SimpleStreamFileLogger(object):
    '''同时在和命令行和日切片中log（简单场景下可用）
    1. 会按照天进行切片, 适用于部署一些服务的时候长时间log日志的情况
    2. 缺点是log时候不能定位到文件和行号

    :param log_path: str, log文件的地址, 如'/home/logs/demo.log'
    :param date_format: str, date在路径中的位置, 可选subdir|prefix|suffix, subdir会以'/home/logs/20231218/demo.log'保存
    '''
    def __init__(self, log_path:str, date_format:str='subdir', level:str='DEBUG', 
                 format:str="[%(asctime)s][%(levelname)s] %(message)s"):
        self.log_path = log_path
        self.date_format = date_format
        self.level = level
        self.format = format

        self.save_dir = os.path.dirname(log_path)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.file_name = os.path.basename(log_path)
        self.initialize()

    def initialize(self):
        self.date = str(datetime.date.today()).replace('-','')
        # 保存路径
        if self.date_format == 'subdir':
            self.file = os.path.join(self.save_dir, self.date, self.file_name)
        elif self.date_format == 'prefix':
            self.file = os.path.join(self.save_dir, self.date + '_' + self.file_name)
        elif self.date_format == 'suffix':
            self.file = os.path.join(self.save_dir, self.file_name + '_' + self.date)
        else:
            raise ValueError('Args `date_format` only support subdir|prefix|suffix.')

        if os.path.dirname(self.file) != '':
            os.makedirs(os.path.dirname(self.file), exist_ok=True)

        # 创建日志器
        self.log = logging.getLogger(self.file_name)

        # 日志器默认级别
        level_dict = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 
                      'ERROR': logging.ERROR, 'CRITICAL':logging.CRITICAL}
        self.log.setLevel(level_dict[self.level])

        # 日志格式器
        self.formatter = logging.Formatter(self.format)
        self.log.handlers.clear()

        # 终端流输出
        stream_handle = logging.StreamHandler()
        self.log.addHandler(stream_handle)
        stream_handle.setLevel(level_dict[self.level])
        stream_handle.setFormatter(self.formatter)

        # 文件流输出
        file_handle = logging.FileHandler(filename=self.file, mode='a', encoding='utf-8')
        self.log.addHandler(file_handle)
        file_handle.setLevel(level_dict[self.level])
        file_handle.setFormatter(self.formatter)
        return self.log
    
    def reinitialize(self):
        if str(datetime.date.today()).replace('-','') != self.date:
            del self.log
            self.initialize()

    def info(self, text):
        self.reinitialize()
        self.log.info(text)
    
    def warn(self, text):
        self.reinitialize()
        self.log.warning(text)
    
    def warning(self, text):
        self.reinitialize()
        self.log.warning(text)

    def error(self, text):
        self.reinitialize()
        self.log.error(text)


class LoggerHandler(logging.Logger):
    '''同时在文件中和命令行中log（推荐）
    1. 适用于部署一些服务的时候长时间log日志的情况, 可按照文件大小, 日期进行切片
    2. 可以定位到对应的文件和行号

    :param log_path: str, log到文件的路径
    :param handles: str/tuple/list/set, handles的类型, 默认为('StreamHandler', 'FileHander')
    - StreamHandler 命令行输出
    - FileHander 文件数据
    - RotatingFileHandler 按照文件size切片的输出
    - TimedRotatingFileHandler 按照时间的切片输出

    :handle_config: dict, handles使用到的config
    :param level: str, DEBUG/INFO/WARNING/ERROR/CRITICAL, 指定log的level
    '''
    def __init__(self, log_path:str=None, handles=None, handle_config=None, name:str='root', 
                 level:str='DEBUG', format:str="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"):
        super().__init__(name)
        # 日志器默认级别
        level_dict = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 
                      'ERROR': logging.ERROR, 'CRITICAL':logging.CRITICAL}
        self.setLevel(level_dict[level])

        if handles is None:
            handles = ('StreamHandler', 'FileHander')
        elif isinstance(handles, str):
            handles = [handles]
        assert isinstance(handles, (tuple, set, list)), 'Args `handles` should be str/tuple/list/set'

        # 日志格式器
        formatter = logging.Formatter(format)
        self.handlers.clear()

        # 终端流输出
        if 'StreamHandler' in handles:
            stream_handle = logging.StreamHandler()
            self.addHandler(stream_handle)
            stream_handle.setLevel(level_dict[level])
            stream_handle.setFormatter(formatter)

        # 文件流输出
        if log_path is None:
            return
        if os.path.dirname(log_path) != '':
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if 'FileHander' in handles:
            file_handle = logging.FileHandler(filename=log_path, mode='a', encoding='utf-8')
        elif 'RotatingFileHandler' in handles:
            if handle_config is None:
                handle_config = {'maxBytes': 1024*1024*10, 'backupCount': 30, 'encoding': 'utf-8'}
            from logging.handlers import RotatingFileHandler
            file_handle = RotatingFileHandler(log_path, **handle_config)
        elif 'TimedRotatingFileHandler' in handles:
            if handle_config is None:
                handle_config = {'when': 'midnight', 'backupCount': 30, 'encoding': 'utf-8'}
            from logging.handlers import TimedRotatingFileHandler
            file_handle = TimedRotatingFileHandler(log_path, **handle_config)
        else:
            return
        self.addHandler(file_handle)
        file_handle.setLevel(level_dict[level])
        file_handle.setFormatter(formatter)


def json_flat(config:dict, sep='.'):
    '''把嵌套的字典flat化
    Examples:
    ```python
    >>> config = {'train_batch_size': 2,
    ...         "optimizer": {
    ...             "type": "AdamW",
    ...             "params": {
    ...                 "lr": 5e-4,
    ...                 "betas": [0.8, 0.999],
    ...                 "eps": 1e-8,
    ...                 "weight_decay": 3e-7
    ...             }
    ...         }
    ...         }
    >>> print_table(flat_config(config), headers=['config_name', 'config_value'])
    ... # +----------------------------------------------------+
    ... # | config_name                         | config_value |
    ... # +----------------------------------------------------+
    ... # | train_batch_size                    | 2            |
    ... # | optimizer -> type                   | AdamW        |
    ... # | optimizer -> params -> lr           | 0.0005       |
    ... # | optimizer -> params -> betas        | [0.8, 0.999] |
    ... # | optimizer -> params -> eps          | 1e-08        |
    ... # | optimizer -> params -> weight_decay | 3e-07        |
    ... # +----------------------------------------------------+
    ```
    '''
    res = []
    def _flat_config(config, pre_k=''):
        for k, v in config.items():
            key = k if pre_k == '' else pre_k + sep + k
            if isinstance(v, dict):
                _flat_config(v, key)
            else:
                res.append([key, v])
    _flat_config(config)
    return res


def print_table(data:Union[List, List[List], List[Dict]], headers:List=None):
    '''格式化打印表格，不依赖第三方包

    Examples:
    ```python
    >>> # 示例数据  
    >>> data = [  
    ...     [1, "Alice", 25],  
    ...     [2, "Bob", 30],  
    ...     [3, "Charlie", 35]  
    ... ]  
    >>> headers = ["ID", "Name", "Age"]  
    >>> # 打印表格  
    >>> print_table(data, headers)
    ... # 结果输出：
    ... # +--------------------+
    ... # | ID | Name    | Age |
    ... # +--------------------+
    ... # | 1  | Alice   | 25  |
    ... # | 2  | Bob     | 30  |
    ... # | 3  | Charlie | 35  |
    ... # +--------------------+
    ```
    '''
    def print_width(s:str):  
        '''字符串的打印宽度'''
        width = 0  
        # 遍历字符串中的每个字符  
        for char in s:  
            width += 2 if has_full_char(char) else 1
        return width
    
    def has_full_char(s:str):
        '''是否包含中文字符'''
        for char in s:  
            if '\u4e00' <= char <= '\u9fff' or '\uff00' <= char <= '\uffef':
                return True
        return False

    assert isinstance(data, list), 'Args `data` only accept list format'
    if isinstance(data[0], dict):
        headers = list(data[0].keys())
        data = [list(i.values()) for i in data]

    # 获取列的最大宽度
    max_widths = [max(len(str(row[i])) for row in data) for i in range(len(data[0]))]
    max_print_widths = [max(print_width(str(row[i])) for row in data) for i in range(len(data[0]))]
    
    # 打印表头  
    if headers:
        max_widths = [max(i, len(j)) for i, j in zip(max_widths, headers)]
        max_print_widths = [max(i, print_width(j)) for i, j in zip(max_print_widths, headers)]
        row_to_print = ' | '.join(str(header).ljust(max_widths[i] if has_full_char(str(header)) 
                                                    else max_print_widths[i]) for i, header in enumerate(headers))  
        print(f'+{"-".join("-" * (w + 2) for w in max_print_widths)}+')  
        print(f'| {row_to_print} |')  
    print(f'+{"-".join("-" * (w + 2) for w in max_print_widths)}+')        
      
    # 打印数据行  
    for row in data:  
        row_to_print = ' | '.join(str(item).ljust(max_widths[i] if has_full_char(str(item)) 
                                                  else max_print_widths[i]) for i, item in enumerate(row))  
        print(f'| {row_to_print} |')  
      
    # 打印表格底部边界  
    print(f'+{"-".join("-" * (w + 2) for w in max_print_widths)}+')
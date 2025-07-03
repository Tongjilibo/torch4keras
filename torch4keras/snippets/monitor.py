from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import os
import traceback
import copy
from .log import log_info, log_warn, log_error, colorful
from .import_utils import is_torch_available, is_xpu_available, is_npu_available, is_mps_available
import datetime


if is_torch_available():
    import torch


def format_timestamp(timestamp, format='%Y-%m-%d %H:%M:%S', verbose=0):
    '''格式化显示时间戳'''
    dt_object = datetime.datetime.fromtimestamp(timestamp)  
    
    # 格式化 datetime 对象  
    formatted_time = dt_object.strftime(format)  
    
    if verbose > 0:
        print(formatted_time)
    return formatted_time


def format_time(eta:Union[int, float], hhmmss=True):
    '''格式化显示时间间隔
    :param hhmmss: bool, 是否只以00:00:00格式显示
    '''
    # 以00:00:00格式显示
    if hhmmss:
        if eta > 86400:  # 1d 12:10:36
            eta_d, eta_h = eta // 86400, eta % 86400
            eta_format = '%dd ' % eta_d + ('%d:%02d:%02d' % (eta_h // 3600, (eta_h % 3600) // 60, eta_h % 60))
        elif eta > 3600:  # 12:10:36
            eta_format = ('%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60))
        else:  # 10:36
            eta_format = '%d:%02d' % (eta // 60, eta % 60)

    else:
        if eta > 86400:  # 1d 12h 10m 36s
            eta_d, eta_h = eta // 86400, eta % 86400
            eta_format = '%dd %dh %dm %ds' % (eta_d, eta_h // 3600, (eta_h % 3600) // 60, eta_h % 60)
        elif eta > 3600:  # 12h 10m 36s
            eta_format = ('%dh %dm %ds' % (eta // 3600, (eta % 3600) // 60, eta % 60))
        elif eta > 60:  # 10m 36s
            eta_format = ('%dm %ds' % (eta // 60, eta % 60))
        elif (eta >= 1) and (eta <= 60):  # 36.02s
            eta_format = '%.2fs' % eta
        elif eta >= 1e-3:  # 25ms
            eta_format = '%.0fms' % (eta * 1e3)
        else:  # 250us
            eta_format = '%.0fus' % (eta * 1e6)
    return eta_format


def timeit(func):
    '''装饰器, 计算函数消耗的时间
    
    Examples:
    ```python
    >>> @timeit
    >>> def main(n=10):
    ...     for i in range(n):
    ...         time.sleep(0.01)

    >>> main(10)
    ```
    '''
    def warpper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        consume = format_time(end - start, hhmmss=False)
        start1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
        end1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))

        log_info(f'Function `{func.__name__}` cost {colorful(consume)} [{start1} < {end1}]')
        return res
    return warpper


class TimeitContextManager:
    '''上下文管理器, 记录耗时/平均耗时

    Examples:
    ```python
    >>> from torch4keras.snippets import TimeitContextManager
    >>> with TimeitContextManager() as ti:
    ...     for i in range(10):
    ...         time.sleep(0.1)
    ...         # ti.lap(name=i, reset=False)  # 统计累计耗时
    ...         # ti.lap(name=i, reset=True)  # 统计间隔耗时
    ...         # ti.lap(count=10, name=i, restart=True)  # 统计每段速度
    ...     # ti(10) # 统计速度
    ```
    '''
    def __enter__(self):
        self.count = None
        self.global_start_tm = self.start_tm = time.time()
        return self

    def __call__(self, count):
        self.count = count

    def reset(self):
        '''自定义开始记录的地方'''
        self.start_tm = time.time()
    
    def lap(self, name:str=None, count:int=None, reset:bool=False):
        '''
        :params name: 打印时候自定义的前缀
        :params count: 需要计算平均生成速度中统计的次数
        :params reset: 是否重置start_tm, True只记录时间间隔, 否则记录的是从一开始的累计时间
        '''
        if count is not None:
            self.count = count
        name = '' if name is None else str(name).strip()

        end_tm = time.time()
        # 如果是reset则使用上一轮重置后的self.start_tm, 否则使用一开始的self.global_start_tm
        if reset:
            start_tm = self.start_tm
            prefix = 'Interval'
        else:
            start_tm = self.global_start_tm
            prefix = 'Accumulated'
        
        consume = end_tm - start_tm
        start1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_tm))
        end1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_tm))
        if self.count is None:  # 只记录耗时
            consume = format_time(consume, hhmmss=False)
            log_info(f'{name} {prefix} Cost {colorful(consume)} [{start1} < {end1}]')
        elif consume > 0:  # 记录平均耗时
            speed = f'{self.count / consume:.2f}/s'
            log_info(f'{name} {prefix} Average speed {colorful(speed)} [{start1} < {end1}]')
        else:
            pass
            # log_warn('Time duration = 0')
        
        if reset:
            self.reset()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lap(name='Total', reset=False)


class TimeitLogger:
    '''记录耗时

    Examples:
    ```python
    >>> ti = TimeitLogger()
    >>> for i in range(10):
    ...     time.sleep(0.1)
    ...     ti.lap(name=i)
    >>> ti.end() # 打印各个步骤时长
    ```
    '''
    def __init__(self):
        self.reset()

    def __call__(self, *args, **kwargs):
        self.lap(*args, **kwargs)

    def reset(self):
        '''自定义开始记录的地方'''
        self.cost = dict()
        self.count = dict()
        self.start_tm = time.time()

    def restart(self):
        self.start_tm = time.time()

    def lap(self, name:str, verbose=0):
        '''
        :params name: 打印时候自定义的前缀
        '''
        end_tm = time.time()
        consume = end_tm - self.start_tm
        name = str(name)
        self.cost[name] = self.cost.get(name, 0) + consume
        self.count[name] = self.count.get(name, 0) + 1
        self.start_tm = time.time()
        
        if verbose > 1:
            start1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_tm))
            end1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_tm))
            log_info(f'{name} Cost {colorful(consume)} [{start1} < {end1}]')

    def end(self, verbose=1):
        for k, v in self.count.items():
            if v > 1:
                self.cost['avg_' + k] = self.cost[k] / v
        
        if verbose > 0:
            log_info(f'Cost detail: {self.cost}')
            self.reset()
        else:
            cost = copy.deepcopy(self.cost)
            self.reset()
            return cost


def send_email(mail_receivers:Union[str,list], mail_subject:str, mail_msg:str="", mail_host:str=None, 
               mail_user:str=None, mail_pwd:str=None, mail_sender:str=None):
    ''' 发送邮件(默认使用笔者自己注册的邮箱, 若含敏感信息请使用自己注册的邮箱)

    :param mail_subject: str, 邮件主题
    :param mail_msg: str, 邮件正文
    :param mail_receivers: str/list, 收件人邮箱
    :param mail_host: str, 发件服务器host
    :param mail_user: str, 发件人
    :param mail_pwd: str, smtp的第三方密码
    :param mail_sender: str, 发件人邮箱
    '''
    import smtplib
    from email.mime.text import MIMEText

    mail_host = mail_host or 'smtp.163.com'
    mail_user = mail_user or 'bert4torch'
    mail_pwd = mail_pwd or 'VDSGQEHFXDZOCVEH'
    mail_sender = mail_sender or 'bert4torch@163.com'

    #构造邮件内容
    message = MIMEText(mail_msg,'plain','utf-8')
    message['Subject'] = mail_subject
    message['From'] = mail_sender
    assert isinstance(mail_receivers, (str, tuple, list)), 'Arg `receivers` only receive `str, tuple, list` format'
    message['To'] = mail_receivers if isinstance(mail_receivers, str) else ';'.join(mail_receivers)

    #登录并发送邮件
    try:
        smtpObj = smtplib.SMTP() 
        smtpObj.connect(mail_host, 25)  # 连接到服务器
        smtpObj.login(mail_user, mail_pwd)  # 登录到服务器
        smtpObj.sendmail(mail_sender, mail_receivers, message.as_string())  # 发送
        smtpObj.quit()  # 退出
        log_info('Send email success')
    except smtplib.SMTPException as e:
        log_error('Send email error : '+str(e))
        return str(e)


def email_when_error(receivers:Union[str,list], **configs):
    '''装饰器, 异常则发邮件
    Examples:
    ```python
    >>> @email_when_error(receivers='tongjilibo@163.com')
    >>> def test():
    ...     return 1/0
    >>> test()  # 调用
    ```
    '''
    def actual_decorator(func):
        def new_func(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                error_msg = traceback.format_exc()
                send_email(receivers, func.__name__, error_msg, **configs)
                raise e
            return res
        return new_func
    return actual_decorator


def watch_system_state(log_dir:str, gpu_id_list:List[int]=None, pids:Union[int,List[int]]=None, interval=1):
    '''监控system的状态
    
    :param log_dir: str, tensorboard的地址
    :param gpu_id_list: List[int], 监控的gpu
    :param pids: int/List[int], 监控的进程号

    Examples:
    ```python
    >>> watch_system_state(log_dir='./system_states')
    ```
    '''
    import psutil
    import pynvml
    from tensorboardX import SummaryWriter

    pynvml.nvmlInit()
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(log_dir))  # prepare summary writer

    pre_time_int = int(time.time())
    pre_time = time.time()
    pids = pids or os.getpid()
    pids = [pids] if isinstance(pids, int) else pids
    G = 1024*1024*1024

    pre_read = {pid:psutil.Process(pid).io_counters().read_bytes for pid in pids}
    pre_write = {pid:psutil.Process(pid).io_counters().write_bytes for pid in pids}
    time_str_init = int(time.time())

    log_info("Watching System Info")
    while True:
        time.sleep(interval)
        now_time = time.time()
        time_str = int(now_time) - time_str_init
        for pid in pids:
            p = psutil.Process(pid)

            # CPU使用情况
            tb_writer.add_scalar(f"Pid_{pid}/CPU Percent", p.cpu_percent(interval=0.5), time_str)

            # 内存使用情况
            tb_writer.add_scalar(f"Pid_{pid}/Memory Percent", p.memory_percent(), time_str)
            tb_writer.add_scalar(f"Pid_{pid}/Memory RSS G_byte", p.memory_info().rss/G, time_str)
            tb_writer.add_scalar(f"Pid_{pid}/Memory VMS G_byte", p.memory_info().vms/G, time_str)

            # 进程IO信息
            data = p.io_counters()
            tb_writer.add_scalar(f"Pid_{pid}/IO read M_byte", data.read_bytes/1024, time_str)
            tb_writer.add_scalar(f"Pid_{pid}/IO write M_byte", data.write_bytes/1024, time_str)
            if (time_str - pre_time_int)%5 == 0:
                tb_writer.add_scalar(f"Pid_{pid}/IO readRate M_byte_s", (data.read_bytes - pre_read[pid])/(now_time-pre_time)/1024, time_str)
                tb_writer.add_scalar(f"Pid_{pid}/IO writeRate M_byte_s", (data.write_bytes - pre_write[pid])/(now_time-pre_time)/1024, time_str)
                pre_read[pid] = data.read_bytes
                pre_write[pid] = data.write_bytes
            pre_time = now_time
        
        # gpu使用情况
        tesorboard_info_list = []
        if gpu_id_list is None:
            deviceCount = pynvml.nvmlDeviceGetCount()
            device_list = [i for i in range(deviceCount)]
        else:
            device_list = gpu_id_list

        tesorboard_info_list = []
        for i in device_list:
            tesorboard_info = {}
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            tesorboard_info['gpu_name'] = pynvml.nvmlDeviceGetName(handle)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            tesorboard_info['gpu_mem_used'] = meminfo.used/G
            UTIL = pynvml.nvmlDeviceGetUtilizationRates(handle)
            tesorboard_info['gpu_gpu_util'] = UTIL.gpu
            tesorboard_info['gpu_mem_util'] = UTIL.memory
            tesorboard_info_list.append(copy.deepcopy(tesorboard_info))
        for tesorboard_info,i in zip(tesorboard_info_list, device_list):
            tb_writer.add_scalar(f'GPU/GPU{i} mem_used unit_G', tesorboard_info['gpu_mem_used'], time_str)
            tb_writer.add_scalar(f'GPU/GPU{i} gpu_util', tesorboard_info['gpu_gpu_util'], time_str)
            tb_writer.add_scalar(f'GPU/GPU{i} mem_util', tesorboard_info['gpu_mem_util'], time_str)

    tb_writer.close()
    pynvml.nvmlShutdown()

    return


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:

    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    mem_size = 0
    err_msg = (
        f"`size` {size} is not in a valid format. Use an integer for bytes, or a string with an unit (like '5.0GB')."
    )
    try:
        if isinstance(size, int):
            mem_size = size
        elif size.upper().endswith("GIB"):
            mem_size = int(float(size[:-3]) * (2**30))
        elif size.upper().endswith("MIB"):
            mem_size = int(float(size[:-3]) * (2**20))
        elif size.upper().endswith("KIB"):
            mem_size = int(float(size[:-3]) * (2**10))
        elif size.upper().endswith("GB"):
            int_size = int(float(size[:-2]) * (10**9))
            mem_size = int_size // 8 if size.endswith("b") else int_size
        elif size.upper().endswith("MB"):
            int_size = int(float(size[:-2]) * (10**6))
            mem_size = int_size // 8 if size.endswith("b") else int_size
        elif size.upper().endswith("KB"):
            int_size = int(float(size[:-2]) * (10**3))
            mem_size = int_size // 8 if size.endswith("b") else int_size
    except ValueError:
        raise ValueError(err_msg)

    if mem_size <= 0:
        raise ValueError(err_msg)
    return mem_size


def get_max_memory(max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None):
    """
    Get the maximum memory available if nothing is passed, converts string to int otherwise.
    """
    import psutil

    if max_memory is None:
        if not (torch.cuda.is_available() or is_npu_available() or is_xpu_available()):
            max_memory = {}

        else:
            # Make sure CUDA is initialized on each GPU to have the right memory info.
            if is_npu_available():
                for i in range(torch.npu.device_count()):
                    _ = torch.tensor(0, device=torch.device("npu", i))
                max_memory = {i: torch.npu.mem_get_info(i)[0] for i in range(torch.npu.device_count())}
            elif is_xpu_available():
                for i in range(torch.xpu.device_count()):
                    _ = torch.tensor(0, device=torch.device("xpu", i))
                max_memory = {i: torch.xpu.max_memory_allocated(i) for i in range(torch.xpu.device_count())}
            else:
                for i in range(torch.cuda.device_count()):
                    _ = torch.tensor([0], device=i)
                max_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}
        # allocate everything in the mps device as the RAM is shared
        if is_mps_available():
            max_memory["mps"] = psutil.virtual_memory().available
        else:
            max_memory["cpu"] = psutil.virtual_memory().available
        return max_memory

    for key in max_memory:
        if isinstance(max_memory[key], str):
            max_memory[key] = convert_file_size_to_int(max_memory[key])

    # Need to sort the device by type to make sure that we allocate the gpu first.
    # As gpu/npu/xpu are represented by int, we need to sort them first.
    gpu_devices = [k for k in max_memory.keys() if isinstance(k, int)]
    gpu_devices.sort()
    # check if gpu/npu/xpu devices are available and if not, throw a warning
    if is_npu_available():
        num_devices = torch.npu.device_count()
    elif is_xpu_available():
        num_devices = torch.xpu.device_count()
    else:
        num_devices = torch.cuda.device_count()
    for device in gpu_devices:
        if device >= num_devices or device < 0:
            log_warn(f"Device {device} is not available, available devices are {list(range(num_devices))}")
    # Add the other devices in the preset order if they are available
    all_devices = gpu_devices + [k for k in ["mps", "cpu", "disk"] if k in max_memory.keys()]
    # Raise an error if a device is not recognized
    for k in max_memory.keys():
        if k not in all_devices:
            raise ValueError(
                f"Device {k} is not recognized, available devices are integers(for GPU/XPU), 'mps', 'cpu' and 'disk'"
            )
    max_memory = {k: max_memory[k] for k in all_devices}

    return max_memory
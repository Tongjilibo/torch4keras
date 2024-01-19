from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import os
import traceback
import copy
import functools
from .log import log_info, log_warn, log_error


def format_time(eta, hhmmss=True):
    '''格式化显示时间
    :param hhmmss: bool, 是否只以00:00:00格式显示
    '''
    if eta > 86400:
        eta_d, eta_h = eta // 86400, eta % 86400
        eta_format = f'{eta_d}d ' + ('%d:%02d:%02d' % (eta_h // 3600, (eta_h % 3600) // 60, eta_h % 60))
    elif eta > 3600:
        eta_format = ('%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60))
    elif hhmmss:
        eta_format = '%d:%02d' % (eta // 60, eta % 60)
    elif (eta >= 1) and (eta < 60):
        eta_format = '%.2fs' % eta
    elif eta >= 1e-3:
        eta_format = '%.0fms' % (eta * 1e3)
    else:
        eta_format = '%.0fus' % (eta * 1e6)
    return eta_format


def cost_time(func):
    '''装饰器，计算函数消耗的时间
    '''
    def warpper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        consume = format_time(end - start, hhmmss=False)
        start1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
        end1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))

        print(f'Cost {consume}\t{start1} ~ {end1}')
        return res
    return warpper


def send_email(mail_receivers:Union[str,list], mail_subject:str, mail_msg:str="", mail_host:str=None, 
               mail_user:str=None, mail_pwd:str=None, mail_sender:str=None):
    ''' 发送邮件(默认使用笔者自己注册的邮箱，若含敏感信息请使用自己注册的邮箱)

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


def monitor_run_by_email(func, mail_receivers:Union[str,list]=None, mail_subject:str=None, mail_host:str=None, 
                         mail_user:str=None, mail_pwd:str=None, mail_sender:str=None):
    """ 通过发邮件的形式监控运行，在程序出现异常的时候发邮件
    """
    @functools.wraps(func)
    def get_except(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            error_msg = traceback.format_exc()
            mail_receivers_ = mail_receivers or kwargs.get('mail_receivers')
            if mail_receivers_ is not None:
                mail_subject_ = mail_subject or kwargs.get('mail_subject') or "[ERROR] " + func.__name__
                mail_host_ = mail_host or kwargs.get('mail_host')
                mail_user_ = mail_user or kwargs.get('mail_user')
                mail_pwd_ = mail_pwd or kwargs.get('mail_pwd')
                mail_sender_ = mail_sender or kwargs.get('mail_sender')
                send_email(mail_receivers_, mail_subject_, error_msg, mail_host=mail_host_, 
                           mail_user=mail_user_, mail_pwd=mail_pwd_, mail_sender=mail_sender_)
            raise e
    return get_except


def email_when_error(receivers:Union[str,list], **configs):
    '''装饰器，异常则发邮件
    Example:
    --------
    @email_when_error(receivers='tongjilibo@163.com')
    def test():
        return 1/0
    test()  # 调用
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


def watch_system_state(log_dir:str, gpu_id_list=None, pid=None):
    '''监控system的状态
    
    :param log_dir: str, tensorboard的地址
    :param gpu_id_list: List[int], 监控的gpu
    :param pid: int, 监控的进程号
    '''
    import psutil
    import pynvml
    from tensorboardX import SummaryWriter

    pynvml.nvmlInit()
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(log_dir))  # prepare summary writer

    pre_time_int = int(time.time())
    pre_time = time.time()
    pid = None or os.getpid()
    p = psutil.Process(pid)
    G = 1024*1024*1024

    pre_read = p.io_counters().read_bytes
    pre_write = p.io_counters().write_bytes
    time_str_init = int(time.time())

    log_info("Watching System Info")
    while True:
        time.sleep(1)
        now_time = time.time()
        time_str = int(now_time) - time_str_init
        p = psutil.Process(pid)

        # CPU使用情况
        tb_writer.add_scalar(f"CPU_pid_{pid}/cpu_percent", p.cpu_percent(interval=0.5), time_str)

        # 内存使用情况
        tb_writer.add_scalar(f"Memory_pid_{pid}/memory_percent", p.memory_percent(), time_str)
        tb_writer.add_scalar(f"Memory_pid_{pid}/RSS G_byte", p.memory_info().rss/G, time_str)
        tb_writer.add_scalar(f"Memory_pid_{pid}/VMS G_byte", p.memory_info().vms/G, time_str)

        # 进程IO信息
        data = p.io_counters()
        tb_writer.add_scalar(f"IO_pid_{pid}/read M_byte", data.read_bytes/1024, time_str)
        tb_writer.add_scalar(f"IO_pid_{pid}/write M_byte", data.write_bytes/1024, time_str)
        if (time_str - pre_time_int)%5 == 0:
            tb_writer.add_scalar(f"IO_pid_{pid}/readRate M_byte_s", (data.read_bytes - pre_read)/(now_time-pre_time)/1024, time_str)
            tb_writer.add_scalar(f"IO_pid_{pid}/writeRate M_byte_s", (data.write_bytes - pre_write)/(now_time-pre_time)/1024, time_str)
            pre_read = data.read_bytes
            pre_write = data.write_bytes
            pre_time = now_time

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
            tb_writer.add_scalar(f'GPU{i}/gpu_mem_used unit_G', tesorboard_info['gpu_mem_used'], time_str)
            tb_writer.add_scalar(f'GPU{i}/gpu_gpu_util', tesorboard_info['gpu_gpu_util'], time_str)
            tb_writer.add_scalar(f'GPU{i}/gpu_mem_util', tesorboard_info['gpu_mem_util'], time_str)

    tb_writer.close()
    pynvml.nvmlShutdown()

    return
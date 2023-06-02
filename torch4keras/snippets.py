import numpy as np
import torch
from torch import nn, Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import inspect
from packaging import version
from torch.utils.data import Dataset, IterableDataset
import os
import random
import traceback

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None
        
def take_along_dim(input_tensor, indices, dim=None):
    '''兼容部分低版本pytorch没有torch.take_along_dim
    '''
    if version.parse(torch.__version__) > version.parse('1.8.1'):
        return torch.take_along_dim(input_tensor, indices, dim)
    else:
        # 该逻辑仅在少量数据上测试，如有bug，欢迎反馈
        if dim is None:
            res = input_tensor.flatten()[indices]
        else:
            res = np.take_along_axis(input_tensor.cpu().numpy(), indices.cpu().numpy(), axis=dim)
            res = torch.from_numpy(res).to(input_tensor.device)
        # assert res.equal(torch.take_along_dim(input_tensor, indices, dim))
        return res


def torch_div(input, other, rounding_mode=None):
    ''' torch.div兼容老版本
    '''
    if version.parse(torch.__version__) < version.parse('1.7.2'):
        indices = input // other  # 兼容老版本
    else:
        indices = torch.div(input, other, rounding_mode=rounding_mode)  # 行索引
    return indices


def softmax(x, axis=-1):
    """numpy版softmax
    """
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)


def search_layer(model, layer_name, retrun_first=True):
    '''根据layer_name搜索并返回参数/参数list
    '''
    return_list = []
    for name, param in model.named_parameters():
        if param.requires_grad and layer_name in name:
            return_list.append(param)
    if len(return_list) == 0:
        return None
    if retrun_first:
        return return_list[0]
    else:
        return return_list


class ListDataset(Dataset):
    '''数据是List格式Dataset，支持传入file_path或者外部已读入的data(List格式)

    :param file_path: str, 待读取的文件的路径，若无可以为None
    :param data: List[Any], list格式的数据，和file_path至少有一个不为None
    '''
    def __init__(self, file_path=None, data=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, tuple, list)):
            self.data = self.load_data(file_path)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path):
        return file_path


class IterDataset(IterableDataset):
    '''流式读取文件，用于大数据量、多小文件使用时候需要注意steps_per_epoch != None

    :param file_path: str, 待读取的文件的路径，若无可以为None
    '''
    def __init__(self, file_path=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, tuple, list)):
            self.file_path = file_path
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')
    
    def __iter__(self):
        return self.load_data(self.file_path)

    @staticmethod
    def load_data(file_path, verbose=0):
        if isinstance(file_path, (tuple, list)):
            for file in file_path:
                if verbose != 0:
                    print("Load data: ", file)
                with open(file, 'r') as file_obj:
                    for line in file_obj:
                        yield line
        elif isinstance(file_path, str):
            with open(file_path, 'r') as file_obj:
                for line in file_obj:
                    yield line


def metric_mapping(metric, func, y_pred, y_true):
    '''metric的计算

    :param metric: str, 自带metrics的名称
    :param func: function, 透传的用户自定的计算指标的函数
    :param y_pred: torch.Tensor, 样本的预测结果
    :param y_true: torch.Tensor, 样本的真实结果
    '''
    # 自定义metrics
    if inspect.isfunction(func):
        metric_res = func(y_pred, y_true)
        if inspect.isfunction(metric):
            # 如果直接传入回调函数（无key），要求回调函数返回Dict[String: Int/Float]类型
            assert isinstance(metric_res, dict), 'Custom metrics callbacks should return `Dict[String: Int/Float]` value'
        elif isinstance(metric, str):
            # 如果直接传入回调函数（有key），要求回调函数返回Int/Float类型
            assert isinstance(metric_res, (int, float)), 'Custom metrics callbacks should return `Int/Float value'
        return metric_res
    elif metric == 'loss':
        pass
    # 自带metrics
    elif isinstance(metric, str):
        # 如果forward返回了list, tuple，则选取第一项
        y_pred_tmp = y_pred[0] if isinstance(y_pred, (list, tuple)) else y_pred
        y_true_tmp = y_true[0] if isinstance(y_true, (list, tuple)) else y_true
        y_pred_tmp = y_pred_tmp.detach()  # 训练过程中评估，detach不进入计算图

        # 根据shape做预处理
        if len(y_pred_tmp.shape) == len(y_true_tmp.shape) + 1:
            y_pred_tmp = torch.argmax(y_pred_tmp, dim=-1)
        elif len(y_pred_tmp.shape) == len(y_true_tmp.shape):
            pass
        else:
            raise ValueError(f'y_pred_tmp.shape={y_pred_tmp.shape} while y_true_tmp.shape={y_true_tmp.shape}')

        # 执行内置的metric
        if metric in {'accuracy', 'acc'}:
            return torch.sum(y_pred_tmp.eq(y_true_tmp)).item() / y_true_tmp.numel()
        elif metric in {'auc'}:
            if roc_auc_score is None:
                raise ImportError('roc_auc_score requires the `sklearn` library.')
            return roc_auc_score(y_true.cpu().numpy(), y_pred_tmp.cpu().numpy())            
        elif metric in {'mae', 'MAE', 'mean_absolute_error'}:
            return torch.mean(torch.abs(y_pred_tmp - y_true_tmp)).item()
        elif metric in {'mse', 'MSE', 'mean_squared_error'}:
            return torch.mean(torch.square(y_pred_tmp - y_true_tmp)).item()
        elif metric in {'mape', 'MAPE', 'mean_absolute_percentage_error'}:
            diff = torch.abs((y_true_tmp - y_pred_tmp) / torch.clamp(torch.abs(y_true_tmp), 1e-7, None))
            return 100. * torch.mean(diff).item()
        elif metric in {'msle', 'MSLE', 'mean_squared_logarithmic_error'}:
            first_log = torch.log(torch.clamp(y_pred_tmp, 1e-7, None) + 1.)
            second_log = torch.log(torch.clamp(y_true_tmp, 1e-7, None) + 1.)
            return torch.mean(torch.square(first_log - second_log)).item()

    return None


def seed_everything(seed=None):
    '''固定seed
    
    :param seed: int, 随机种子
    '''
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed is None) or not (min_seed_value <= seed <= max_seed_value):
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    print(f"Global seed set to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def spend_time(func):
    '''装饰器，计算函数消耗的时间
    '''
    start = time.time()
    def warpper(*args, **kwargs):
        res = func(*args, **kwargs)
        end = time.time()
        consume = end - start
        start1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
        end1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
        print(f'{start1} ~ {end1}  spent {consume:.2f}s')
        return res
    return warpper


def send_email(receivers, subject, msg="", mail_host=None, mail_user=None, mail_pwd=None, sender=None):
    """ 发送邮件(默认使用笔者自己注册的邮箱，若含敏感信息请使用自己注册的邮箱)

    :param subject: str, 邮件主题
    :param msg: str, 邮件正文
    :param receivers: str/list, 收件人邮箱
    :param mail_host: str, 发件服务器host
    :param mail_user: str, 发件人
    :param mail_pwd: str, smtp的第三方密码
    :param sender: str, 发件人邮箱
    """
    import smtplib
    from email.mime.text import MIMEText

    mail_host = mail_host or 'smtp.163.com'
    mail_user = mail_user or 'bert4torch'
    mail_pwd = mail_pwd or 'VDSGQEHFXDZOCVEH'
    sender = sender or 'bert4torch@163.com'

    #构造邮件内容
    message = MIMEText(msg,'plain','utf-8')  
    message['Subject'] = subject
    message['From'] = sender     
    message['To'] = receivers[0]  

    #登录并发送邮件
    try:
        smtpObj = smtplib.SMTP() 
        smtpObj.connect(mail_host, 25)  # 连接到服务器
        smtpObj.login(mail_user, mail_pwd)  # 登录到服务器
        smtpObj.sendmail(sender, receivers, message.as_string())  # 发送
        smtpObj.quit()  # 退出
        print('[INFO] Send email success')
    except smtplib.SMTPException as e:
        error = info_level_prefix('Send email error : '+str(e), 2)
        print(error)
        return error


def email_when_error(receivers, **configs):
    """装饰器，异常则发邮件
    Example:
    --------
    @email_when_error(receivers='tongjilibo@163.com')
    def test():
        return 1/0
    test()  # 调用
    """
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
    

def colorful(obj, color="yellow", display_type="plain"):
    '''
    # 彩色输出格式：
    # 设置颜色开始 ：\033[显示方式;前景色;背景色m
    # 说明：
    # 前景色            背景色           颜色
    # ---------------------------------------
    # 30                40              黑色
    # 31                41              红色
    # 32                42              绿色
    # 33                43              黃色
    # 34                44              蓝色
    # 35                45              紫红色
    # 36                46              青蓝色
    # 37                47              白色
    # 显示方式           意义
    # -------------------------
    # 0                终端默认设置
    # 1                高亮显示
    # 4                使用下划线
    # 5                闪烁
    # 7                反白显示
    # 8                不可见
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


def info_level_prefix(string, level=0):
    '''在字符串前面加上有颜色的[INFO][WARNING][ERROR]字样'''
    if level in {0, 'i', 'info', 'INFO'}:
        res = colorful('[INFO]', color='green') + string
    elif level in {1, 'w', 'warn', 'warning', 'WARN', 'WARNING'}:
        res = colorful('[WARNING]', color='yellow') + string
    elif level == {2, 'e', 'error', 'ERROR'}:
        res = colorful('[ERROR]', color='red') + string
    elif level == 1:
        res = string
    return res

   
def print_trainable_parameters(module):
    """打印可训练的参数量"""
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
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


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

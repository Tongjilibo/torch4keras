import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from packaging import version
import inspect
from .import_utils import is_safetensors_available, is_sklearn_available, is_torch_available
import os


if is_torch_available():
    import torch
    from torch.utils.data import Dataset, IterableDataset
    from torch import nn, Tensor
    from torch.nn import Module
else:
    class Tensor: pass
    class Module: pass
    class Dataset: pass
    class IterableDataset: pass


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file


if is_sklearn_available():
    from sklearn.metrics import roc_auc_score
else:
    roc_auc_score = None


def take_along_dim(input_tensor:Tensor, indices:Tensor, dim:int=None):
    '''兼容部分低版本pytorch没有torch.take_along_dim
    '''
    if version.parse(torch.__version__) > version.parse('1.8.1'):
        return torch.take_along_dim(input_tensor, indices, dim)
    else:
        # 该逻辑仅在少量数据上测试, 如有bug, 欢迎反馈
        if dim is None:
            res = input_tensor.flatten()[indices]
        else:
            res = np.take_along_axis(input_tensor.cpu().numpy(), indices.cpu().numpy(), axis=dim)
            res = torch.from_numpy(res).to(input_tensor.device)
        # assert res.equal(torch.take_along_dim(input_tensor, indices, dim))
        return res


def torch_div(input:Tensor, other:Tensor, rounding_mode:Optional[str] = None):
    ''' torch.div兼容老版本
    '''
    if version.parse(torch.__version__) < version.parse('1.7.2'):
        indices = input // other  # 兼容老版本
    else:
        indices = torch.div(input, other, rounding_mode=rounding_mode)  # 行索引
    return indices


def softmax(x:np.ndarray, axis:int=-1):
    '''numpy版softmax
    '''
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)


def search_layer(model:Module, layer_name:str, retrun_first:bool=True):
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
    '''数据是List格式Dataset, 支持传入file_path或者外部已读入的data(List格式)

    :param file_path: str, 待读取的文件的路径, 若无可以为None
    :param data: List[Any], list格式的数据, 和file_path至少有一个不为None
    '''
    def __init__(self, file_path:Union[str, tuple, list]=None, data:Optional[list]=None, **kwargs):
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
    '''流式读取文件, 用于大数据量、多小文件使用时候需要注意steps_per_epoch != None

    :param file_path: str, 待读取的文件的路径, 若无可以为None
    '''
    def __init__(self, file_path:Union[str, tuple, list]=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, tuple, list)):
            self.file_path = file_path
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')
    
    def __iter__(self):
        return self.load_data(self.file_path)

    @staticmethod
    def load_data(file_path:Union[str,tuple,list], verbose=0):
        if isinstance(file_path, (tuple, list)):
            for file in file_path:
                if verbose != 0:
                    print("Load data: ", file)
                with open(file, 'r') as file_obj:
                    for line in file_obj:
                        yield line
        elif isinstance(file_path, str):
            if verbose != 0:
                print("Load data: ", file_path)
            with open(file_path, 'r') as file_obj:
                for line in file_obj:
                    yield line


def set_precision(num:float, dense_round:int=1):
    '''设置数字的精度
    '''
    if np.isinf(num):
        return num
    if abs(num) >= 10:
        return int(round(num))

    precision = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    for len_, prec in enumerate(precision):
        if abs(num) >= prec:
            return round(num, len_ + dense_round)
    return num


def metric_mapping(metric:str, func:Callable, y_pred:Union[Tensor, List[Tensor], Tuple[Tensor]], 
                   y_true:Union[Tensor, List[Tensor], Tuple[Tensor]]):
    '''metric的计算

    :param metric: str, 自带metrics的名称
    :param func: function, 透传的用户自定的计算指标的函数
    :param y_pred: Tensor, 样本的预测结果
    :param y_true: Tensor, 样本的真实结果
    '''
    # 自定义metrics
    if inspect.isfunction(func):
        metric_res = func(y_pred, y_true)
        if inspect.isfunction(metric):
            # 如果直接传入回调函数（无key）, 要求回调函数返回Dict[String: Int/Float]类型
            assert isinstance(metric_res, dict), 'Custom metrics callbacks should return `Dict[String: Int/Float]` value'
        elif isinstance(metric, str):
            # 如果直接传入回调函数（有key）, 要求回调函数返回Int/Float类型
            assert isinstance(metric_res, (int, float)), 'Custom metrics callbacks should return `Int/Float value'
        return metric_res
    elif metric == 'loss':
        pass
    # 自带metrics
    elif isinstance(metric, str):
        # 如果forward返回了list, tuple, 则选取第一项
        y_pred_tmp = y_pred[0] if isinstance(y_pred, (list, tuple)) else y_pred
        y_true_tmp = y_true[0] if isinstance(y_true, (list, tuple)) else y_true
        y_pred_tmp = y_pred_tmp.detach()  # 训练过程中评估, detach不进入计算图

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


def safe_torch_load(checkpoint:str, map_location='cpu'):
    '''安全加载torch checkpoint, 支持weights_only参数'''
    if 'weights_only' in inspect.signature(torch.load).parameters:
        return torch.load(checkpoint, map_location=map_location, weights_only=True)
    else:
        return torch.load(checkpoint, map_location=map_location)


def load_checkpoint(checkpoint:str, load_safetensors:bool=False):
    '''加载ckpt, 支持torch.load和safetensors
    '''
    if load_safetensors or checkpoint.endswith(".safetensors"):
        # 加载safetensors格式
        with safe_open(checkpoint, framework="pt") as f:
            metadata = f.metadata()
        
        if metadata is None:
            pass
        elif metadata.get("format") not in ["pt", "tf", "flax"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        elif metadata["format"] != "pt":
            raise NotImplementedError(
                f"Conversion from a {metadata['format']} safetensors archive to PyTorch is not implemented yet."
            )
        return safe_load_file(checkpoint)
    else:
        # 正常加载pytorch_model.bin
        return safe_torch_load(checkpoint)


def save_checkpoint(state_dict:dict, save_path:str, save_safetensors:bool=False):
    '''保存ckpt, 支持torch.save和safetensors
    '''
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if save_safetensors or save_path.endswith('.safetensors'):
        safe_save_file(state_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(state_dict, save_path)
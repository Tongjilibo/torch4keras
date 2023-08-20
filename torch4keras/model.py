import torch
from torch import nn
from torch4keras.trainer import *


class BaseModel(Trainer, nn.Module):
    """BaseModel, 使用继承的方式来使用
    """
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        Trainer.__init__(self, *args, **kwargs)
        

class BaseModelDP(nn.DataParallel, BaseModel):
    '''DataParallel模式使用多gpu的方法, 父类顺序颠倒也会出问题
    '''
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self)
        nn.DataParallel.__init__(self, *args, **kwargs)


class BaseModelDDP(nn.parallel.DistributedDataParallel, BaseModel):
    '''DistributedDataParallel模式使用多gpu的方法, 父类顺序颠倒也会出问题
    '''
    def __init__(self, *args, master_rank=0, **kwargs):
        BaseModel.__init__(self)
        nn.parallel.DistributedDataParallel.__init__(self, *args, **kwargs)

        # 默认仅对master_rank=0打印信息
        assert isinstance(master_rank, (int, list, tuple)), 'Args `master_rank` only supoorts int, list, tuple'
        if isinstance(master_rank, int):
            master_rank = [master_rank]
        self.verbose = (torch.distributed.get_rank() in master_rank)

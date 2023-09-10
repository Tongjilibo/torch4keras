from torch import nn
from torch4keras.trainer import *


class BaseModel(Trainer, nn.Module):
    '''BaseModel, 使用继承的方式来使用
    '''
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        Trainer.__init__(self, *args, **kwargs)
        
BaseModelDP = TrainerDP  # 使用方式和nn.DataParallel一致
BaseModelDDP = TrainerDDP  # 使用方式和DistributedDataParallel一致
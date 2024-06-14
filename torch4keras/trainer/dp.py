from torch import nn
from .base import Trainer


class DPTrainer(nn.DataParallel, Trainer):
    '''DataParallel模式使用多gpu的方法, 
    1) 父类顺序颠倒也会出问题
    2) 使用方式和nn.DataParallel一致, DPTrainer(net, *args, **kwargs)来使用
    '''
    def __init__(self, *args, **kwargs):
        Trainer.__init__(self)
        nn.DataParallel.__init__(self, *args, **kwargs)
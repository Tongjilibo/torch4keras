from torch import nn
from torch4keras.trainer import *


class BaseModel(nn.Module, Trainer):
    '''BaseModel, 使用继承的方式来使用
    
    Examples:
    ```python
    >>> class MyModel(BaseModel):
    ...     def __init__(self, module):
    ...         super().__init__()
    ...         self.model = module
    ...     def forward(self, inputs):
    ...         return self.model(inputs)

    >>> model = MyModel(net)  # 实例化模型对象
    ```
    '''
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        Trainer.__init__(self, *args, **kwargs)
        
BaseModelDP = DPTrainer  # 使用方式和nn.DataParallel一致
BaseModelDDP = DDPTrainer  # 使用方式和DistributedDataParallel一致
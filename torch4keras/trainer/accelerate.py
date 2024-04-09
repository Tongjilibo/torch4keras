from torch import nn
from torch.utils.data import DataLoader
from torch4keras.snippets import log_warn
from typing import Union
from .base import Trainer


class AccelerateTrainer(Trainer):
    '''accelerate来训练'''
    def __init__(self, module: nn.Module, **configs):
        super().__init__(module)
        from accelerate import Accelerator
        accelerator = Accelerator(**configs)
        self.model = accelerator.prepare(module)
        self.accelerator = accelerator
        self.device = accelerator.device
        self.verbose = 1 if accelerator.is_local_main_process else 0
        log_warn('AcclerateTrainer may not be compatible with several callbacks, you may use custom callbacks instead.')
    
    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.optimizer, self.scheduler, self.criterion = self.accelerator.prepare(self.optimizer, self.scheduler, self.criterion)

    def _prepare_inputs(self, train_dataloader:DataLoader, steps_per_epoch:Union[int,None], epochs:int, verbose:int):
        # 如果使用ddp的时候没有使用DistributedSampler，这里会自动修改一下
        train_dataloader = self.accelerator.prepare(train_dataloader)
        super()._prepare_inputs(train_dataloader, steps_per_epoch, epochs, verbose)

    def prepare(self, *args, **kwargs):
        '''调用acclerate的prepare, 如在外面评估时候需要对dev_dataloader使用'''
        return self.accelerator.prepare(*args, **kwargs)

    def unwrap_model(self):
        '''返回nn.Module模块'''
        unwrap_model = self.accelerator.unwrap_model(self.model)
        if isinstance(unwrap_model, nn.Module): return unwrap_model
        return unwrap_model.module if hasattr(unwrap_model, 'module') else unwrap_model

    def loss_backward(self, loss):
        self.accelerator.backward(loss)
        return loss
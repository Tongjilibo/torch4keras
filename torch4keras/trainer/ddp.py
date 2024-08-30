from torch import nn
import torch
from torch.utils.data import DataLoader
from torch4keras.snippets import DottableDict, seed_everything
from torch4keras.callbacks import Callback
from typing import Union, List
import os
from .base import Trainer


class DDPTrainer(nn.parallel.DistributedDataParallel, Trainer):
    '''DistributedDataParallel模式使用多gpu的方法,
    1) 父类顺序颠倒也会出问题
    2) 使用方式和DistributedDataParallel一致, DDPTrainer(net, *args, **kwargs)来使用

    Examples:
    ```python
    >>> from torch4keras.trainer import DDPTrainer
    >>> args = DDPTrainer.init_process_group()
    >>> # 用户自行定义好model
    >>> model = DDPTrainer(
    ...     model,
    ...     master_rank=0,
    ...     device_ids=[args.local_rank],
    ...     output_device=args.local_rank,
    ...     find_unused_parameters=False
    ... )
    >>> # 后续执行和Trainer一致
    ```
    '''
    def __init__(self, *args, master_rank=0, **kwargs):
        Trainer.__init__(self)
        kwargs['device_ids'] = kwargs.get('device_ids', [int(os.getenv('LOCAL_RANK'))])
        kwargs['output_device'] = kwargs.get('output_device', int(os.getenv('LOCAL_RANK')))
        nn.parallel.DistributedDataParallel.__init__(self, *args, **kwargs)

        # 默认仅对master_rank=0打印信息
        assert isinstance(master_rank, (int, list, tuple)), 'Args `master_rank` only supoorts int, list, tuple'
        if isinstance(master_rank, int):
            master_rank = [master_rank]
        self.master_rank = master_rank
        self.verbose = (torch.distributed.get_rank() in master_rank)
    
    def _prepare_inputs(self, train_dataloader:DataLoader, steps_per_epoch:Union[int,None], epochs:int, verbose:int):
        # 如果使用ddp的时候没有使用DistributedSampler，这里会自动修改一下
        from torch.utils.data.distributed import DistributedSampler 
        if (train_dataloader.sampler is None) and (not isinstance(train_dataloader.sampler, DistributedSampler)):
            train_dataloader.sampler = DistributedSampler(train_dataloader.dataset)
        super()._prepare_inputs(train_dataloader, steps_per_epoch, epochs, verbose)
    
    def disable_workers_callback(self, callbacks: Union[Callback, List[Callback]]):
        '''非master_rank上不使用callback'''
        for callback in callbacks:
            if torch.distributed.get_rank() not in self.master_rank:
                callback.run_callback = False

    @classmethod
    def init_process_group(cls, master_rank=0, seed=42):
        '''初始化各项参数'''
        if os.name == 'nt':
            # windows: Diff between backends: https://pytorch.org/docs/stable/distributed.html
            torch.distributed.init_process_group(backend="gloo")
        else:  # linux
            torch.distributed.init_process_group(backend='nccl')

        cls.ddp_config = DottableDict()
        cls.ddp_config.rank = int(os.environ["RANK"])
        cls.ddp_config.local_rank = int(os.getenv('LOCAL_RANK'))
        cls.ddp_config.device = torch.device('cuda', cls.ddp_config.local_rank)
        cls.ddp_config.world_size = int(os.environ["WORLD_SIZE"])
        cls.ddp_config.master_process = cls.ddp_config.rank == master_rank
        torch.cuda.set_device(cls.ddp_config.local_rank)
        seed_everything(seed + cls.ddp_config.rank)
        return cls.ddp_config

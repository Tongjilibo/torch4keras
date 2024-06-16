from .accelerate import *
from .base import *
from .ddp import *
from .deepspeed import *
from .dp import *
from .utils import *
from typing import Literal


# 这里理论上不应该从Trainer继承，这里只是为了后续的代码提示
class AutoTrainer(Trainer):
    def __new__(cls, *args, trainer_type:Literal['deepspeed', 'ddp', 'dp', 'accelerate', 'auto', 'base']='base', **kwargs) -> Trainer:
        from torch4keras.snippets import is_deepspeed_available, is_accelerate_available, log_info
        if trainer_type == 'auto':
            if is_deepspeed_available():
                trainer_type = 'deepspeed'
            elif is_accelerate_available():
                trainer_type = 'accelerate'
            elif torch.cuda.device_count() > 1:
                if int(os.environ.get("RANK", -1)) != -1:
                    trainer_type = 'ddp'
                else:
                    trainer_type = 'dp'
            else:
                trainer_type = 'base'

        if trainer_type == 'deepspeed':
            trainer = DeepSpeedTrainer(*args, **kwargs)
        elif trainer_type == 'ddp':
            trainer = DDPTrainer(*args, **kwargs)
        elif trainer_type == 'dp':
            trainer = DPTrainer(*args, **kwargs)
        elif trainer_type == 'accelerate':
            trainer = AccelerateTrainer(*args, **kwargs)
        elif trainer_type == 'base':
            trainer = Trainer(*args, **kwargs)
        else:
            raise ValueError(f'Args `{trainer_type}` not supported')
        
        # log_info(f'Initialize `{trainer_type}` trainer success.')
        return trainer
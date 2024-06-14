from torch import nn
from torch4keras.snippets import log_info, log_warn
from .base import Trainer
from .dp import DPTrainer
from .ddp import DDPTrainer


def add_trainer(obj, include=None, exclude=None, verbose=0, replace_func=False):
    '''为nn.Module添加Triner对应的方法'''
    if isinstance(obj, (Trainer, DPTrainer, DDPTrainer)):
        log_warn('obj is not a Trainer object')
        return obj
    elif not isinstance(obj, nn.Module):
        log_warn('obj is not a nn.Module object')
        return obj

    if include is None:
        include = set()
    elif isinstance(include, str):
        include = set([include])
    elif isinstance(include, (tuple, list)):
        include = set(include)
    else:
        raise TypeError(f'Arg `include` only receive str/list format, not {type(include)}')

    if exclude is None:
        exclude = set()
    elif isinstance(exclude, (tuple, list)):
        exclude = set(exclude)

    import types
    added_funcs = []
    for k in dir(Trainer):
        set_func = False
        if k in include:  # 必须包含的
            set_func = True
        elif k in exclude:  # 必须屏蔽的
            pass
        elif k.startswith('__') and k.endswith('__'):  # 内部函数不执行
            pass
        elif hasattr(obj, k):  # 如果下游重新定义, 则不继
            if replace_func:
                set_func = True
        else:
            set_func = True

        if set_func and eval(f'isfunction(Trainer.{k})'):
            exec(f'obj.{k} = types.MethodType(Trainer.{k}, obj)')
            added_funcs.append(k)
    obj.initialize()  # 这里初始化会得到一些其他的成员变量, 不可缺省

    if verbose and (len(added_funcs) > 0):
        log_info(f'Already add `{",".join(added_funcs)}` method')
    return obj


def add_module(obj, include=None, exclude=None, verbose=0, replace_func=False):
    '''为Trainer增加nn.Module的方法
    方便外部访问, 如obj.parameters()可以直接访问到obj.module.parameters()
    '''
    if isinstance(obj, nn.Module):
        return obj
    elif not isinstance(obj, Trainer):
        log_warn('obj is not a Trainer object')
        return obj
    elif not isinstance(obj.unwrap_model(), nn.Module):
        log_warn('obj.unwrap_model() is not a nn.Module object')
        return obj
    
    if include is None:
        include = set()
    elif isinstance(include, str):
        include = set([include])
    elif isinstance(include, (tuple, list)):
        include = set(include)
    else:
        raise TypeError(f'Arg `include` only receive str/list format, not {type(include)}')

    if exclude is None:
        exclude = set()
    elif isinstance(exclude, (tuple, list)):
        exclude = set(exclude)


    import types
    added_funcs = []
    for k in dir(obj.unwrap_model()):
        set_func = False
        if k in include:  # 必须包含的
            set_func = True
        elif k in exclude:  # 必须屏蔽的
            pass
        elif k.startswith('__') and k.endswith('__'):
            pass
        elif hasattr(obj, k):  # 如果下游重新定义, 则不继
            if replace_func:
                set_func = True
        else:
            set_func = True
        
        if set_func and eval(f'isinstance(obj.unwrap_model().{k}, types.MethodType)'):
            exec(f'obj.{k} = obj.unwrap_model().{k}')
            added_funcs.append(k)

    if verbose and (len(added_funcs) > 0):
        log_info(f'Already add `{",".join(added_funcs)}` method')
    return obj

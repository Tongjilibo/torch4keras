import pstats
import torch.nn as nn
import torch
from torch4keras.snippets import metric_mapping, ProgbarLogger, Callback, CallbackList, BaseLogger, History
from collections import OrderedDict
from inspect import isfunction


def trainer(cls: nn.Module):
    '''通过装饰器的来为nn.Module的模型结构增加Trainer
    '''
    class cls_trainer(Trainer):
        def __init__(self, *args, **kwargs):
            module = cls(*args, **kwargs)
            super().__init__(module)
    return cls_trainer


class Trainer:
    """Trainer：参考keras实现的模型的训练过程
    支持梯度累积，混合精度，梯度裁剪等功能，也包装了下nn.Module的主要函数方便调用，仍可通过实例trainer.module.*来调用
    """
    def __init__(self, module: nn.Module):
        self.module = module
        # 这里主要是为了外面调用用到
        self.global_step, self.local_step, self.total_steps, self.epoch, self.steps_per_epoch, self.train_dataloader = 0, 0, 0, 0, None, None
        self.resume_step, self.resume_epoch = 0, 0
        self.retain_graph = False  # loss.backward()是否保留计算图
        self.callbacks = []
    
    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs):
        self.module = self.module.cpu(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        self.module = self.module.cuda(*args, **kwargs)
        return self

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)
    
    def named_buffers(self, *args, **kwargs):
        return self.module.named_buffers(*args, **kwargs)

    def named_children(self, *args, **kwargs):
        return self.module.named_children(*args, **kwargs)

    def named_modules(self, *args, **kwargs):
        return self.module.named_modules(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.module.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.module.eval(*args, **kwargs)
    
    def zero_grad(self, *args, **kwargs):
        self.module.zero_grad(*args, **kwargs)
    
    def state_dict(self):
        return self.module.state_dict()
    
    def load_state_dict(self, *args, **kwargs):
        self.module.load_state_dict(*args, **kwargs)

    def save_steps_params(self, save_path):
        '''保存训练过程参数
        '''
        step_params = {'resume_step': (self.local_step+1) % self.steps_per_epoch, 
                       'resume_epoch': self.epoch + (self.local_step+1) // self.steps_per_epoch}
        torch.save(step_params, save_path)

    def load_steps_params(self, save_path):
        '''导入训练过程参数
        '''
        step_params = torch.load(save_path)
        self.resume_step = step_params['resume_step'] 
        self.resume_epoch = step_params['resume_epoch']
        return step_params

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def compile(self, loss, optimizer, scheduler=None, clip_grad_norm=None, use_amp=False, metrics=None, stateful_metrics=None, grad_accumulation_steps=1):
        '''定义loss, optimizer, metrics, 是否在计算loss前reshape
        loss: loss
        optimizer: 优化器
        scheduler: scheduler
        clip_grad_norm: 是否使用梯度裁剪, 默认不启用
        use_amp: 是否使用混合精度，默认不启用
        metrics: 训练过程中需要打印的指标, loss相关指标默认会打印, 目前支持accuracy, 也支持自定义metric，形式为{key: func}
        stateful_metrics: 不滑动平均仅进行状态记录的metric，指标抖动会更加明显
        grad_accumulation_steps: 梯度累积
        '''
        self.criterion = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.use_amp = use_amp
        if use_amp:
            from torch.cuda.amp import autocast
            self.autocast = autocast
            self.scaler = torch.cuda.amp.GradScaler()

        # 训练过程观测的指标
        self.metrics = OrderedDict({'loss': None})
        if metrics is None:
            metrics = []
        elif isinstance(metrics, (str, dict)) or isfunction(metrics):
            metrics = [metrics]

        for metric in metrics:
            # 字符类型，目前仅支持accuracy
            if isinstance(metric, str) and metric != 'loss':
                self.metrics[metric] = None
            # 字典形式 {metric: func}
            elif isinstance(metric, dict):
                self.metrics.update(metric)
            # 函数形式，key和value都赋值metric
            elif isfunction(metric):
                self.metrics.update({metric: metric})
            else:
                raise ValueError('Args metrics only support `String, Dict, Callback, List[String, Dict, Callback]` format')
        self.stateful_metrics = stateful_metrics

        # 梯度累积
        self.grad_accumulation_steps = grad_accumulation_steps

    def args_segmentate(self, train_X):
        '''参数是否展开
        '''
        if isinstance(train_X, torch.Tensor):  # tensor不展开
            pass
        elif self.module.forward.__code__.co_argcount >= 3:
            return True
        return False

    def train_step(self, train_X, train_y):
        '''forward并返回loss
        '''
        if self.use_amp:
            with self.autocast():
                output = self.forward(*train_X) if self.args_segmentate(train_X) else self.forward(train_X)
                loss_detail = self.criterion(output, train_y)
        else:
            output = self.forward(*train_X) if self.args_segmentate(train_X) else self.forward(train_X)
            loss_detail = self.criterion(output, train_y)

        if isinstance(loss_detail, torch.Tensor):
            loss = loss_detail
            loss_detail = {}
        elif isinstance(loss_detail, dict):
            loss = loss_detail['loss']  # 还存在其他loss，仅用于打印
            del loss_detail['loss']
        elif isinstance(loss_detail, (tuple, list)):
            loss = loss_detail[0]
            loss_detail = {f'loss{i}':v for i, v in enumerate(loss_detail[1:], start=1)}
        else:
            raise ValueError('Return loss only support `Tensor/dict/tuple/list` format')

        # 梯度累积
        loss = loss / self.grad_accumulation_steps if self.grad_accumulation_steps > 1 else loss

        # backward
        self.scale_before_step = 0
        if self.use_amp:  # 混合精度
            self.scale_before_step = self.scaler.get_scale()
            self.scaler.scale(loss).backward(retain_graph=self.retain_graph)
        else:
            loss.backward(retain_graph=self.retain_graph)
        return output, loss, loss_detail

    def fit(self, train_dataloader, steps_per_epoch=None, epochs=1, callbacks=None, verbose=1):
        if not hasattr(train_dataloader, '__len__'):
            assert steps_per_epoch is not None, 'Either train_dataloader has attr `__len__` or steps_per_epoch is not None'

        self.steps_per_epoch = len(train_dataloader) if steps_per_epoch is None else steps_per_epoch
        self.total_steps = self.steps_per_epoch * epochs
        self.train_dataloader = train_dataloader  # 设置为成员变量，可由外部的callbacks进行修改
        train_dataloader_iter = iter(self.train_dataloader)  # 循环epoch时不重生成
        
        # callbacks设置
        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]
        for callback in callbacks:
            assert isinstance(callback, Callback), "Args `callbacks` only support Callback() inputs"
        progbarlogger = ProgbarLogger(stateful_metrics=self.stateful_metrics)  # 进度条
        history = History()
        master_rank = self.master_rank if hasattr(self, 'master_rank') else None
        self.callbacks = CallbackList([BaseLogger(self.stateful_metrics), progbarlogger] + callbacks + [history], master_rank=master_rank)
        callback_trainer = self
        self.callbacks.set_trainer(callback_trainer)
        self.callbacks.set_model(self.module)
        self.callbacks.set_optimizer(self.optimizer)
        self.callbacks.set_params({
            'epochs': epochs,
            'steps': self.steps_per_epoch,
            'verbose': verbose,
            'metrics': [i for i in self.metrics.keys() if isinstance(i, str)],
        })
        logs = {}
        self.callbacks.on_train_begin(logs)
        callback_trainer.stop_training = False  # 在EarlyStopping中会重新设置

        # epoch：当前epoch
        # global_step：当前全局训练步数
        # local_step: 当前epoch内的训练步数，不同epoch中相同local_step对应的batch数据不一定相同，在steps_per_epoch=None时相同
        # bti：在dataloader中的index，不同epoch中相同的bti对应的batch数据一般相同，除非重新生成dataloader
        self.bti = 0
        for epoch in range(self.resume_epoch, epochs):
            self.epoch = epoch
            # resume_step：判断local_step的起点，以及进度条的起始位置
            resume_step = self.resume_step if epoch==self.resume_epoch else 0
            self.callbacks.on_epoch_begin(self.global_step, self.epoch)
            progbarlogger.seen = resume_step  # 这里设置进度条的seen，在callbacks中也会修改
            
            for local_step in range(resume_step, self.steps_per_epoch):
                self.local_step = local_step
                self.global_step = self.epoch * self.steps_per_epoch + self.local_step
                # 循环dataloader, 不要试用itertools的cycle，遇到过变量不释放的问题
                try:
                    batch = next(train_dataloader_iter)
                except StopIteration:
                    self.callbacks.on_dataloader_end()  # 适用于数据量较大时，动态读取文件并重新生成self.train_dataloader的情况，如预训练
                    train_dataloader_iter = iter(self.train_dataloader)  # shuffle=True时候，其实顺序也重新生成了
                    self.bti = 0
                    batch = next(train_dataloader_iter)
                self.train_X, self.train_y = batch

                # 从train_X中取batch_size，最多允许嵌套两层，即encoder和decoder的((token_ids1, mask1), (token_ids2, mask2))
                if isinstance(self.train_X, (list, tuple)) and isinstance(self.train_X[0], (list, tuple)):
                    btz = self.train_X[0][0].size(0)
                elif isinstance(self.train_X, (list, tuple)) and (not isinstance(self.train_X[0], (list, tuple))):
                    btz = self.train_X[0].size(0)
                elif isinstance(self.train_X, torch.Tensor):
                    btz = self.train_X.size(0)
                else:
                    raise ValueError('Input only support `[list, tuple, tensor]`')
                logs = {'size': btz}
                self.callbacks.on_batch_begin(self.global_step, self.local_step, logs)

                self.module.train()  # 设置为train模式
                # 入参个数判断，如果入参>=3表示是多个入参，如果=2则表示是一个入参
                self.output, self.loss, self.loss_detail = self.train_step(self.train_X, self.train_y)
                self.callbacks.on_train_step_end()
                                
                # 参数更新, 真实的参数更新次数要除以grad_accumulation_steps，注意调整总的训练步数
                if (self.global_step+1) % self.grad_accumulation_steps == 0:
                    skip_scheduler = False
                    # 混合精度
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        if self.clip_grad_norm is not None:  # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        skip_scheduler = self.scaler.get_scale() != self.scale_before_step
                    else:
                        if self.clip_grad_norm is not None:  # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
                        self.optimizer.step()

                    self.optimizer.zero_grad()  # 清梯度
                    if (self.scheduler is not None) and not skip_scheduler:
                        if isinstance(self.scheduler, (tuple, list)):
                            for scheduler in self.scheduler:
                                scheduler.step()
                        else:
                            self.scheduler.step()

                # 添加loss至log打印
                logs.update({'loss': self.loss.item()})
                logs_loss_detail = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.loss_detail.items()}
                logs.update(logs_loss_detail)
                if self.global_step == resume_step:
                    progbarlogger.add_metrics(list(logs_loss_detail.keys()), add_position=1)
                    
                # 添加metrics至log打印
                for metric, func in self.metrics.items():
                    perf = metric_mapping(metric, func, self.output, self.train_y)  # 内置的一些accuracy指标
                    if perf is not None:
                        if isfunction(metric):  # 直接传入回调函数(无key)
                            if self.global_step == resume_step:
                                progbarlogger.add_metrics(list(perf.keys()))
                            logs.update(perf)
                        elif isinstance(metric, str):  # 直接传入回调函数(有key)
                            logs[metric] = perf

                self.callbacks.on_batch_end(self.global_step, self.local_step, logs)

                self.bti += 1
            self.callbacks.on_epoch_end(self.global_step, self.epoch, logs)
            # TerminateOnNaN、EarlyStopping等停止训练策略
            if callback_trainer.stop_training:
                break
        self.callbacks.on_train_end(logs)
        return history

    @torch.no_grad()
    def predict(self, train_X, return_all=None):
        self.module.eval()
        output = self.forward(*train_X) if self.args_segmentate(train_X) else self.forward(train_X)
        if return_all is None:
            return output
        elif isinstance(output, (tuple, list)) and isinstance(return_all, int) and return_all < len(output):
            return output[return_all]
        else:
            raise ValueError('Return format error')

    def load_weights(self, load_path, strict=True, mapping={}):
        '''加载模型权重
           save_path: 权重加载路径
           mapping：指定key的映射
        '''
        state_dict = torch.load(load_path, map_location='cpu')
        state_dict_raw = {}
        for k, v in state_dict.items():
            k = mapping.get(k, k)
            state_dict_raw[k] = v
        self.module.load_state_dict(state_dict_raw, strict=strict)

    def save_weights(self, save_path, mapping={}):
        '''保存模型权重
           save_path: 权重保存路径
           mapping：指定key的映射
        '''
        state_dict_raw = {}
        for k, v in self.module.state_dict().items():
            k = mapping.get(k, k)
            state_dict_raw[k] = v
        torch.save(state_dict_raw, save_path)


class BaseModel:
    def __init__(self):
        raise DeprecationWarning('The BaseModel module has been deprecated from v0.0.5, you can use new module `Trainer(net)` or `@trainer class Model(nn.Module)`')


class TrainerDP(nn.DataParallel, Trainer):
    '''DataParallel模式使用多gpu的方法, 父类顺序颠倒也会出问题
    '''
    def __init__(self, *args, **kwargs):
        nn.DataParallel.__init__(self, *args, **kwargs)
        Trainer.__init__(self, *args, **kwargs)


class TrainerDDP(nn.parallel.DistributedDataParallel, Trainer):
    '''DistributedDataParallel模式使用多gpu的方法, 父类顺序颠倒也会出问题
    '''
    def __init__(self, *args, master_rank=0, **kwargs):
        self.master_rank = master_rank  # 用于记录打印条的rank
        nn.parallel.DistributedDataParallel.__init__(self, *args, **kwargs)
        Trainer.__init__(self, *args, **kwargs)

import torch.nn as nn
import torch
from torch4keras.snippets import metric_mapping, ProgbarLogger, EarlyStopping
from collections import OrderedDict
from inspect import isfunction


class Model(nn.Module):
    """Trainer, 支持继承、传入Module实例两种方式
    """
    def __init__(self, module=None):
        super(Model, self).__init__()
        # 这里主要是为了外面调用用到
        self.global_step, self.local_step, self.total_steps, self.epoch, self.steps_per_epoch, self.train_dataloader = 0, 0, 0, 0, None, None
        self.resume_step, self.resume_epoch = 0, 0
        self.callbacks = []
        # 传入Module实例方式
        if module is not None:
            self.module = module
    
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
        # 如果传入了网络结构module，则调用module的forward
        if hasattr(self, 'module'):
            return self.module.forward(*inputs, **kwargs)
        # 如果是继承方式，则调用自身的forward
        else:
            return self.forward(*inputs, **kwargs)

    def compile(self, loss, optimizer, scheduler=None, clip_grad_norm=None, use_amp=False, metrics=None, grad_accumulation_steps=1):
        '''定义loss, optimizer, metrics, 是否在计算loss前reshape
        loss: loss
        optimizer: 优化器
        scheduler: scheduler
        clip_grad_norm: 是否使用梯度裁剪, 默认不启用
        use_amp: 是否使用混合精度，默认不启用
        metrics: 训练过程中需要打印的指标, loss相关指标默认会打印, 目前支持accuracy, 也支持自定义metric，形式为{key: func}
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
                raise ValueError('Args metrics only support "String, Dict, Callback, List[String, Dict, Callback]" format')

        # 梯度累积
        self.grad_accumulation_steps = grad_accumulation_steps

    def args_segmentate(self, train_X):
        '''参数是否展开
        '''
        if isinstance(train_X, torch.Tensor):  # tensor不展开
            pass
        elif isinstance(self, (ModelDP, ModelDDP)) or hasattr(self, 'module'):
            if self.module.forward.__code__.co_argcount >= 3:
                return True
        elif self.forward.__code__.co_argcount >= 3:
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
            raise ValueError('Return loss only support Tensor/dict/tuple/list format')

        # 梯度累积
        loss = loss / self.grad_accumulation_steps if self.grad_accumulation_steps > 1 else loss
        return output, loss, loss_detail

    def callback_fun(self, mode, logs={}):
        '''统一调用callback, 方便一些判断条件的触发
        '''
        # 如果是分布式DDP训练，则仅masker_rank可以callback
        if isinstance(self, ModelDDP) and self.master_rank!=torch.distributed.get_rank():
            return

        if mode == 'train_begin':
            for callback in self.callbacks:
                callback.on_train_begin()
        elif mode == 'epoch_begin':
            for callback in self.callbacks:
                callback.on_epoch_begin(self.global_step, self.epoch, logs)
        elif mode == 'batch_begin':
            for callback in self.callbacks:
                callback.on_batch_begin(self.global_step, self.local_step, logs)
        elif mode == 'batch_end':
            for callback in self.callbacks:
                callback.on_batch_end(self.global_step, self.local_step, logs)
        elif mode == 'epoch_end':
            for callback in self.callbacks:
                callback.on_epoch_end(self.global_step, self.epoch, logs)
        elif mode == 'train_end':
            for callback in self.callbacks:
                callback.on_train_end()
        elif mode == 'dataloader_end':
            for callback in self.callbacks:
                callback.on_dataloader_end()

    def fit(self, train_dataloader, steps_per_epoch=None, epochs=1, callbacks=None):
        if not hasattr(train_dataloader, '__len__'):
            assert steps_per_epoch is not None, 'Either train_dataloader has attr "__len__" or steps_per_epoch is not None'

        self.steps_per_epoch = len(train_dataloader) if steps_per_epoch is None else steps_per_epoch
        self.total_steps = self.steps_per_epoch * epochs
        self.train_dataloader = train_dataloader  # 设置为成员变量，可由外部的callbacks进行修改
        train_dataloader_iter = iter(self.train_dataloader)  # 循环epoch时不重生成

        callbacks = [] if callbacks is None else callbacks
        callbacks = callbacks if isinstance(callbacks, (list, tuple)) else [callbacks]
        self.callbacks = [ProgbarLogger(epochs, self.steps_per_epoch, [i for i in self.metrics.keys() if isinstance(i, str)])] + callbacks
        self.callback_fun('train_begin')

        # epoch：当前epoch
        # global_step：当前全局训练步数
        # local_step: 当前epoch内的训练步数，不同epoch中相同local_step对应的batch数据不一定相同，在steps_per_epoch=None时相同
        # bti：在dataloader中的index，不同epoch中相同的bti对应的batch数据一般相同，除非重新生成dataloader
        self.bti = 0
        for epoch in range(self.resume_epoch, epochs):
            self.epoch = epoch
            # resume_step：判断local_step的起点，以及进度条的起始位置
            resume_step = self.resume_step if epoch==self.resume_epoch else 0
            self.callback_fun('epoch_begin')
            self.callbacks[0].seen = resume_step
            
            for local_step in range(resume_step, self.steps_per_epoch):
                self.local_step = local_step
                self.global_step = self.epoch * self.steps_per_epoch + self.local_step
                # 循环dataloader, 不要试用itertools的cycle，遇到过变量不释放的问题
                try:
                    batch = next(train_dataloader_iter)
                except StopIteration:
                    self.callback_fun('dataloader_end')  # 适用于数据量较大时，动态读取文件并重新生成dataloader的情况，如预训练
                    train_dataloader_iter = iter(self.train_dataloader)  # shuffle=True时候，其实顺序也重新生成了
                    self.bti = 0
                    batch = next(train_dataloader_iter)
                train_X, train_y = batch

                logs = OrderedDict()
                self.callback_fun('batch_begin', logs)

                self.train()  # 设置为train模式
                # 入参个数判断，如果入参>=3表示是多个入参，如果=2则表示是一个入参
                output, loss, loss_detail = self.train_step(train_X, train_y)
                
                # 主要是方便bert4torch中的对抗训练
                scale_before_step, loss, loss_detail = self.after_train_step(train_X, train_y, output, loss, loss_detail)
                
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
                        skip_scheduler = self.scaler.get_scale() != scale_before_step
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
                logs.update({'loss': loss.item()})
                logs_loss_detail = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_detail.items()}
                logs.update(logs_loss_detail)
                if self.global_step == resume_step:
                    self.callbacks[0].add_metrics(list(logs_loss_detail.keys()), add_position=1)
                    
                # 添加metrics至log打印
                for metric, func in self.metrics.items():
                    perf = metric_mapping(metric, func, output, train_y)  # 内置的一些accuracy指标
                    if perf is not None:
                        if isfunction(metric):  # 直接传入回调函数(无key)
                            if self.global_step == resume_step:
                                self.callbacks[0].add_metrics(list(perf.keys()))
                            logs.update(perf)
                        elif isinstance(metric, str):  # 直接传入回调函数(有key)
                            logs[metric] = perf

                self.callback_fun('batch_end', logs)

                self.bti += 1
            self.callback_fun('epoch_end', logs)
            # earlystop策略
            callback_tmp = [callback_tmp for callback_tmp in self.callbacks if isinstance(callback_tmp, EarlyStopping)]
            if callback_tmp and callback_tmp[0].stopped_epoch > 0:
                break
        self.callback_fun('train_end', logs)

    @torch.no_grad()
    def predict(self, train_X, return_all=None):
        self.eval()
        output = self.forward(*train_X) if self.args_segmentate(train_X) else self.forward(train_X)
        if return_all is None:
            return output
        elif isinstance(output, (tuple, list)) and isinstance(return_all, int) and return_all < len(output):
            return output[return_all]
        else:
            raise ValueError('Return format error')
    
    def after_train_step(self, train_X, train_y, output, loss, loss_detail):
        scale_before_step = 0
        if self.use_amp:  # 混合精度
            scale_before_step = self.scaler.get_scale()
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return scale_before_step, loss, loss_detail


class ModelDP(Model, nn.DataParallel):
    '''DataParallel模式使用多gpu的方法
    '''
    def __init__(self, *args, **kwargs):
        nn.DataParallel.__init__(self, *args, **kwargs)


class ModelDDP(Model, nn.parallel.DistributedDataParallel):
    '''DistributedDataParallel模式使用多gpu的方法
    '''
    def __init__(self, *args, master_rank=0, **kwargs):
        self.master_rank = master_rank  # 用于记录打印条的rank
        nn.parallel.DistributedDataParallel.__init__(self, *args, **kwargs)

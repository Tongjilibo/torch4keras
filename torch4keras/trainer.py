from torch import nn
import torch
from torch4keras.snippets import DottableDict, metric_mapping, get_parameter_device, log_info, log_warn, log_error, print_trainable_parameters, colorful
from torch4keras.callbacks import KerasProgbar, SmoothMetricsCallback, TqdmProgbar, ProgressBar2Progbar, Callback, CallbackList, History
from collections import OrderedDict
from inspect import isfunction
import os
import json
import math


class Trainer:
    '''Trainer, 传入Module实例

    :param module: None/nn.Module，nn.Module()的模型实例
    '''
    def __init__(self, module:nn.Module=None):
        super(Trainer, self).__init__()
        self.initialize(module)
    
    def initialize(self, module:nn.Module=None):
        # 传入Module实例方式
        if module is not None:
            assert isinstance(module, nn.Module), 'Args `module` only support nn.Module format'
            self.module = module

        self.global_step, self.local_step, self.total_steps, self.batch_step = 0, 0, 0, 0
        self.epoch, self.steps_per_epoch, self.train_dataloader = 0, None, None
        self.resume_step, self.resume_epoch = 0, 0
        self.retain_graph = False  # loss.backward()是否保留计算图
        self.move_to_model_device = True  # 自动把tensor转到model所在的device
        self.log_first_step = False  # 是否打印第一个step的数据
        self.criterion = None  # criterion
        self.optimizer = None  # optimizer
        self.scheduler = None  # scheduler
        self.callbacks = []  # 所有的Callbacks，如果fit中不传入，则默认为[progbarlogger, smoothmetrics, history]三项
        self.run_callbacks = True  # 是否运行Callbacks，目前主要是在DDP模式下运用
        self.loss2metrics = True  # 把loss_detail打印在进度条的metrics上
        # add_module(self)  # 增加nn.Module的成员方法

    def compile(self, loss=None, optimizer=None, scheduler=None, clip_grad_norm=None, mixed_precision=False, metrics=None, 
                grad_accumulation_steps=1, progbar_config=None, smooth_metrics_config=None, **kwargs):
        '''complile: 定义loss, optimizer, metrics等参数
        
        :param loss: loss
        :param optimizer: 优化器
        :param scheduler: lr_scheduler
        :param clip_grad_norm: bool, 是否使用梯度裁剪, 默认为False
        :param mixed_precision: bool, 是否使用混合精度，默认为False
        :param metrics: str/List[str]/dict, 训练过程中需要打印的指标, loss相关指标默认会打印, 目前支持accuracy, 也支持自定义metric，形式为{key: func}
        :param grad_accumulation_steps: int, 梯度累积步数，默认为1
        :param bar: str, 使用进度条的种类，从kwargs中解析，默认为keras, 可选keras, tqdm, progressbar2
        :param progbar_config: 进度条的配置，默认是对整个epoch计算均值指标
            bar: str, 默认为keras
            stateful_metrics: List[str], 表示不使用指标平滑仅进行状态记录的metric，指标抖动会更加明显，默认为None表示使用指标平滑
            width: int, keras进度条下表示进度条的长度
        :param smooth_metrics_config: 指标平滑的配置，默认为None表示采取默认平滑设置；传入False表示不使用平滑
            stateful_metrics: List[str], 表示不使用指标平滑仅进行状态记录的metric，指标抖动会更加明显，默认为None表示使用指标平滑
            interval: int, 表示指标平滑时候的累计步数，默认为100

        :return: None
        '''
        self.criterion = loss or self.criterion  # loss
        self.optimizer = optimizer or self.optimizer  # 优化器
        self.scheduler = scheduler or self.scheduler  # lr_scheduler
        self.clip_grad_norm = clip_grad_norm  # 梯度裁剪
        self.grad_accumulation_steps = grad_accumulation_steps  # 梯度累积

        # 混合精度
        assert mixed_precision in {True, False, 'fp16', 'bf16'}
        self.mixed_precision = 'fp16' if mixed_precision is True else mixed_precision
        if self.mixed_precision:
            self.autocast = torch.cuda.amp.autocast
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

        # 进度条参数
        self.progbar_config = progbar_config or {'bar': 'keras', 'stateful_metrics': None}
        self.progbar_config.update({k:v for k, v in kwargs.items() if k in ['bar', 'stateful_metrics', 'width']})  # 直接传参
        self.progbar_config['bar'] = self.progbar_config.get('bar', 'keras')
        assert self.progbar_config['bar'] in {'keras', 'tqdm', 'progressbar2'}, \
            f'Args `bar`={self.progbar_config["bar"]} illegal, only support `keras, tqdm, progressbar2`'

        # smooth_metrics参数: 默认平滑
        if smooth_metrics_config is False:  # compile时传入False, 表示不使用平滑
            self.smooth_metrics_config = None
        else:
            self.smooth_metrics_config = smooth_metrics_config or {}
            self.smooth_metrics_config.update({k:v for k, v in kwargs.items() if k in ['stateful_metrics', 'interval', 'verbose']})  # 直接传参

        # 其他参数设置
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def print_trainable_parameters(self):
        '''打印可训练的参数量'''
        print_trainable_parameters(self.unwrap_model())

    @property
    def device(self) -> torch.device:
        '''获取model所在的device'''
        if hasattr(self, '_device'):
            return self._device
        return get_parameter_device(self.unwrap_model())

    @device.setter
    def device(self, value):
        '''允许修改self.device'''
        self._device = value

    def _move_to_model_device(self, inputs):
        '''遍历并转移到model.device上（递归）'''
        if self.move_to_model_device:
            if isinstance(inputs, torch.Tensor) and hasattr(self, 'device') and (inputs.device != self.device):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, (tuple, list)):
                inputs = list(inputs) if isinstance(inputs, tuple) else inputs
                for i, ts in enumerate(inputs):
                    inputs[i] = self._move_to_model_device(ts)
            elif isinstance(inputs, dict):
                for k, v in inputs.items():
                    inputs[k] = self._move_to_model_device(v)
        return inputs

    def _log_first_step(self, resume_step, train_X, train_y):
        '''打印第一个step的数据'''
        if self.log_first_step and self.verbose and (self.global_step == resume_step):
            print(colorful('[Train_data]: ', color='green'), + train_X)
            print(colorful('[Label]: ', color='green'), + train_y)

    def _forward(self, *inputs, **input_kwargs):
        '''调用模型的forward，方便下游继承的时候可以自定义使用哪个模型的forward
        '''
        return self._argparse_forward(self.unwrap_model(), *inputs, **input_kwargs)

    def _argparse_forward(self, model, *inputs, **input_kwargs):
        '''调用模型的forward
        如果传入了网络结构module，则调用module的forward；如果是继承方式，则调用自身的forward
        这里声明为staticmethod的话，使用add_trainer会有问题
        '''
        if (len(inputs)==1) and isinstance(inputs[0], (tuple,list)):  # 防止([])嵌套
            inputs = inputs[0]
        
        if isinstance(inputs, torch.Tensor):  # tensor不展开
            return model.forward(inputs, **input_kwargs)
        elif isinstance(inputs, (tuple, list)):
            return model.forward(*inputs, **input_kwargs)
        else:
            return model.forward(inputs, **input_kwargs)

    def train_step(self, train_X, train_y):
        ''' Perform a training step on a batch of inputs. '''
        # 计算loss
        if self.mixed_precision:
            with self.autocast(dtype=torch.float16 if self.mixed_precision=='fp16' else torch.bfloat16):
                output = self._forward(train_X)
                loss_detail = self.criterion(output, train_y)
        else:
            output = self._forward(train_X)
            loss_detail = self.criterion(output, train_y)

        # 整理loss
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

        # loss backward
        loss = self.loss_backward(loss)
        loss_detail = {k: (v.item() if isinstance(v, torch.Tensor) else v) / self.grad_accumulation_steps for k, v in loss_detail.items()}
        return output, loss, loss_detail

    def loss_backward(self, loss):
        '''loss.backward'''
        self.scale_before_step = 0
        if self.mixed_precision:  # 混合精度
            self.scale_before_step = self.scaler.get_scale()
            self.scaler.scale(loss).backward(retain_graph=self.retain_graph)
        else:
            loss.backward(retain_graph=self.retain_graph)
        return loss
    
    def step(self):
        '''参数更新'''
        skip_scheduler = False
        # 混合精度
        if self.mixed_precision:
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

    def _prepare_inputs(self, train_dataloader, steps_per_epoch, epochs, verbose):
        '''对fit的输入进行类型检查并置为成员变量'''
        if not hasattr(train_dataloader, '__len__'):
            assert steps_per_epoch is not None, 'Either train_dataloader has attr `__len__` or steps_per_epoch is not None'
        if steps_per_epoch is None:
            self.steps_per_epoch = math.ceil(len(train_dataloader) // self.grad_accumulation_steps)
        else:
            self.steps_per_epoch = steps_per_epoch
        self.batch_size = train_dataloader.batch_size
        self.epochs = epochs
        self.total_steps = self.steps_per_epoch * epochs
        self.train_dataloader = train_dataloader  # 设置为成员变量，可由外部的callbacks进行修改
        self.train_dataloader_iter = iter(self.train_dataloader)  # 循环epoch时不重生成
        self.verbose = self.verbose if hasattr(self, 'verbose') else verbose

    def _prepare_callbacks(self, callbacks):
        '''callbacks设置'''
        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks]
        for callback in callbacks:
            assert isinstance(callback, Callback), "Args `callbacks` only support Callback() inputs"

        history = History()
        callbacks_ = []

        # 进度条
        progbarlogger = None
        if self.verbose:
            if self.progbar_config['bar'] == 'keras':
                progbarlogger = KerasProgbar(**self.progbar_config)
            elif self.progbar_config['bar'] == 'tqdm':
                progbarlogger = TqdmProgbar(**self.progbar_config)
            elif self.progbar_config['bar'] == 'progressbar2':
                progbarlogger = ProgressBar2Progbar(**self.progbar_config)
            else:
                progbarlogger = KerasProgbar(**self.progbar_config)
            callbacks_.append(progbarlogger)

        # 指标平滑
        if self.smooth_metrics_config is not None:
            if any([isinstance(i, SmoothMetricsCallback) for i in callbacks]):
                # 用户自定的callbacks中包含了SmoothMetricsCallback
                log_warn(f'SmoothMetricsCallback already in use and args `smooth_metrics_config` will be ignored')
            else:
                callbacks_.append(SmoothMetricsCallback(**self.smooth_metrics_config))

            # 检查指标平滑的设置和后续callback的设置的interval是不是一致
            smooth_callback = [callback for callback in callbacks_+callbacks if isinstance(callback, SmoothMetricsCallback)][0]
            for callback in callbacks_+callbacks:
                if hasattr(callback, 'interval') and (callback != smooth_callback) and (callback.interval is not None) and \
                    (callback.interval != smooth_callback.interval):
                    log_warn(f'{type(callback).__name__}.interval={callback.interval} while SmoothMetricsCallback.interval={smooth_callback.interval}')

        callbacks_  += callbacks + [history]
        self.callbacks = CallbackList(callbacks_, run_callbacks=self.run_callbacks)
        callback_trainer = self
        callback_model = self.unwrap_model()
        params = {
            'epochs': self.epochs,
            'steps': self.steps_per_epoch,
            'verbose': self.verbose,
            'metrics': [i for i in self.metrics.keys() if isinstance(i, str)],
        }
        self.callbacks.set_all(trainer=callback_trainer, model=callback_model, optimizer=self.optimizer, scheduler=self.scheduler, params=params)
        callback_trainer.stop_training = False  # 在EarlyStopping中会重新设置
        return history, callback_trainer, progbarlogger

    def _prepare_nextbatch(self):
        '''准备下一个batch数据'''
        # 循环dataloader, 不要试用itertools的cycle，遇到过变量不释放的问题
        try:
            batch = next(self.train_dataloader_iter)
            self.batch_step += 1
        except StopIteration:
            self.callbacks.on_dataloader_end()  # 适用于数据量较大时，动态读取文件并重新生成self.train_dataloader的情况，如预训练
            # DDP训练时候为了避免每个epoch样本一致，修改随机种子
            if isinstance(self.train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler) and \
                hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(self.epoch)
            self.train_dataloader_iter = iter(self.train_dataloader)  # shuffle=True时候，其实顺序也重新生成了
            self.batch_step = 0
            batch = next(self.train_dataloader_iter)

        batch = self._move_to_model_device(batch)
        return batch

    def fit(self, train_dataloader, steps_per_epoch=None, epochs=1, callbacks=None, verbose=1):
        '''模型训练
        
        :param train_dataloader: Dataloader, 训练数据集
        :param steps_per_epoch: int, 每个epoch训练的steps，默认为None表示自行计算 
        :param epochs: int, 训练的轮次, 默认为1
        :param callbacks: Callback/List[Callback], 回调函数，可调用预制的Callback或者自定义，默认为None 
        :param verbose: int, 是否打印，默认为1表示打印
        :return: History
        '''
        # 输入处理
        self._prepare_inputs(train_dataloader, steps_per_epoch, epochs, verbose)

        # 准备callbacks
        history, callback_trainer, progbarlogger  = self._prepare_callbacks(callbacks)

        #       epoch: 当前epoch
        # global_step: 当前全局训练步数
        #  local_step: 当前epoch内的训练步数，不同epoch中相同local_step对应的batch数据不一定相同，在steps_per_epoch=None时相同
        #  batch_step: 在dataloader中的index，不同epoch中相同的bti对应的batch数据一般相同，除非重新生成dataloader
        self.callbacks.on_train_begin()
        for epoch in range(self.resume_epoch, epochs):
            self.epoch = epoch
            # resume_step：判断local_step的起点，以及进度条的起始位置
            resume_step = self.resume_step if epoch==self.resume_epoch else 0
            self.callbacks.on_epoch_begin(self.global_step, self.epoch)
            if self.verbose:
                progbarlogger.seen = resume_step  # 这里设置进度条的seen，在callbacks中也会修改
            
            for local_step in range(resume_step, self.steps_per_epoch):
                self.local_step = local_step
                self.global_step = self.epoch * self.steps_per_epoch + self.local_step
                logs = self._log_init()
                self.callbacks.on_batch_begin(self.global_step, self.local_step, logs)

                # forward和backward
                self.unwrap_model().train()  # 设置为train模式
                tr_loss, tr_loss_detail = 0, {}
                for _ in range(self.grad_accumulation_steps):
                    train_X, train_y = self._prepare_nextbatch()  # 获取下一个batch的训练数据
                    self._log_first_step(resume_step, train_X, train_y)  # log第一个step
                    output, loss, loss_detail = self.train_step(train_X, train_y)
                    self.callbacks.on_train_step_end()
                    tr_loss += loss.item()
                    for k, v in loss_detail.items():
                        tr_loss_detail[k] = tr_loss_detail.get(k, 0) + v
                # TODO: 理论上梯度累积时需对output和train_y进行合并，主要是为了metric_mapping计算的准确
                
                # 参数更新
                self.step()

                # 添加loss至log打印
                logs.update(dict({'loss': tr_loss}, **tr_loss_detail))
                if self.verbose and self.loss2metrics and (self.global_step == resume_step):
                    # 把loss_detail添加到进度条metrics中
                    progbarlogger.add_metrics(list(tr_loss_detail.keys()), add_position=1)
                    
                # 添加metrics至log打印
                for metric, func in self.metrics.items():
                    perf = metric_mapping(metric, func, output, train_y)  # 内置的一些accuracy指标
                    if perf is not None:
                        if isfunction(metric):  # 直接传入回调函数(无key)
                            if self.verbose and (self.global_step == resume_step):
                                progbarlogger.add_metrics(list(perf.keys()))
                            logs.update(perf)
                        elif isinstance(metric, str):  # 直接传入回调函数(有key)
                            logs[metric] = perf

                self.callbacks.on_batch_end(self.global_step, self.local_step, logs)

            self.callbacks.on_epoch_end(self.global_step, self.epoch, logs)
            # TerminateOnNaN、EarlyStopping等停止训练策略
            if callback_trainer.stop_training:
                break
        self.callbacks.on_train_end(logs)
        return history

    def _log_init(self):
        '''获取batch_size，主要是用于callback中的BaseLogger和Callback
        '''
        logs = {}

        # 添加lr
        try:
            logs['lr'] = self.optimizer.param_groups[0]["lr"]
        except:
            pass
        return logs

    @torch.no_grad()
    def predict(self, *inputs, **input_kwargs):
        '''模型预测，调用forward()'''
        self.unwrap_model().eval()
        inputs = self._move_to_model_device(inputs)
        input_kwargs = self._move_to_model_device(input_kwargs)
        return self._forward(*inputs, **input_kwargs)
        
    def load_steps_params(self, save_path):
        '''导入训练过程参数
        
        :param save_path: str, 训练过程参数保存路径
        '''
        step_params = torch.load(save_path)
        self.resume_step = step_params['resume_step'] 
        self.resume_epoch = step_params['resume_epoch']
        return step_params

    def save_steps_params(self, save_path):
        '''保存训练过程参数

        :param save_path: str, 训练过程参数保存路径
        '''
        step_params = {'resume_step': (self.local_step+1) % self.steps_per_epoch, 
                       'resume_epoch': self.epoch + (self.local_step+1) // self.steps_per_epoch}
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(step_params, save_path)

    def from_pretrained(self, load_path, strict=True):
        '''按照pretrained的格式来加载权重
        '''
        self.load_weights(load_path, strict, 'pretrained')

    def load_weights(self, load_path, strict=True, mapping={}):
        '''加载模型权重, 支持加载权重文件list

        :param save_path: str/tuple/list, 权重加载路径
        :param strict: bool, torch.load()是否严格加载
        :param mapping: dict, 指定key的映射
            1. mapping={}, 表示按照模型自身结构加载，一般加载finetune后使用save_weights()保存出来的权重
            2. mapping='pretrained', 表示按照预训练格式来加载，在bert4torch中一般是使用build_transformer_model来加载预训练权重
            3. mapping自定义，根据用户自定义mapping来加载权重
        '''
        state_dict_raw = {}
        if isinstance(load_path, (tuple, list)):
            strict = False  # 加载多个权重文件时候，strict设置为False
        elif isinstance(load_path, str):
            load_path = [load_path]
        else:
            raise ValueError('Args `load_path` only support str/tuple/list format')
        
        if mapping == 'pretrained':
            if hasattr(self, 'variable_mapping'):
                mapping = {v:k for k, v in self.variable_mapping().items()}
            else:
                 log_warn('Model do not have `variable_mapping()` function and will load_weights() using original keys')

        for load_path_i in load_path:
            state_dict = torch.load(load_path_i, map_location='cpu')
            for k, v in state_dict.items():
                k = mapping.get(k, k)
                state_dict_raw[k] = v
            self.unwrap_model().load_state_dict(state_dict_raw, strict=strict)

    def save_pretrained(self, save_path, trainable_only=False, verbose=1):
        '''按照pretrained的格式来保存权重
        '''
        self.save_weights(save_path, 'pretrained', trainable_only, verbose)

    def save_weights(self, save_path, mapping={}, trainable_only=False, verbose=1):
        '''保存模型权重

        :param save_path: str, 权重保存路径
        :param mapping: dict/str, 指定key的映射, 如果mapping='pretrained', 则按照自带的mapping的reverse来保存
            1. mapping={}, 表示按照模型自身结构的key来保存，后续可直接load_weights()加载
            2. mapping='pretrained', 表示按照预训练格式来保存，后续在bert4torch中可使用build_transformer_model来加载，或者load_weights(mapping='pretrained')加载
            3. mapping自定义，根据用户自定义mapping来保存权重
        :param trainable_only: bool, 指定仅保存可训练参数
        '''
        state_dict_raw = {}
        state_dict = self.unwrap_model().state_dict()
        trainable_parameters = set(p for p,v in self.unwrap_model().named_parameters() if v.requires_grad)

        if mapping == 'pretrained':
            if hasattr(self, 'variable_mapping'):
                mapping = self.variable_mapping()
            else:
                 log_warn('Model do not have `variable_mapping()` function and will save_weights() using original keys')
        
        for k, v in state_dict.items():
            # 只保存可训练的模型部分
            if trainable_only and (k not in trainable_parameters):
                continue
            k = mapping.get(k, k)
            state_dict_raw[k] = v
        
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(state_dict_raw, save_path)
        if trainable_only and (verbose > 0):
            params_all = sum(p.numel() for p in self.unwrap_model().parameters())
            params_trainable = sum(p.numel() for p in self.unwrap_model().parameters() if p.requires_grad)
            ratio = params_trainable/params_all*100
            log_info(f"Only trainable parameters saved and occupy {params_trainable}/{params_all}={ratio:.2f}%")

    def resume_from_checkpoint(self, save_dir=None, model_path=None, optimizer_path=None, scheduler_path=None, steps_params_path=None, 
                               mapping={}, verbose=0, **kwargs):
        '''同时加载模型、优化器、训练过程参数

        :param save_dir: str, 保存目录
        :param model_path: str, 模型文件路径
        :param optimizer_path: str, 优化器文件路径
        :param scheduler_path: str, scheduler文件路径
        :param steps_params_path: str, 训练过程参数保存路径
        :param mapping: dict, 模型文件的mapping
        '''
        model_path = model_path or os.path.join(save_dir, 'model.pt')
        optimizer_path = optimizer_path or os.path.join(save_dir, 'optimizer.pt')
        scheduler_path = scheduler_path or os.path.join(save_dir, 'scheduler.pt')
        steps_params_path = steps_params_path or os.path.join(save_dir, 'steps_params.pt')

        verbose_str = ''
        # 加载模型权重
        if model_path:
            self.load_weights(model_path, mapping=mapping)
            verbose_str += f'Model weights successfuly resumed from {model_path}\n'
        # 加载优化器，断点续训使用
        if optimizer_path:
            state_dict = torch.load(optimizer_path, map_location='cpu')
            self.optimizer.load_state_dict(state_dict)
            verbose_str += f'Optimizer successfuly resumed from {optimizer_path}\n'
        # 加载优化器，断点续训使用
        if scheduler_path:
            state_dict = torch.load(scheduler_path, map_location='cpu')
            self.scheduler.load_state_dict(state_dict)
            verbose_str += f'Scheduler successfuly resumed from {scheduler_path}\n'
        # 加载训练进度参数，断点续训使用
        if steps_params_path:
            self.load_steps_params(steps_params_path)
            verbose_str += f'Steps_params successfuly resumed from {steps_params_path}'
        if verbose == 1:
            print(verbose_str)

    def save_to_checkpoint(self, save_dir=None, model_path=None, optimizer_path=None, scheduler_path=None, steps_params_path=None, 
                           mapping={}, trainable_only=False, verbose=0, **kwargs):
        '''同时保存模型、优化器、训练过程参数、scheduler

        :param save_dir: str, 保存目录
        :param model_path: str, 模型文件路径
        :param optimizer_path: str, 优化器文件路径
        :param scheduler_path: str, scheduler文件路径
        :param steps_params_path: str, 训练过程参数保存路径
        :param mapping: dict, 模型文件的mapping
        :param trainable_only
        '''
        model_path = model_path or os.path.join(save_dir, 'model.pt')
        optimizer_path = optimizer_path or os.path.join(save_dir, 'optimizer.pt')
        scheduler_path = scheduler_path or os.path.join(save_dir, 'scheduler.pt')
        steps_params_path = steps_params_path or os.path.join(save_dir, 'steps_params.pt')

        verbose_str = ''
        if model_path:
            self.save_weights(model_path, mapping=mapping, trainable_only=trainable_only)
            verbose_str += f'Model weights successfuly saved to {model_path}\n'
        if optimizer_path:
            save_dir = os.path.dirname(optimizer_path)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            verbose_str += f'Optimizer successfuly saved to {optimizer_path}\n'
        if scheduler_path and (self.scheduler is not None):
            save_dir = os.path.dirname(scheduler_path)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.scheduler.state_dict(), scheduler_path)
            verbose_str += f'Scheduler successfuly saved to {scheduler_path}\n'
        if steps_params_path:
            self.save_steps_params(steps_params_path)
            verbose_str += f'Steps_params successfuly saved to {steps_params_path}'
        if verbose == 1:
            print(verbose_str)

    def unwrap_model(self):
        '''返回nn.Module模块
        '''
        if isinstance(self, nn.Module): return self
        return self.module if hasattr(self, 'module') else self


class TrainerDP(nn.DataParallel, Trainer):
    '''DataParallel模式使用多gpu的方法, 
    1) 父类顺序颠倒也会出问题
    2) 使用方式和nn.DataParallel一致，TrainerDP(net, *args, **kwargs)来使用
    '''
    def __init__(self, *args, **kwargs):
        Trainer.__init__(self)
        nn.DataParallel.__init__(self, *args, **kwargs)


class TrainerDDP(nn.parallel.DistributedDataParallel, Trainer):
    '''DistributedDataParallel模式使用多gpu的方法,
    1) 父类顺序颠倒也会出问题
    2) 使用方式和DistributedDataParallel一致，TrainerDDP(net, *args, **kwargs)来使用
    '''
    def __init__(self, *args, master_rank=0, **kwargs):
        Trainer.__init__(self)
        nn.parallel.DistributedDataParallel.__init__(self, *args, **kwargs)

        # 默认仅对master_rank=0打印信息
        assert isinstance(master_rank, (int, list, tuple)), 'Args `master_rank` only supoorts int, list, tuple'
        if isinstance(master_rank, int):
            master_rank = [master_rank]
        self.verbose = (torch.distributed.get_rank() in master_rank)
    

def add_trainer(obj, include=None, exclude=None, verbose=0):
    '''为nn.Module添加Triner对应的方法'''
    if isinstance(obj, (Trainer, TrainerDP, TrainerDDP)):
        log_warn('obj is not a Trainer object')
        return obj
    elif not isinstance(obj, nn.Module):
        log_warn('obj is not a nn.Module object')
        return obj

    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]
    if (include is not None) and (exclude is not None):
        log_warn('Args `include` and `exclude` can not be valid at the same time')

    import types
    for k in dir(Trainer):
        if (include is not None) and (k not in include):  # 必须包含的
            continue
        elif (exclude is not None) and (k in exclude):  # 必须屏蔽的
            continue
        elif k.startswith('__') and k.endswith('__'):
            continue
        elif hasattr(obj, k):  # 如果下游重新定义，则不继
            continue
        
        if eval(f'isfunction(Trainer.{k})'):
                # 方法
            exec(f'obj.{k} = types.MethodType(Trainer.{k}, obj)')
            if verbose:
                log_info(f'Already add obj.{k} method')
    obj.initialize()  # 这里初始化会得到一些其他的成员变量，不可缺省
    return obj


def add_module(obj, include=None, exclude=None, verbose=0):
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
    
    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]
    if (include is not None) and (exclude is not None):
        log_warn('Args `include` and `exclude` can not be valid at the same time')

    import types
    for k in dir(obj.unwrap_model()):
        if (include is not None) and (k not in include):  # 必须包含的
            continue
        elif (exclude is not None) and (k in exclude):  # 必须屏蔽的
            continue
        elif k.startswith('__') and k.endswith('__'):
            continue
        elif hasattr(obj, k):  # 如果下游重新定义，则不继
            continue
        if eval(f'isinstance(obj.unwrap_model().{k}, types.MethodType)'):
            exec(f'obj.{k} = obj.unwrap_model().{k}')
            if verbose:
                log_info(f'Already add obj.{k} method')
    return obj


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

    def _prepare_inputs(self, *args):
        super()._prepare_inputs(*args)
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        self.train_dataloader_iter = iter(self.train_dataloader)

    def prepare(self, *args, **kwargs):
        '''调用acclerate的prepare，如在外面评估时候需要对dev_dataloader使用'''
        return self.accelerator.prepare(*args, **kwargs)

    def unwrap_model(self):
        '''返回nn.Module模块'''
        unwrap_model = self.accelerator.unwrap_model(self.model)
        if isinstance(unwrap_model, nn.Module): return unwrap_model
        return unwrap_model.module if hasattr(unwrap_model, 'module') else unwrap_model

    def loss_backward(self, loss):
        self.accelerator.backward(loss)
        return loss


class DeepSpeedTrainer(Trainer):
    '''deepspeed来训练'''
    def __init__(self, module, config_path):
        super().__init__(module)
        self.model = module
        self.config = DottableDict(json.load(open(config_path)))
        self.config['steps_per_print'] = self.config.get('steps_per_print', 1e9)  # 默认不打印，防止进度条打印问题

    def compile(self, *args, log_level='warning', inference=False, master_rank=0, **kwargs):
        super().compile(*args, **kwargs)
        import deepspeed
        from deepspeed.utils import logger as ds_logger
        import logging
        log_levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }

        ds_logger.setLevel(log_levels.get(log_level, logging.WARNING))

        if inference:
            # only Z3 makes sense for the inference
            log_warn("ZeRO inference only makes sense with ZeRO Stage 3")
            self.optimizer, self.scheduler = None, None
            model_parameters = None
        else:
            model_parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        kwargs = {
            "model": self.model,  # deepspeed的forward默认是计算到loss输出的
            "model_parameters": model_parameters,
            "config_params": self.config,
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }
        if self.config.get('zero_optimization', {}).get('offload_optimizer', {}).get('device') == 'cpu':
            kwargs.pop('optimizer')
            if self.optimizer is not None:
                self.optimizer = None
                log_warn('You may not use custom optimizer when offload_optimizer=`cpu`')
        self.deepspeed_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(**kwargs)
        self.verbose = 1 if self.deepspeed_engine.local_rank == master_rank else 0

    def unwrap_model(self):
        # 执行deepspeed_engine的forward
        return self.deepspeed_engine

    def loss_backward(self, loss):
        self.deepspeed_engine.backward(loss)
        return loss
    
    def step(self):
        self.deepspeed_engine.step()

    def resume_from_checkpoint(self, *args, **kwargs):
        return self.deepspeed_engine.load_checkpoint(*args, **kwargs)

    def save_to_checkpoint(self, *args, **kwargs):
        return self.deepspeed_engine.save_checkpoint(*args, **kwargs)

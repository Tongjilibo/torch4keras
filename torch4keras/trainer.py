from torch import nn
import torch
from torch4keras.snippets import DottableDict, metric_mapping, get_parameter_device, log_info, log_warn, log_error, seed_everything
from torch4keras.snippets import print_trainable_parameters, colorful, send_email, load_checkpoint, save_checkpoint
from torch4keras.callbacks import KerasProgbar, SmoothMetricsCallback, TqdmProgbar, ProgressBar2Progbar, Callback, CallbackList, History
from collections import OrderedDict
from typing import Union, List, Literal, Tuple, Set, Callable
from inspect import isfunction
import os
import json
import math
import re
import traceback


class Trainer:
    '''Trainer, 传入Module实例

    :param module: None/nn.Module, nn.Module()的模型实例
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
        self.callbacks = []  # 所有的Callbacks, 如果fit中不传入, 则默认为[progbarlogger, smoothmetrics, history]三项
        self.run_callbacks = True  # 是否运行Callbacks, 目前主要是在DDP模式下运用
        self.loss2metrics = True  # 把loss_detail打印在进度条的metrics上
        # add_module(self)  # 增加nn.Module的成员方法

    def compile(self, loss=None, optimizer=None, scheduler=None, clip_grad_norm:float=None, 
                mixed_precision:Literal[True, False, 'fp16', 'bf16']=False, metrics:Union[str, dict, Callable, List[str, dict, Callable]]=None, 
                grad_accumulation_steps:int=1, progbar_type:Literal['keras', 'tqdm', 'progressbar2']='keras', progbar_width:int=None,
                stateful_metrics:Union[str, Set[str], Tuple[str], List[str]]=None, smooth_interval:int=100, **kwargs):
        '''complile: 定义loss, optimizer, metrics等参数
        
        :param loss: loss
        :param optimizer: 优化器
        :param scheduler: lr_scheduler
        :param clip_grad_norm: float, 是否使用梯度裁剪, 默认为False
        :param mixed_precision: bool, 是否使用混合精度, 默认为False
        :param metrics: str/List[str]/dict, 训练过程中需要打印的指标, loss相关指标默认会打印, 目前支持accuracy, 也支持自定义metric, 形式为{key: func}
        :param grad_accumulation_steps: int, 梯度累积步数, 默认为1
        :param bar: str, 使用进度条的种类, 从kwargs中解析, 默认为keras, 可选keras, tqdm, progressbar2

        > 进度条的配置
            progbar_type: str, 默认为keras, 可选keras, tqdm, progressbar2
            width: int, keras进度条下表示进度条的长度
        > 指标平滑的配置, 默认为None表示采取默认平滑设置; 传入False表示不使用平滑
            stateful_metrics: List[str], 表示不使用指标平滑仅进行状态记录的metric, 指标抖动会更加明显, 默认为None表示使用指标平滑
            smooth_interval: int, 表示指标平滑时候的累计步数, 默认为100

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
            # 字符类型, 目前仅支持accuracy
            if isinstance(metric, str) and metric != 'loss':
                self.metrics[metric] = None
            # 字典形式 {metric: func}
            elif isinstance(metric, dict):
                self.metrics.update(metric)
            # 函数形式, key和value都赋值metric
            elif isfunction(metric):
                self.metrics.update({metric: metric})
            else:
                raise TypeError('Args metrics only support `String, Dict, Callable, List[String, Dict, Callable]` format')

        # 进度条参数
        assert progbar_type in {'keras', 'tqdm', 'progressbar2'}
        self.progbar_config = {'bar': progbar_type, 'width': progbar_width}
        self.progbar_config = {k:v for k,v in self.progbar_config.items() if v is not None}

        # smooth_metrics参数: 默认平滑
        self.smooth_metrics_config = {'stateful_metrics': stateful_metrics, 'interval': smooth_interval, 'verbose': kwargs.get('verbose')}
        self.smooth_metrics_config = {k:v for k,v in self.smooth_metrics_config.items() if v is not None}

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

    def _move_to_model_device(self, inputs:Union[torch.Tensor, tuple, list, dict]):
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
        '''调用模型的forward, 方便下游继承的时候可以自定义使用哪个模型的forward
        '''
        return self._argparse_forward(self.unwrap_model(), *inputs, **input_kwargs)

    def _argparse_forward(self, model, *inputs, **input_kwargs):
        '''调用模型的forward
        如果传入了网络结构module, 则调用module的forward; 如果是继承方式, 则调用自身的forward
        这里声明为staticmethod的话, 使用add_trainer会有问题
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
            loss = loss_detail['loss']  # 还存在其他loss, 仅用于打印
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
                torch.nn.utils.clip_grad_norm_(self.unwrap_model().parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            skip_scheduler = self.scaler.get_scale() != self.scale_before_step
        else:
            if self.clip_grad_norm is not None:  # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.unwrap_model().parameters(), self.clip_grad_norm)
            self.optimizer.step()

        self.optimizer.zero_grad()  # 清梯度
        if (self.scheduler is not None) and not skip_scheduler:
            if isinstance(self.scheduler, (tuple, list)):
                for scheduler in self.scheduler:
                    scheduler.step()
            else:
                self.scheduler.step()

    def _prepare_inputs(self, train_dataloader, steps_per_epoch:Union[int,None], epochs:int, verbose:int):
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
        self.train_dataloader = train_dataloader  # 设置为成员变量, 可由外部的callbacks进行修改
        self.train_dataloader_iter = iter(self.train_dataloader)  # 循环epoch时不重生成
        self.verbose = self.verbose if hasattr(self, 'verbose') else verbose

    def _prepare_callbacks(self, callbacks:Union[Callback, List[Callback]]=None):
        '''callbacks设置'''
        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks]
        for callback in callbacks:
            assert isinstance(callback, Callback), "Args `callbacks` only support Callback() inputs"

        history = History()
        callbacks_ = []

        # 指标平滑
        if any([isinstance(i, SmoothMetricsCallback) for i in callbacks]):
            # 用户自定的callbacks中包含了SmoothMetricsCallback
            log_warn(f'SmoothMetricsCallback already in use and args `smooth_metrics_config` will be ignored')
            smooth_callback = [callback for callback in callbacks if isinstance(callback, SmoothMetricsCallback)][0]
            callbacks_.append(smooth_callback)
            callbacks = [callback for callback in callbacks if not isinstance(callback, SmoothMetricsCallback)]
        elif self.smooth_metrics_config.get('interval') is not None:
            smooth_callback = SmoothMetricsCallback(**self.smooth_metrics_config)
            callbacks_.append(smooth_callback)
        else:
            # 不平滑
            smooth_callback = None

        # 检查指标平滑的设置和后续callback的设置的interval是不是一致
        for callback in callbacks:
            if hasattr(callback, 'interval') and (smooth_callback is not None) and (callback != smooth_callback) and \
                (callback.interval is not None) and (callback.interval % smooth_callback.interval != 0):
                log_warn(f'{type(callback).__name__}.interval={callback.interval} while SmoothMetricsCallback.interval={smooth_callback.interval}')
        
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
        # 循环dataloader, 不要试用itertools的cycle, 遇到过变量不释放的问题
        try:
            batch = next(self.train_dataloader_iter)
            self.batch_step += 1
        except StopIteration:
            self.callbacks.on_dataloader_end()  # 适用于数据量较大时, 动态读取文件并重新生成self.train_dataloader的情况, 如预训练
            # DDP训练时候为了避免每个epoch样本一致, 修改随机种子
            if isinstance(self.train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler) and \
                hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(self.epoch)
            self.train_dataloader_iter = iter(self.train_dataloader)  # shuffle=True时候, 其实顺序也重新生成了
            self.batch_step = 0
            batch = next(self.train_dataloader_iter)

        batch = self._move_to_model_device(batch)
        return batch

    def fit(self, train_dataloader, steps_per_epoch:int=None, epochs:int=1, callbacks:Union[list]=None, verbose:int=1, **kwargs):
        ''' 模型训练
        :param train_dataloader: Dataloader, 训练数据集
        :param steps_per_epoch: int, 每个epoch训练的steps, 默认为None表示自行计算 
        :param epochs: int, 训练的轮次, 默认为1
        :param callbacks: Callback/List[Callback], 回调函数, 可调用预制的Callback或者自定义, 默认为None 
        :param verbose: int, 是否打印, 默认为1表示打印
        
        > 其他参数
        :param mail_receivers: str, 发生异常的时候邮件通知
        :param save_dir_when_error: str, 发生异常时候保存权重的路径
        
        :return: History
        '''
        try:
            return self._fit(train_dataloader, steps_per_epoch, epochs, callbacks, verbose, **kwargs)
        except Exception as e:
            # 训练异常则发邮件
            error_msg = traceback.format_exc()
            mail_receivers_ = kwargs.get('mail_receivers')
            if mail_receivers_ is not None:
                mail_subject_ = kwargs.get('mail_subject') or "[ERROR] fit"
                mail_host_ = kwargs.get('mail_host')
                mail_user_ = kwargs.get('mail_user')
                mail_pwd_ = kwargs.get('mail_pwd')
                mail_sender_ = kwargs.get('mail_sender')
                send_email(mail_receivers_, mail_subject_, error_msg, mail_host=mail_host_, 
                           mail_user=mail_user_, mail_pwd=mail_pwd_, mail_sender=mail_sender_)

            # 训练异常则保存权重
            if (save_dir_when_error := kwargs.get('save_dir_when_error')) is not None:
                self.save_to_checkpoint(save_dir_when_error, verbose=verbose, **kwargs)
            raise e

    def _fit(self, train_dataloader, steps_per_epoch:int=None, epochs:int=1, callbacks:Union[list]=None, verbose:int=1, **kwargs):
        '''模型训练'''
        # 输入处理
        self._prepare_inputs(train_dataloader, steps_per_epoch, epochs, verbose)

        # 准备callbacks
        history, callback_trainer, progbarlogger  = self._prepare_callbacks(callbacks)

        #       epoch: 当前epoch
        # global_step: 当前全局训练步数
        #  local_step: 当前epoch内的训练步数, 不同epoch中相同local_step对应的batch数据不一定相同, 在steps_per_epoch=None时相同
        #  batch_step: 在dataloader中的index, 不同epoch中相同的bti对应的batch数据一般相同, 除非重新生成dataloader
        self.callbacks.on_train_begin()
        logs = self._log_init()  # 防止数据集为空时候
        for epoch in range(self.resume_epoch, epochs):
            self.epoch = epoch
            # resume_step：判断local_step的起点, 以及进度条的起始位置
            resume_step = self.resume_step if epoch==self.resume_epoch else 0
            self.callbacks.on_epoch_begin(self.global_step, self.epoch)
            if self.verbose:
                progbarlogger.seen = resume_step  # 这里设置进度条的seen, 在callbacks中也会修改
            
            for local_step in range(resume_step, self.steps_per_epoch):
                self.local_step = local_step
                self.global_step = self.epoch * self.steps_per_epoch + self.local_step
                logs = self._log_init()
                self.callbacks.on_batch_begin(self.global_step, self.local_step, logs)

                # forward和backward
                if not self.unwrap_model().training:
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
                # TODO: 理论上梯度累积时需对output和train_y进行合并, 主要是为了metric_mapping计算的准确
                
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
        '''获取batch_size, 主要是用于callback中的BaseLogger和Callback
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
        '''模型预测, 调用forward()'''
        self.unwrap_model().eval()
        inputs = self._move_to_model_device(inputs)
        input_kwargs = self._move_to_model_device(input_kwargs)
        return self._forward(*inputs, **input_kwargs)
        
    def load_steps_params(self, save_path:str):
        '''导入训练过程参数
        
        :param save_path: str, 训练过程参数保存路径
        '''
        step_params = torch.load(save_path)
        self.resume_step = step_params['resume_step'] 
        self.resume_epoch = step_params['resume_epoch']
        return step_params

    def save_steps_params(self, save_path:str):
        '''保存训练过程参数

        :param save_path: str, 训练过程参数保存路径
        '''
        step_params = {'resume_step': (self.local_step+1) % self.steps_per_epoch, 
                       'resume_epoch': self.epoch + (self.local_step+1) // self.steps_per_epoch}
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(step_params, save_path)

    def load_weights(self, load_path:Union[str,tuple,list], strict:bool=True, mapping:Union[dict,Callable]=None):
        '''加载模型权重, 支持加载权重文件list

        :param save_path: str/tuple/list, 权重加载路径
        :param strict: bool, torch.load()是否严格加载
        :param mapping: dict/func, 指定key的映射
            1. mapping=None, 表示按照模型自身结构加载, 一般加载finetune后使用save_weights()保存出来的权重
            2. mapping自定义, 根据用户自定义mapping来加载权重
        '''
        if isinstance(load_path, (tuple, list)):
            strict = False  # 加载多个权重文件时候, strict设置为False
        elif isinstance(load_path, str):
            load_path = [load_path]
        else:
            raise ValueError('Args `load_path` only support str/tuple/list format')
        
        mapping = mapping or dict()
        for load_path_i in load_path:
            state_dict = load_checkpoint(load_path_i)
            for k in list(state_dict.keys()):
                if isinstance(mapping, dict) and k in mapping:
                    state_dict[mapping[k]] = state_dict.pop(k)
                elif isinstance(mapping, Callable):
                    state_dict[mapping(k)] = state_dict.pop(k)
            self.unwrap_model().load_state_dict(state_dict, strict=strict)

    def save_weights(self, save_path:str, mapping:Union[dict,Callable]=None, trainable_only:bool=False):
        '''保存模型权重

        :param save_path: str, 权重保存路径
        :param mapping: dict/func, 指定key的映射
            1. mapping=None, 表示按照模型自身结构的key来保存, 后续可直接load_weights()加载
            2. mapping自定义, 根据用户自定义mapping来保存权重
        :param trainable_only: bool, 指定仅保存可训练参数
        '''
        state_dict = self.unwrap_model().state_dict()
        trainable_parameters = set(p for p,v in self.unwrap_model().named_parameters() if v.requires_grad)
        
        mapping = mapping or dict()
        for k in list(state_dict.keys()):
            # 只保存可训练的模型部分
            if trainable_only and (k not in trainable_parameters):
                continue
            if isinstance(mapping, dict) and k in mapping:
                state_dict[mapping[k]] = state_dict.pop(k)
            elif isinstance(mapping, Callable):
                state_dict[mapping(k)] = state_dict.pop(k)        
        save_checkpoint(state_dict, save_path)
        if trainable_only:
            params_all = sum(p.numel() for p in self.unwrap_model().parameters())
            params_trainable = sum(p.numel() for p in self.unwrap_model().parameters() if p.requires_grad)
            ratio = params_trainable/params_all*100
            log_info(f"Only trainable parameters saved and occupy {params_trainable}/{params_all}={ratio:.2f}%")

    def save_pretrained(self, save_path:str, weight_map:dict=None, mapping:Union[dict,Callable]=None):
        '''按照预训练模型的key来保存模型, 可供transformers包加载

        :param save_path: str, 保存的文件/文件夹路径
        '''
        state_dict = dict()
        for name, child in self.unwrap_model().named_children():
            if (name != '') and hasattr(child, 'save_pretrained'):
                tmp = child.save_pretrained(save_path, weight_map, mapping, write_to_disk=False)
                state_dict.update(tmp if tmp else {})
            else:
                state_dict.update({f'{name}.{k}': v for k,v in child.state_dict().items()})
        if len(state_dict) > 0:
            save_dir = None if re.search('\.[a-zA-z0-9]+$', save_path) else save_path
            save_checkpoint(state_dict, os.path.join(save_dir, 'pytorch_model.bin') if save_dir else save_path)
    
    def resume_from_checkpoint(self, save_dir:str=None, model_path:str=None, optimizer_path:str=None, scheduler_path:str=None, 
                               steps_params_path:str=None, mapping:Union[dict,Callable]=None, verbose:int=0, strict:bool=True, 
                               device=None, **kwargs):
        '''同时加载模型、优化器、训练过程参数

        :param save_dir: str, 保存目录
        :param model_path: str, 模型文件路径
        :param optimizer_path: str, 优化器文件路径
        :param scheduler_path: str, scheduler文件路径
        :param steps_params_path: str, 训练过程参数保存路径
        :param mapping: dict, 模型文件的mapping
        '''
        # 加载模型权重
        if model_path or save_dir:
            model_path = model_path or os.path.join(save_dir, 'model.pt')
            self.load_weights(model_path, strict=strict, mapping=mapping)
            if verbose == 1:
                log_info(f'Model weights successfuly resumed from {model_path}')

        # 加载优化器
        if optimizer_path or save_dir:
            optimizer_path = optimizer_path or os.path.join(save_dir, 'optimizer.pt')
            state_dict = torch.load(optimizer_path, map_location = device or self.device)
            self.optimizer.load_state_dict(state_dict)
            if verbose == 1:
                log_info(f'Optimizer successfuly resumed from {optimizer_path}')

        # 加载scheduler
        if (scheduler_path or save_dir) and (self.scheduler is not None):
            scheduler_path = scheduler_path or os.path.join(save_dir, 'scheduler.pt')
            state_dict = torch.load(scheduler_path, map_location = device or self.device)
            self.scheduler.load_state_dict(state_dict)
            if verbose == 1:
                log_info(f'Scheduler successfuly resumed from {scheduler_path}')

        # 加载训练进度参数
        if steps_params_path or save_dir:
            steps_params_path = steps_params_path or os.path.join(save_dir, 'steps_params.pt')
            self.load_steps_params(steps_params_path)
            if verbose == 1:
                log_info(f'Steps_params successfuly resumed from {steps_params_path}')

    def save_to_checkpoint(self, save_dir:str=None, model_path:str=None, optimizer_path:str=None, scheduler_path:str=None, 
                           steps_params_path:str=None, mapping:Union[dict,Callable]=None, trainable_only:bool=False, verbose:int=0, **kwargs):
        '''同时保存模型、优化器、训练过程参数、scheduler

        :param save_dir: str, 保存目录
        :param model_path: str, 模型文件路径
        :param optimizer_path: str, 优化器文件路径
        :param scheduler_path: str, scheduler文件路径
        :param steps_params_path: str, 训练过程参数保存路径
        :param mapping: dict/func, 模型文件的mapping
        :param trainable_only
        '''
        if model_path or save_dir:
            model_path = model_path or os.path.join(save_dir, 'model.pt')
            self.save_weights(model_path, mapping=mapping, trainable_only=trainable_only)
            if verbose == 1:
                log_info(f'Model weights successfuly saved to {model_path}')

        if optimizer_path or save_dir:
            optimizer_path = optimizer_path or os.path.join(save_dir, 'optimizer.pt')
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            if verbose == 1:
                log_info(f'Optimizer successfuly saved to {optimizer_path}')

        if (scheduler_path or save_dir) and (self.scheduler is not None):
            scheduler_path = scheduler_path or os.path.join(save_dir, 'scheduler.pt')
            os.makedirs(os.path.dirname(scheduler_path), exist_ok=True)
            torch.save(self.scheduler.state_dict(), scheduler_path)
            if verbose == 1:
                log_info(f'Scheduler successfuly saved to {scheduler_path}')

        if steps_params_path or save_dir:
            steps_params_path = steps_params_path or os.path.join(save_dir, 'steps_params.pt')
            self.save_steps_params(steps_params_path)
            if verbose == 1:
                log_info(f'Steps_params successfuly saved to {steps_params_path}')

    def unwrap_model(self):
        '''返回nn.Module模块
        '''
        if isinstance(self, nn.Module): return self
        return self.module if hasattr(self, 'module') else self


Trainer.compile_training_components = Trainer.compile


class TrainerDP(nn.DataParallel, Trainer):
    '''DataParallel模式使用多gpu的方法, 
    1) 父类顺序颠倒也会出问题
    2) 使用方式和nn.DataParallel一致, TrainerDP(net, *args, **kwargs)来使用
    '''
    def __init__(self, *args, **kwargs):
        Trainer.__init__(self)
        nn.DataParallel.__init__(self, *args, **kwargs)


class TrainerDDP(nn.parallel.DistributedDataParallel, Trainer):
    '''DistributedDataParallel模式使用多gpu的方法,
    1) 父类顺序颠倒也会出问题
    2) 使用方式和DistributedDataParallel一致, TrainerDDP(net, *args, **kwargs)来使用
    '''
    def __init__(self, *args, master_rank=0, **kwargs):
        Trainer.__init__(self)
        nn.parallel.DistributedDataParallel.__init__(self, *args, **kwargs)

        # 默认仅对master_rank=0打印信息
        assert isinstance(master_rank, (int, list, tuple)), 'Args `master_rank` only supoorts int, list, tuple'
        if isinstance(master_rank, int):
            master_rank = [master_rank]
        self.master_rank = master_rank
        self.verbose = (torch.distributed.get_rank() in master_rank)
    
    def _prepare_inputs(self, *args):
        super()._prepare_inputs(*args)
        from torch.utils.data.distributed import DistributedSampler 
        if (self.train_dataloader.sampler is None) and (not isinstance(self.train_dataloader.sampler, DistributedSampler)):
            self.train_dataloader.sampler = DistributedSampler(self.train_dataloader.dataset)
            self.train_dataloader_iter = iter(self.train_dataloader)
    
    def disable_run_callbacks(self, callbacks: Union[Callback, List[Callback]]):
        for callback in callbacks:
            if torch.distributed.get_rank() not in self.master_rank:
                callback.run_callback = False

    @classmethod
    def init_process_group(master_rank=0, seed=42):
        if os.name == 'nt':
            # windows: Diff between backends: https://pytorch.org/docs/stable/distributed.html
            torch.distributed.init_process_group(backend="gloo")
        else:  # linux
            torch.distributed.init_process_group(backend='nccl')
        
        ddp = DottableDict()
        ddp.rank = int(os.environ["RANK"])
        ddp.local_rank = int(os.getenv('LOCAL_RANK'))
        ddp.device = torch.device('cuda', ddp.local_rank)
        ddp.world_size = int(os.environ["WORLD_SIZE"])
        ddp.master_process = ddp.rank == master_rank
        torch.cuda.set_device(ddp.local_rank)
        seed_everything(seed + ddp.rank)
        return ddp


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


class DeepSpeedTrainer(Trainer):
    '''deepspeed来训练'''
    def __init__(self, module, config_path):
        super().__init__(module)
        self.model = module
        self.config = DottableDict(json.load(open(config_path)))
        self.config['steps_per_print'] = self.config.get('steps_per_print', 1e9)  # 默认不打印, 防止进度条打印问题

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


def add_trainer(obj, include=None, exclude=None, verbose=0, replace_func=False):
    '''为nn.Module添加Triner对应的方法'''
    if isinstance(obj, (Trainer, TrainerDP, TrainerDDP)):
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

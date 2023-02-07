from torch import nn
import torch
from torch4keras.snippets import metric_mapping, ProgbarLogger, Callback, CallbackList, BaseLogger, History, TqdmProgressBar
from collections import OrderedDict
from inspect import isfunction
import os


class Trainer:
    """Trainer, 传入Module实例

    :param module: None/nn.Module，nn.Module()的模型实例
    """
    def __init__(self, module=None):
        super(Trainer, self).__init__()
        self.initialize(module)
    
    def initialize(self, module=None):
        # 这里主要是为了外面调用用到
        self.global_step, self.local_step, self.total_steps, self.epoch, self.steps_per_epoch, self.train_dataloader = 0, 0, 0, 0, None, None
        self.resume_step, self.resume_epoch = 0, 0
        self.retain_graph = False  # loss.backward()是否保留计算图
        self.callbacks = []
        # 传入Module实例方式
        if module is not None:
            assert isinstance(module, nn.Module), 'Args `module` only support nn.Module format'
            self.module = module
        # 是否运行Callbacks，目前主要是在DDP模式下运用
        self.run_callbacks = True

    def compile(self, loss, optimizer, scheduler=None, clip_grad_norm=None, mixed_precision=False, metrics=None, 
                stateful_metrics=None, grad_accumulation_steps=1, **kwargs):
        '''complile: 定义loss, optimizer, metrics等参数
        
        :param loss: loss
        :param optimizer: 优化器
        :param scheduler: scheduler
        :param clip_grad_norm: bool, 是否使用梯度裁剪, 默认为False
        :param mixed_precision: bool, 是否使用混合精度，默认为False
        :param metrics: str/List[str]/dict, 训练过程中需要打印的指标, loss相关指标默认会打印, 目前支持accuracy, 也支持自定义metric，形式为{key: func}
        :param stateful_metrics: List[str], 不滑动平均仅进行状态记录的metric，指标抖动会更加明显
        :param grad_accumulation_steps: int, 梯度累积步数，默认为1
        :param tqdmbar: bool, 是否使用tqdm进度条，从kwargs中解析，默认为False
        :return: None
        '''
        self.criterion = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.mixed_precision = mixed_precision
        if mixed_precision:
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
        self.stateful_metrics = stateful_metrics

        # 梯度累积
        self.grad_accumulation_steps = grad_accumulation_steps

        # 进度条参数
        self.tqdmbar = kwargs.get('tqdmbar', False)

    def args_segmentate(self, train_X):
        '''参数是否展开
        '''
        if isinstance(train_X, torch.Tensor):  # tensor不展开
            pass
        elif self.get_module().forward.__code__.co_argcount >= 3:
            return True
        return False
    
    def forward(self, *inputs, **kwargs):
        # 如果传入了网络结构module，则调用module的forward
        # 如果是继承方式，则调用自身的forward
        return self.get_module().forward(*inputs, **kwargs)

    def train_step(self, train_X, train_y):
        '''forward并返回loss

        :param train_X: List[torch.Tensor], 训练数据
        :param train_y: torch.Tensor/List[torch.Tensor], 标签信息
        :return: [output, loss, loss_detail] output: torch.Tensor/List[torch.Tensor], 模型输出; loss: nn.Tensor, 计算得到的loss; loss_detail: dict[nn.Tensor], 具体的各个loss
        '''
        # 计算loss
        if self.mixed_precision:
            with self.autocast():
                output = self.forward(*train_X) if self.args_segmentate(train_X) else self.forward(train_X)
                loss_detail = self.criterion(output, train_y)
        else:
            output = self.forward(*train_X) if self.args_segmentate(train_X) else self.forward(train_X)
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

        # loss.backward
        self.scale_before_step = 0
        if self.mixed_precision:  # 混合精度
            self.scale_before_step = self.scaler.get_scale()
            self.scaler.scale(loss).backward(retain_graph=self.retain_graph)
        else:
            loss.backward(retain_graph=self.retain_graph)
        return output, loss, loss_detail

    def fit(self, train_dataloader, steps_per_epoch=None, epochs=1, callbacks=None, verbose=1):
        '''模型训练
        
        :param train_dataloader: Dataloader, 训练数据集
        :param steps_per_epoch: int, 每个epoch训练的steps，默认为None表示自行计算 
        :param epochs: int, 训练的轮次, 默认为1
        :param callbacks: Callback/List[Callback], 回调函数，可调用预制的Callback或者自定义，默认为None 
        :param verbose: int, 是否打印，默认为1表示打印
        :return: None
        '''
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
        
        # 进度条
        if self.tqdmbar:
            progbarlogger = TqdmProgressBar(stateful_metrics=self.stateful_metrics)
        else:
            progbarlogger = ProgbarLogger(stateful_metrics=self.stateful_metrics)
            
        history = History()
        callbacks_ = [BaseLogger(self.stateful_metrics), progbarlogger] + callbacks + [history]
        self.callbacks = CallbackList(callbacks_, run_callbacks=self.run_callbacks)
        callback_trainer = self
        callback_model = self.get_module()
        self.callbacks.set_trainer(callback_trainer)
        self.callbacks.set_model(callback_model)
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

                self.get_module().train()  # 设置为train模式

                # 入参个数判断，如果入参>=3表示是多个入参，如果=2则表示是一个入参
                self.output, self.loss, self.loss_detail = self.train_step(self.train_X, self.train_y)
                self.callbacks.on_train_step_end()
                                
                # 参数更新, 真实的参数更新次数要除以grad_accumulation_steps，注意调整总的训练步数
                if (self.global_step+1) % self.grad_accumulation_steps == 0:
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
        '''模型预测，调用forward()

        :param train_X: torch.Tensor, 预测用的数据集
        :param return_all: None/int, 若返回为多个时候指定仅返回第几个，默认为None表示全部返回
        :return: Any, 预测输出
        '''
        self.get_module().eval()
        output = self.forward(*train_X) if self.args_segmentate(train_X) else self.forward(train_X)
        if return_all is None:
            return output
        elif isinstance(output, (tuple, list)) and isinstance(return_all, int) and return_all < len(output):
            return output[return_all]
        else:
            raise ValueError('Return format error')

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
        torch.save(step_params, save_path)

    def load_weights(self, load_path, strict=True, mapping={}):
        '''加载模型权重

        :param save_path: str, 权重加载路径
        :param mapping: dict, 指定key的映射
        '''
        state_dict = torch.load(load_path, map_location='cpu')
        state_dict_raw = {}
        for k, v in state_dict.items():
            k = mapping.get(k, k)
            state_dict_raw[k] = v
        self.get_module().load_state_dict(state_dict_raw, strict=strict)

    def save_weights(self, save_path, mapping={}):
        '''保存模型权重

        :param save_path: str, 权重保存路径
        :param mapping: dict, 指定key的映射
        '''
        state_dict_raw = {}
        state_dict = self.get_module().state_dict()
        for k, v in state_dict.items():
            k = mapping.get(k, k)
            state_dict_raw[k] = v
        torch.save(state_dict_raw, save_path)

    def resume_from_checkpoint(self, model_path=None, optimizer_path=None, step_params_path=None):
        '''同时加载模型、优化器、训练过程参数

        :param model_path: str, 模型文件路径
        :param optimizer_path: str, 优化器文件路径
        :param step_params_path: str, 训练过程参数保存路径
        '''
        # 加载模型权重
        if model_path:
            self.load_weights(model_path)
        # 加载优化器，断点续训使用
        if optimizer_path:
            state_dict = torch.load(optimizer_path, map_location='cpu')
            self.optimizer.load_state_dict(state_dict)
        # 加载训练进度参数，断点续训使用
        self.load_steps_params(step_params_path)

    def save_to_checkpoint(self, model_path=None, optimizer_path=None, step_params_path=None):
        '''同时保存模型、优化器、训练过程参数

        :param model_path: str, 模型文件路径
        :param optimizer_path: str, 优化器文件路径
        :param step_params_path: str, 训练过程参数保存路径
        '''
        if model_path:
            save_dir = os.path.dirname(model_path)
            os.makedirs(save_dir, exist_ok=True)
            self.save_weights(model_path)
        if optimizer_path:
            save_dir = os.path.dirname(optimizer_path)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.optimizer.state_dict(), optimizer_path)
        if step_params_path:
            save_dir = os.path.dirname(step_params_path)
            os.makedirs(save_dir, exist_ok=True)
            self.save_steps_params(step_params_path)

    def get_module(self):
        '''返回nn.Module模块
        '''
        return self.module if hasattr(self, 'module') else self


class BaseModel(Trainer, nn.Module):
    """BaseModel, 支持继承、传入Module实例两种方式, 建议使用继承的方式来使用
    """
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        Trainer.__init__(self, *args, **kwargs)
        

class BaseModelDP(nn.DataParallel, BaseModel):
    '''DataParallel模式使用多gpu的方法, 父类顺序颠倒也会出问题
    '''
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self)
        nn.DataParallel.__init__(self, *args, **kwargs)


class BaseModelDDP(nn.parallel.DistributedDataParallel, BaseModel):
    '''DistributedDataParallel模式使用多gpu的方法, 父类顺序颠倒也会出问题
    '''
    def __init__(self, *args, master_rank=0, **kwargs):
        BaseModel.__init__(self)
        nn.parallel.DistributedDataParallel.__init__(self, *args, **kwargs)
        self.run_callbacks = (master_rank==torch.distributed.get_rank())  # 用于记录打印条的rank


TrainerDP = BaseModelDP
TrainerDDP = BaseModelDDP


def add_trainer(obj):
    '''为对象添加Triner对应的方法
    '''
    if isinstance(obj, (Trainer, TrainerDP, TrainerDDP, BaseModel, BaseModelDP, BaseModelDDP)):
        return obj
    
    if isinstance(obj, nn.Module):
        import types
        for k in dir(Trainer):
            if k.startswith('__') and k.endswith('__'):
                continue
            elif (k == 'forward') and hasattr(obj, 'forward'):
                continue
            exec(f'obj.{k} = types.MethodType(Trainer.{k}, obj)')
        obj.initialize()
    return obj
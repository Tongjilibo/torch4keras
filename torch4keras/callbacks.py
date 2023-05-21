import sys
import collections
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
from collections import deque
import json
import copy
import os
from torch4keras.snippets import send_email


class Progbar(object):
    """进度条，直接从keras引入
    """
    def __init__(self, target, width=30, verbose=1, interval=0.05, stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()) or 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class CallbackList(object):
    '''把原来在model.py中的callback_fun移植出来，参考Keras的CallbackList重构
    '''
    def __init__(self, callbacks=None, queue_length=10, run_callbacks=True):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length
        self.run_callbacks = run_callbacks  # 控制全部开启/关闭

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def set_optimizer(self, optimizer):
        for callback in self.callbacks:
            callback.set_optimizer(optimizer)

    def set_scheduler(self, scheduler):
        for callback in self.callbacks:
            callback.set_scheduler(scheduler)

    def set_all(self, trainer=None, model=None, optimizer=None, scheduler=None, params=None):
        for callback in self.callbacks:
            callback.set_trainer(trainer)
            callback.set_model(model)
            callback.set_optimizer(optimizer)
            callback.set_scheduler(scheduler)
            callback.set_params(params)

    def on_epoch_begin(self, global_step, epoch, logs=None):
        # 如果是分布式DDP训练，则仅masker_rank可以callback
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): return
            callback.on_epoch_begin(global_step, epoch, logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

    def on_epoch_end(self, global_step, epoch, logs=None):
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): return
            callback.on_epoch_end(global_step, epoch, logs)

    def on_batch_begin(self, global_step, local_step, logs=None):
        if not self.run_callbacks: return
        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): return
            callback.on_batch_begin(global_step, local_step, logs)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_begin)
        if (self._delta_t_batch > 0. and delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1):
            warnings.warn(f'Method on_batch_begin() is slow compared to the batch update {delta_t_median}. Check your callbacks.')
        self._t_enter_batch = time.time()

    def on_batch_end(self, global_step, local_step, logs=None):
        if not self.run_callbacks: return
        logs = logs or {}
        if not hasattr(self, '_t_enter_batch'):
            self._t_enter_batch = time.time()
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): return
            callback.on_batch_end(global_step, local_step, logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if (self._delta_t_batch > 0. and (delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1)):
            warnings.warn(f'Method on_batch_end() is slow compared to the batch update {delta_t_median}. Check your callbacks.')

    def on_train_begin(self, logs=None):
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): return
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): return
            callback.on_train_end(logs)

    def on_dataloader_end(self, logs=None):
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): return
            callback.on_dataloader_end(logs)

    def on_train_step_end(self, logs=None):
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): return
            callback.on_train_step_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    '''Callback基类
    '''
    def __init__(self, run_callback=True, **kwargs):
        self.trainer = None  # trainer
        self.model = None  # nn.Module模型，或者包含Trainer的nn.Module
        self.optimizer = None  # 优化器
        self.run_callback = run_callback  # 是否运行该callback
    def set_params(self, params):
        self.params = params
    def set_trainer(self, trainer):
        self.trainer = trainer
    def set_model(self, model):
        self.model = model
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
    def on_train_begin(self, logs=None):
        pass
    def on_train_end(self, logs=None):
        pass
    def on_epoch_begin(self, global_step, epoch, logs=None):
        pass
    def on_epoch_end(self, global_step, epoch, logs=None):
        pass
    def on_batch_begin(self, global_step, local_step, logs=None):
        pass
    def on_batch_end(self, global_step, local_step, logs=None):
        pass
    def on_dataloader_end(self, logs=None):
        pass
    def on_train_step_end(self, logs=None):
        pass


class BaseLogger(Callback):
    """计算metrics的均值, 默认是callbacks中第一项, 主要是为History服务
    
    :param stateful_metrics: List[str], 仅保留状态信息的指标
    """
    def __init__(self, stateful_metrics=None, **kwargs):
        super(BaseLogger, self).__init__(**kwargs)
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

    def on_epoch_begin(self, global_step, epoch, logs=None):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, global_step, local_step, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.stateful_metrics:
                self.totals[k] = v
            else:
                if k in self.totals:
                    self.totals[k] += v * batch_size
                else:
                    self.totals[k] = v * batch_size

    def on_epoch_end(self, global_step, epoch, logs=None):
        '''在epoch_end对指标计算epoch的均值
        '''
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    if k in self.stateful_metrics:
                        logs[k] = self.totals[k]
                    else:
                        logs[k] = self.totals[k] / self.seen


class TerminateOnNaN(Callback):
    """Loss出现NAN停止训练
    """
    def on_batch_end(self, global_step, local_step, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print('Step %d: Invalid loss, terminating training' % global_step)
                self.trainer.stop_training = True


class ProgbarLogger(Callback):
    """ keras进度条
    """
    def __init__(self, stateful_metrics=None, **kwargs):
        super(ProgbarLogger, self).__init__(**kwargs)
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

    def add_metrics(self, metrics, stateful_metrics=None, add_position=None):
        '''在指定位置插入metrics指标
        '''
        if add_position is None:
            add_position = len(self.params['metrics'])
        metrics = [metrics] if isinstance(metrics, str) else metrics
        if stateful_metrics:
            stateful_metrics = [stateful_metrics] if isinstance(stateful_metrics, str) else stateful_metrics
            self.stateful_metrics.update(set(stateful_metrics))
            self.progbar.stateful_metrics.update(set(stateful_metrics))

        add_metrics = []
        for metric in metrics:
            if metric not in self.params['metrics']:
                add_metrics.append(metric)
        self.params['metrics'] = self.params['metrics'][:add_position] + add_metrics + self.params['metrics'][add_position:]

    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']
        if self.verbose:
            time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('%s - Start Training' % (time_start))

    def on_epoch_begin(self, global_step=None, epoch=None, logs=None):
        if self.verbose:
            time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('%s - Epoch: %d/%d' % (time_start, epoch+1, self.epochs))
            self.target = self.params['steps']
            self.progbar = Progbar(target=self.target, verbose=self.verbose, stateful_metrics=self.stateful_metrics)
        self.seen = 0

    def on_batch_begin(self, global_step=None, local_step=None, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, global_step=None, local_step=None, logs=None):
        logs = logs or {}
        self.seen += 1
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, global_step=None, epoch=None, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values)
    
    def on_train_end(self, logs=None):
        if self.verbose:
            time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('%s - Finish Training' % (time_start))


class TqdmProgressBar(ProgbarLogger):
    """ Tqdm进度条
    """
    def on_epoch_begin(self, global_step=None, epoch=None, logs=None):
        if self.verbose:
            time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('%s - Epoch: %d/%d' % (time_start, epoch+1, self.epochs))
            self.target = self.params['steps']
            self.progbar = tqdm(total=self.params['steps'], desc='Training', dynamic_ncols=False, file=sys.stdout, smoothing=0)
        self.seen = 0
        self._values = collections.OrderedDict()
        self._seen_so_far = 0

    def on_batch_begin(self, global_step=None, local_step=None, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, global_step=None, local_step=None, logs=None):
        self.seen += 1
        logs = self.smooth_values(self.seen, logs or {})
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.progbar.n = self.seen
            self.progbar.refresh()
            self.progbar.set_postfix(self.log_values)

    def on_epoch_end(self, global_step=None, epoch=None, logs=None):
        logs = self.smooth_values(self.seen, logs or {})
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.n = self.seen
            self.progbar.refresh()
            self.progbar.set_postfix(self.log_values)
            self.progbar.close()
    
    def smooth_values(self, current, values=None):
        '''从Progbar迁移过来
        '''
        values = values or []
        for k, v in values.items():
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far), current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = [v, 1]
        self._seen_so_far = current

        logs = collections.OrderedDict()
        for k in self._values:
            if isinstance(self._values[k], list):
                avg = np.mean(
                    self._values[k][0] / max(1, self._values[k][1]))
                if abs(avg) > 1e-3:
                    logs[k] = ' %.4f' % avg
                else:
                    logs[k] = ' %.4e' % avg
            else:
                logs[k] = ' %s' % self._values[k]

        return logs


class History(Callback):
    """指标历史，默认是fit的返回项, 这里仅记录epoch_end的指标
    """
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, global_step, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch+1)  # 这里和keras相比+1了
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class EarlyStopping(Callback):
    '''Stop training策略, 从keras中移植
       使用时候需要保证monitor在logs中，因此如果以valid指标评估需要在EarlyStopping前进行评估并设置logs[monitor]

       :param monitor: str, 监控指标，需要在logs中，默认为'loss', 
       :param min_delta: float, 最小变动，默认为0 
       :param patience: int, 最长等候的次数，默认为0
       :param verbose: int, 是否打印，默认为0表示不打印
       :param mode: str, 控制监控指标monitor的大小方向，默认为'auto', 可选{'auto', 'min', 'max'}
       :param method: str, 控制是按照epoch还是step来计算，默认为'epoch', 可选{'step', 'epoch'}
       :param baseline: None/float, 基线, 默认为None 
       :param restore_best_weights: bool, stopping时候是否恢复最优的权重，默认为False
    '''
    def __init__(self, monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto', method='epoch', baseline=None, restore_best_weights=False, **kwargs):
        super(EarlyStopping, self).__init__(**kwargs)
        assert method in {'step', 'epoch'}, 'Args `method` only support `step` or `epoch`'
        self.method = method  # 默认的epoch和原版一样
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience  # method=step时候表示最多wait的训练步数
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_iteration = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in {'auto', 'min', 'max'}:
            warnings.warn('EarlyStopping mode %s is unknown, fallback to auto mode.' % mode, RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.greater if 'acc' in self.monitor else np.less
        self.min_delta = self.min_delta if self.monitor_op == np.greater else -self.min_delta

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_iteration = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_batch_end(self, global_step, local_step, logs=None):
        if self.method == 'step':
            return self.process(global_step, logs)

    def on_epoch_end(self, global_step, epoch, logs=None):
        if self.method == 'epoch':
            self.process(epoch, logs)

    def process(self, iteration, logs=None):
        logs =  logs or {}
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(self.model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_iteration = iteration
                self.trainer.stop_training = True
                # 恢复最优权重
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best iteration')
                    self.model.load_state_dict(self.best_weights, strict=True)

    def on_train_end(self, logs=None):
        if self.stopped_iteration > 0 and self.verbose > 0:
            print(f'Iteration {self.stopped_iteration+1}: early stopping\n')

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn('Early stopping conditioned on metric `%s` which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)
        return monitor_value


class ReduceLROnPlateau(Callback):
    """当monitor指标不下降时候，降低学习率

    :param monitor: str, 监控指标，需要在logs中，默认为'loss'
    :param factor: float, 权重衰减系数，取值范围(0, 1)，默认为0.1
    :param patience: int, 最长等候的次数，默认为0
    :param method: str, 控制是按照epoch还是step来计算，默认为'epoch', 可选{'step', 'epoch'}
    :param verbose: int, 是否打印，默认为0表示不打印
    :param mode: str, 控制监控指标monitor的大小方向，默认为'auto', 可选{'auto', 'min', 'max'}
    :param min_delta: float, 最小变动，默认为0 
    :param cooldown: float
    :param min_lr: float, 最小学习率
    """
    def __init__(self, monitor='loss', factor=0.1, patience=10, method='epoch', 
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0,
                 **kwargs):
        super(ReduceLROnPlateau, self).__init__(**kwargs)
        assert method in {'step', 'epoch'}, 'Args `method` only support `step` or `epoch`'
        self.method = method  # 默认的epoch和原版一样
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('`epsilon` argument is deprecated and will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience  # method=step时候表示最多wait的训练步数
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, fallback to auto mode.' % (self.mode), RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_batch_end(self, global_step, local_step, logs=None):
        if self.method == 'step':
            return self.process(global_step, logs)

    def on_epoch_end(self, global_step, epoch, logs=None):
        if self.method == 'epoch':
            self.process(epoch, logs)

    def process(self, iteration, logs=None):
        logs = logs or {}
        for i, params in enumerate(self.optimizer.param_groups):
            if i == 0:
                logs['lr'] = params["lr"]  # 默认第一个param_group作为lr
            else:
                logs[f'lr_param_group{i}'] = params["lr"]

        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Reduce LR on plateau conditioned on metric `%s` which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    for params in self.optimizer.param_groups:
                        if 'lr' not in params:
                            continue
                        old_lr = float(params["lr"])
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            params["lr"] = new_lr
                            if self.verbose > 0:
                                print('\nEpoch %05d: ReduceLROnPlateau reducing learning rate to %s.' % (iteration + 1, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0

        
class RemoteMonitor(Callback):
    """Callback used to stream events to a server.

    :param root: str, url+port
    :param path: str, router
    :param field: str, 字段名
    :param headers: header
    :param send_as_json: bool, 是否以json形式请求，默认为False
    """
    def __init__(self, root='http://localhost:9000', path='/publish/epoch/end/', field='data',
                 headers=None, send_as_json=False, **kwargs):
        super(RemoteMonitor, self).__init__(**kwargs)
        self.root = root
        self.path = path
        self.field = field
        self.headers = headers
        self.send_as_json = send_as_json
        import requests
        self.requests = requests

    def on_epoch_end(self, global_step, epoch, logs=None):
        logs = logs or {}
        send = {}
        send['epoch'] = epoch
        for k, v in logs.items():
            send[k] = v.item() if isinstance(v, (np.ndarray, np.generic)) else k
        try:
            if self.send_as_json:
                self.requests.post(self.root + self.path, json=send, headers=self.headers)
            else:
                self.requests.post(self.root + self.path, {self.field: json.dumps(send)}, headers=self.headers)
        except self.requests.exceptions.RequestException:
            warnings.warn('Warning: could not reach RemoteMonitor root server at ' + str(self.root))


class Checkpoint(Callback):
    '''保存Checkpoint, 可以每个epoch或者每隔一定的steps保存

    :param model_path: str, 模型保存路径(含文件名)，可以使用{epoch}和{step}占位符
    :param optimizer_path: str, 优化器保存路径(含文件名)，可以使用{epoch}和{step}占位符，默认为None表示不保存
    :param scheduler_path: str, scheduler保存路径(含文件名)，可以使用{epoch}和{step}占位符，默认为None表示不保存
    :param steps_params_path: str, 模型训练进度保存路径(含文件名)，可以使用{epoch}和{step}占位符，默认为None表示不保存
    :param method: str, 按照轮次保存还是按照步数保存，默认为'epoch'表示每个epoch保存一次, 可选['epoch', 'step'] 
    :param step_interval: int, method设置为'step'时候指定每隔多少步数保存模型，默认为100表示每隔100步保存一次
    '''
    def __init__(self, model_path, optimizer_path=None, scheduler_path=None, steps_params_path=None, method='epoch', step_interval=100, verbose=0, **kwargs):
        super().__init__(**kwargs)
        assert method in {'step', 'epoch'}, 'Args `method` only support `step` or `epoch`'
        self.method = method
        self.model_path = model_path  # 保存路径，可设置{epoch}{step}{loss}等占位符
        self.optimizer_path = optimizer_path  # 是否保存优化器
        self.scheduler_path = scheduler_path  # 是否保存scheduler
        self.steps_params_path = steps_params_path  # 是否保存训练步数
        self.step_interval = step_interval  # method='step'时候生效
        self.verbose = verbose
    
    def on_epoch_end(self, global_step, epoch, logs=None):
        logs = logs or {}
        if self.method == 'epoch':
            self.process(epoch+1, logs)

    def on_batch_end(self, global_step, local_step, logs=None):
        logs = logs or {}
        if (self.method == 'step') and ((global_step+1) % self.step_interval == 0):
            self.process(global_step+1, logs)

    def process(self, suffix, logs):
        file_paths = []
        for filepath in [self.model_path, self.optimizer_path, self.scheduler_path, self.steps_params_path]:
            if filepath:
                filepath = filepath.format(epoch=suffix, **logs) if self.method == 'epoch' else filepath.format(step=suffix, **logs)
            file_paths.append(filepath)

        self.trainer.save_to_checkpoint(model_path=file_paths[0], optimizer_path=file_paths[1], scheduler_path=file_paths[2], 
                                        step_params_path=file_paths[3], verbose=self.verbose)


class Evaluator(Checkpoint):
    '''评估并保存最优Checkpoint, 也可以只评估

    :param monitor: str, 监控指标，需要在logs中，默认为'perf'
    :param verbose: int, 是否打印，默认为1表示打印
    :param mode: str, 控制监控指标monitor的大小方向，默认为'auto', 可选{'auto', 'min', 'max'}
    :param method: str, 控制是按照epoch还是step来计算，默认为'epoch', 可选{'step', 'epoch'}
    :param baseline: None/float, 基线, 默认为None 
    :param model_path: str, 模型保存路径(含文件名)，可以使用{epoch}和{step}占位符
    :param optimizer_path: str, 优化器保存路径(含文件名)，可以使用{epoch}和{step}占位符，默认为None表示不保存
    :param scheduler_path: str, scheduler保存路径(含文件名)，可以使用{epoch}和{step}占位符，默认为None表示不保存
    :param steps_params_path: str, 模型训练进度保存路径(含文件名)，可以使用{epoch}和{step}占位符，默认为None表示不保存
    :param step_interval: int, method设置为'step'时候指定每隔多少步数保存模型，默认为100表示每隔100步保存一次
    '''
    def __init__(self, monitor='perf', mode='max', verbose=1, model_path=None, optimizer_path=None, scheduler_path=None,
                 steps_params_path=None, method='epoch', step_interval=100, **kwargs):
        super().__init__(model_path, optimizer_path, scheduler_path, steps_params_path, method, step_interval, **kwargs)
        self.monitor = monitor
        assert mode in {'max', 'min'}, 'Compare performance only support `max/min` mode'
        self.mode = mode
        self.verbose = verbose
        self.best_perf = np.inf if mode == 'min' else -np.inf

    def process(self, suffix, logs):
        perf = self.evaluate()
        perf = perf if isinstance(perf, dict) else {'perf': perf}
        logs.update(perf)  # 评估的指标后续可能会用到
        
        # 满足条件
        if ((self.mode == 'max') and (perf[self.monitor] >= self.best_perf)) or ((self.mode == 'min') and (perf[self.monitor] <= self.best_perf)):
            self.best_perf = perf[self.monitor]
            # 保存ckpt
            super().process(suffix, logs)

        if self.verbose > 0:
            print_str = ', '.join([f'{k}: {v:.5f}' for k, v in perf.items()])
            print(print_str + f'. best_{self.monitor}: {self.best_perf:.5f}\n')
        
    # 定义评价函数
    def evaluate(self):
        # 需要返回一个字典，且self.monitor在字典key中
        # 如果返回的是一个数值型，则默认使用'perf'作为指标名
        raise NotImplemented


class Logger(Callback):
    '''默认logging
    对于valid/dev和test的日志需要在evaluate之后对log进行赋值，如log['dev_f1']=f1，并在Evaluator之后调用
    若每隔一定steps对验证集评估，则Logger的interval设置成和Evaluater一致或者约数，保证日志能记录到

    :param log_path: str, log文件的保存路径
    :param interval: int, 保存log的间隔
    :param mode: str, log保存的模式, 默认为'a'表示追加
    :param separator: str, 指标间分隔符
    :param verbosity: int, 可选[0,1,2]，指定log的level
    :param name: str, 默认为None
    '''
    def __init__(self, log_path, interval=10, mode='a', separator='\t', verbosity=1, name=None, **kwargs):
        super(Logger, self).__init__(**kwargs)
        import logging

        self.interval = interval
        self.sep = separator
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level_dict[verbosity])
        save_dir = os.path.dirname(log_path)
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(log_path, mode)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def on_train_begin(self, logs=None):
        self.logger.info('Start Training'.center(40, '='))

    def on_train_end(self, logs=None):
        self.logger.info('Finish Training'.center(40, '='))

    def on_epoch_begin(self, global_step, epoch, logs=None):
        self.logger.info(f'Epoch {epoch+1}'.center(40, '='))

    def on_epoch_end(self, global_step, epoch, logs=None):
        log_str = f'{self.sep}'.join([f'{k}={v:.5f}' for k, v in logs.items() if k not in {'size'}])
        self.logger.info(f'epoch={epoch+1}{self.sep}{log_str}')

    def on_batch_end(self, global_step, local_step, logs=None):
        if (global_step+1) % self.interval == 0:
            log_str = f'{self.sep}'.join([f'{k}={v:.5f}' for k, v in logs.items() if k not in {'size'}])
            self.logger.info(f'step={global_step+1}{self.sep}{log_str}')


class Tensorboard(Callback):
    '''默认Tensorboard
    对于valid/dev和test的Tensorboard需要在evaluate之后对log进行赋值，如log['dev/f1']=f1，并在Evaluator之后调用
    赋值需要分栏目的用'/'进行分隔
    若每隔一定steps对验证集评估，则Tensorboard的interval设置成和Evaluater一致或者约数，保证Tensorboard能记录到

    :param log_dir: str, tensorboard文件的保存路径
    :param method: str, 控制是按照epoch还是step来计算，默认为'epoch', 可选{'step', 'epoch'}
    :param interval: int, 保存tensorboard的间隔
    :param prefix: str, tensorboard分栏的前缀，默认为'train'
    :param on_epoch_end_scalar_epoch: bool, epoch结束后是横轴是按照epoch还是global_step来记录
    '''
    def __init__(self, log_dir, method='epoch', interval=10, prefix='train', on_epoch_end_scalar_epoch=True, **kwargs):
        super(Tensorboard, self).__init__(**kwargs)
        assert method in {'step', 'epoch'}, 'Args `method` only support `step` or `epoch`'
        self.method = method
        self.interval = interval
        self.prefix = prefix+'/' if len(prefix.strip()) > 0 else ''  # 控制默认的前缀，用于区分栏目
        self.on_epoch_end_scalar_epoch = on_epoch_end_scalar_epoch  # 控制on_epoch_end记录的是epoch还是global_step

        from tensorboardX import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))  # prepare summary writer

    def on_epoch_end(self, global_step, epoch, logs=None):
        if self.method == 'epoch':
            # 默认记录的是epoch
            log_step = epoch+1 if self.on_epoch_end_scalar_epoch else global_step+1
            self.process(log_step, logs)

    def on_batch_end(self, global_step, local_step, logs=None):
        # 默认记录的是global_step
        if (self.method == 'step') and ((global_step+1) % self.interval == 0):
            self.process(global_step+1, logs)

    def process(self, iteration, logs):
        logs = logs or {}
        for k, v in logs.items():
            if k in {'size'}:
                continue
            index = k if '/' in k else f"{self.prefix}{k}"
            self.writer.add_scalar(index, v, iteration)


class LambdaCallback(Callback):
    """lambda表达式
    """
    def __init__(self, on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, 
                 on_train_begin=None, on_train_end=None, on_dataloader_end=None, **kwargs):
        super(LambdaCallback, self).__init__(**kwargs)
        self.__dict__.update(kwargs)
        if on_epoch_begin is not None:
            self.on_epoch_begin = on_epoch_begin
        else:
            self.on_epoch_begin = lambda global_step, epoch, logs: None

        if on_epoch_end is not None:
            self.on_epoch_end = on_epoch_end
        else:
            self.on_epoch_end = lambda global_step, epoch, logs: None

        if on_batch_begin is not None:
            self.on_batch_begin = on_batch_begin
        else:
            self.on_batch_begin = lambda global_step, local_step, logs: None

        if on_batch_end is not None:
            self.on_batch_end = on_batch_end
        else:
            self.on_batch_end = lambda global_step, local_step, logs: None

        if on_train_begin is not None:
            self.on_train_begin = on_train_begin
        else:
            self.on_train_begin = lambda logs: None

        if on_train_end is not None:
            self.on_train_end = on_train_end
        else:
            self.on_train_end = lambda logs: None
        
        if on_dataloader_end is not None:
            self.on_dataloader_end = on_train_end
        else:
            self.on_dataloader_end = lambda logs: None


class Summary(Callback):
    '''调用torchinfo的summary
    '''
    def on_train_begin(self, logs=None):
        from torchinfo import summary
        print()
        summary(self.model, input_data=next(iter(self.trainer.train_dataloader))[0])
        print()


class EmailCallback(Callback):
    '''发送Email

    :param receivers: str/list, 收件人邮箱
    :param method: str, 控制是按照epoch还是step来发送邮件，默认为'epoch', 可选{'step', 'epoch'}
    :param interval: int, 发送邮件的的step间隔
    '''
    def __init__(self, receivers, subject='', method='epoch', interval=10, **kwargs):
        super(EmailCallback, self).__init__(**kwargs)
        self.method = method
        self.interval = interval
        self.receivers = receivers
        self.subject = subject

    def on_epoch_end(self, global_step, epoch, logs=None):
        if self.method == 'epoch':
            msg = json.dumps({k:f'{v:.5f}' for k,v in logs.items() if k!='size'}, indent=2, ensure_ascii=False)
            subject = f'[Epoch {epoch+1}] performance'
            if self.subject != '':
                subject = self.subject + ' | ' + subject
            send_email(self.receivers, subject=subject, msg=msg)

    def on_batch_end(self, global_step, local_step, logs=None):
        if (self.method == 'step') and ((global_step+1) % self.interval == 0):
            msg = json.dumps({k:f'{v:.5f}' for k,v in logs.items() if k!='size'}, indent=2, ensure_ascii=False)
            subject = f'[Step {global_step}] performance'
            if self.subject != '':
                subject = self.subject + ' | ' + subject
            send_email(self.receivers, subject=subject, msg=msg)

    def on_train_end(self, logs=None):
        msg = json.dumps({k:f'{v:.5f}' for k,v in logs.items() if k!='size'}, indent=2, ensure_ascii=False)
        subject = f'Finish training'
        if self.subject != '':
            subject = self.subject + ' | ' + subject
        send_email(self.receivers, subject=subject, msg=msg)


class WandbCallback(Callback):
    """从transformers迁移过来
    A :class:`~transformers.TrainerCallback` that sends the logs to `Weight and Biases <https://www.wandb.com/>`__.

    :param method: str, 控制是按照epoch还是step来log，默认为'epoch', 可选{'step', 'epoch'}
    :param interval: int, log的的step间隔
    :param watch (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
        Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
        logging or :obj:`"all"` to log gradients and parameters.
    :param project: str，wandb的project name, 默认为bert4torch
    :param 
    """
    def __init__(self, method='epoch', project='bert4torch', trial_name=None, run_name=None, watch='gradients', 
                 interval=100, save_code=False, config=None):
        try:
            import wandb
            self._wandb = wandb
        except:
            print("[WARNING] WandbCallback requires wandb to be installed. Run `pip install wandb`.")
            self._wandb = None

        self.method = method
        self._initialized = False
        # log outputs
        self.project = project
        if trial_name is None:
            self.trial_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if run_name is None:
            self.run_name = self.trial_name
        self.watch = watch
        self.interval = interval
        self.save_code = save_code
        self.config = config or {}
        self.run_id = None
        self.metrics = set()
        self.hidden_metrics = {'size'}

    def define_metric(self, logs=None):
        if getattr(self._wandb, "define_metric", None):
            for m in logs.keys():
                if m not in self.metrics:
                    self._wandb.define_metric(name=m, step_metric=self.method, hidden=True if m in self.hidden_metrics else False)
                    self.metrics.add(m)
    def adjust_logs(self, logs, **kwargs):
        logs_new = {**logs, **kwargs}
        return {k:v for k,v in logs_new.items() if k not in self.hidden_metrics}

    def on_epoch_end(self, global_step, epoch, logs=None):
        if self._wandb is None:
            return
        if self.method == 'epoch':
            self.define_metric(logs)
            self._wandb.log(self.adjust_logs(logs, epoch=epoch+1))

    def on_batch_end(self, global_step, local_step, logs=None):
        if self._wandb is None:
            return
        if (self.method == 'step') and ((global_step+1) % self.interval == 0):
            self.define_metric(logs)
            self._wandb.log(self.adjust_logs(logs, step=global_step))
        
    def on_train_begin(self, logs=None):
        if self._wandb is None:
            return

        self._initialized = True
        combined_dict = {**self.trainer.__dict__, **self.config}

        if hasattr(self.model, "config") and self.model.config is not None:
            combined_dict = {**self.model.config, **combined_dict}
        init_args = {}
        if self.trial_name is not None:
            run_name = self.trial_name
            init_args["group"] = self.run_name
        else:
            run_name = self.run_name

        if self._wandb.run is None:
            run = self._wandb.init(project=self.project, name=run_name, save_code=self.save_code, **init_args)
            self.run_id = run.id

        # add config parameters (run may have been created manually)
        self._wandb.config.update(combined_dict, allow_val_change=True)

        # keep track of model topology and gradients, unsupported on TPU
        if self.watch != "false":
            self._wandb.watch(self.model, log=self.watch, log_freq=max(100, self.interval))
 
    def on_train_end(self, logs=None):
        if self._wandb is None:
            return

        # transformer中的on_log
        self._wandb.finish()
        self._initialized = False


class AccelerateCallback(Callback):
    """Accelerate的Callback
    """
    def __init__(self, accelerator):
        self.accelerator = accelerator

    def get_module(self):
        '''返回nn.Module模块
        '''
        unwrap_model = self.accelerator.unwrap_model(self.model)
        return unwrap_model.module if hasattr(unwrap_model, 'module') else unwrap_model

    def on_train_begin(self, logs=None):
        self.trainer.loss_backward = False
        self.trainer.get_module = self.get_module

    def on_train_step_end(self, logs=None):
        self.accelerator.backward(self.trainer.loss)

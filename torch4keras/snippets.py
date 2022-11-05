import numpy as np
import torch
import time
import sys
import collections
import inspect
from datetime import datetime
import warnings
import os
import random
from collections import deque


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
    def __init__(self, callbacks=None, queue_length=10, master_rank=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length
        if master_rank:
            self.master_rank = master_rank

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def return_distributed(self):
        '''分布式训练中，控制只有master_rank才执行callbacks
        '''
        # 如果是分布式DDP训练，则仅masker_rank可以callback
        if hasattr(self, 'master_rank') and self.master_rank!=torch.distributed.get_rank():
            return

    def on_epoch_begin(self, global_step, epoch, logs=None):
        self.return_distributed()
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(global_step, epoch, logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

    def on_epoch_end(self, global_step, epoch, logs=None):
        self.return_distributed()
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(global_step, epoch, logs)

    def on_batch_begin(self, global_step, local_step, logs=None):
        self.return_distributed()
        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_begin(global_step, local_step, logs)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_begin)
        if (self._delta_t_batch > 0. and delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1):
            warnings.warn(f'Method on_batch_begin() is slow compared to the batch update {delta_t_median}. Check your callbacks.')
        self._t_enter_batch = time.time()

    def on_batch_end(self, global_step, local_step, logs=None):
        self.return_distributed()
        logs = logs or {}
        if not hasattr(self, '_t_enter_batch'):
            self._t_enter_batch = time.time()
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_end(global_step, local_step, logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if (self._delta_t_batch > 0. and (delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1)):
            warnings.warn(f'Method on_batch_end() is slow compared to the batch update {delta_t_median}. Check your callbacks.')

    def on_train_begin(self, logs=None):
        self.return_distributed()
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.return_distributed()
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_dataloader_end(self, logs=None):
        self.return_distributed()
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_dataloader_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    '''Callback基类
    '''
    def __init__(self):
        self.model = None
    def set_params(self, params):
        self.params = params
    def set_model(self, model):
        self.model = model
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


class BaseLogger(Callback):
    """计算metrics的均值, 默认是callbacks中第一项
    """

    def __init__(self, stateful_metrics=None):
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
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    if k in self.stateful_metrics:
                        logs[k] = self.totals[k]
                    else:
                        logs[k] = self.totals[k] / self.seen


class ProgbarLogger(Callback):
    """ keras进度条
    """

    def __init__(self, stateful_metrics=None):
        super(ProgbarLogger, self).__init__()
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

    def add_metrics(self, metrics, stateful_metrics=None, add_position=None):
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
            print('%s - Epoch: %d/%d' % (time_start, epoch + 1, self.epochs))
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


class History(Callback):
    """指标历史，默认是fit的返回项
    """

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, global_step, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class EarlyStopping(Callback):
    '''Stop training策略, 从keras中移植
       使用时候需要保证monitor在logs中，因此如果以valid指标评估需要在EarlyStopping前进行评估并设置logs[monitor]
    '''
    def __init__(self, monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
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
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, steps, epoch, logs=None):
        logs =  logs or {}
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                # 恢复最优权重
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch')
                    self.model.load_state_dict(self.best_weights, strict=True)
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'Epoch {self.stopped_epoch+1}: early stopping\n')

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn('Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)
        return monitor_value


class Checkpoint(Callback):
    '''保存Checkpoint, 可以每个epoch或者每隔一定的steps保存
    '''
    def __init__(self, filepath, optimizer=None, method='epoch', step_interval=100):
        super().__init__()
        assert method in {'step', 'epoch'}, 'Save checkpints only support step or epoch method'
        self.optimizer = optimizer
        self.filepath = filepath
        self.method = method
        self.step_interval = step_interval
    
    def on_epoch_end(self, global_step, epoch, logs=None):
        logs = logs or {}
        if self.method == 'epoch':
            self.process(epoch+1, logs)

    def on_batch_end(self, global_step, local_step, logs=None):
        logs = logs or {}
        if (self.method == 'step') and ((global_step+1) % self.step_interval == 0):
            self.process(global_step+1, logs)

    def process(self, suffix, logs):
        filepath = self.filepath.format(epoch=suffix, **logs) if self.method == 'epoch' else self.filepath.format(step=suffix, **logs)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(os.path.dirname(filepath), f'optimizer_{suffix}.pt'))


class Evaluator(Checkpoint):
    '''评估并保存最优Checkpoint, 也可以只评估
    '''
    def __init__(self, file_path, optimizer=None, method='epoch', step_interval=100, monitor='perf', mode='max', save_ckpt=True):
        super().__init__(file_path, optimizer, method, step_interval)
        self.monitor = monitor
        assert mode in {'max', 'min'}, 'Compare performance only support max/min mode'
        self.mode = mode
        self.best_perf = np.inf if mode == 'min' else -np.inf
        self.save_ckpt = save_ckpt

    def process(self, suffix, logs):
        perf = self.evaluate()
        perf = perf if isinstance(perf, dict) else {'perf': perf}
        logs.update(perf)  # 评估的指标后续可能会用到
        
        # 满足条件
        if ((self.mode == 'max') and (perf[self.monitor] > self.best_perf)) or ((self.mode == 'min') and (perf[self.monitor] < self.best_perf)):
            self.best_perf = perf[self.monitor]
            # 保存ckpt
            if self.save_ckpt:
                filepath = self.filepath.format(epoch=suffix, **logs) if self.method == 'epoch' else self.filepath.format(step=suffix, **logs)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                torch.save(self.model.state_dict(), filepath)
                if self.optimizer is not None:
                    torch.save(self.optimizer.state_dict(), os.path.join(self.filepath, f'best_optimizer.pt'))
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
    '''
    def __init__(self, filename, interval=10, mode='a', separator='\t', verbosity=1, name=None):
        super(Logger, self).__init__()
        import logging

        self.interval = interval
        self.sep = separator
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level_dict[verbosity])
        fh = logging.FileHandler(filename, mode)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def on_train_begin(self, logs=None):
        self.logger.info('Start Training'.center(40, '='))

    def on_train_end(self, logs=None):
        self.logger.info('Finish Training'.center(40, '='))

    def on_epoch_begin(self, global_step, epoch, logs=None):
        self.logger.info(f'Epoch {epoch}'.center(40, '='))

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
    '''
    def __init__(self, dirname, interval=10, prefix='train', on_epoch_end_scalar_epoch=True):
        super(Tensorboard, self).__init__()
        self.interval = interval
        self.prefix = prefix+'/' if len(prefix.strip()) > 0 else ''  # 控制默认的前缀，用于区分栏目
        self.on_epoch_end_scalar_epoch = on_epoch_end_scalar_epoch  # 控制on_epoch_end记录的是epoch还是global_step

        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(log_dir=str(dirname))  # prepare summary writer

    def on_epoch_end(self, global_step, epoch, logs=None):
        # 默认记录的是epoch
        for k, v in logs.items():
            if k not in {'size'}:
                continue
            index = k if '/' in k else f"{self.prefix}{k}"
            log_step = epoch+1 if self.on_epoch_end_scalar_epoch else global_step+1
            self.writer.add_scalar(index, v, log_step)

    def on_batch_end(self, global_step, local_step, logs=None):
        # 默认记录的是global_step
        if (global_step+1) % self.interval == 0:
            for k, v in logs.items():
                if k not in {'size'}:
                    continue
                index = k if '/' in k else f"{self.prefix}{k}"
                self.writer.add_scalar(index, v, global_step+1)


class LambdaCallback(Callback):
    """lambda表达式
    """

    def __init__(self, on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, 
                 on_train_begin=None, on_train_end=None, on_dataloader_end=None, **kwargs):
        super(LambdaCallback, self).__init__()
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


def metric_mapping(metric, func, y_pred, y_true):
    # 自定义metrics
    if inspect.isfunction(func):
        metric_res = func(y_pred, y_true)
        if inspect.isfunction(metric):
            # 如果直接传入回调函数（无key），要求回调函数返回Dict[String: Int/Float]类型
            assert isinstance(metric_res, dict), 'Custom metrics callbacks should return "Dict[String: Int/Float]" value'
        elif isinstance(metric, str):
            # 如果直接传入回调函数（有key），要求回调函数返回Int/Float类型
            assert isinstance(metric_res, (int, float)), 'Custom metrics callbacks should return "Int, Float" value'
        return metric_res
    elif metric == 'loss':
        pass
    # 自带metrics
    elif isinstance(metric, str):
        # 如果forward返回了list, tuple，则选取第一项
        y_pred_tmp = y_pred[0] if isinstance(y_pred, (list, tuple)) else y_pred
        y_true_tmp = y_true[0] if isinstance(y_true, (list, tuple)) else y_true

        # 根据shape做预处理
        if len(y_pred_tmp.shape) == len(y_true_tmp.shape) + 1:
            y_pred_tmp = torch.argmax(y_pred_tmp, dim=-1)
        elif len(y_pred_tmp.shape) == len(y_true_tmp.shape):
            pass
        else:
            raise ValueError(f'y_pred_tmp.shape={y_pred_tmp.shape} while y_true_tmp.shape={y_true_tmp.shape}')

        # 执行内置的metric
        if metric in {'accuracy', 'acc'}:
            return torch.sum(y_pred_tmp.eq(y_true_tmp)).item() / y_true_tmp.numel()
        elif metric in {'mae', 'MAE', 'mean_absolute_error'}:
            return torch.mean(torch.abs(y_pred_tmp - y_true_tmp)).item()
        elif metric in {'mse', 'MSE', 'mean_squared_error'}:
            return torch.mean(torch.square(y_pred_tmp - y_true_tmp)).item()
        elif metric in {'mape', 'MAPE', 'mean_absolute_percentage_error'}:
            diff = torch.abs((y_true_tmp - y_pred_tmp) / torch.clamp(torch.abs(y_true_tmp), 1e-7, None))
            return 100. * torch.mean(diff).item()
        elif metric in {'msle', 'MSLE', 'mean_squared_logarithmic_error'}:
            first_log = torch.log(torch.clamp(y_pred_tmp, 1e-7, None) + 1.)
            second_log = torch.log(torch.clamp(y_true_tmp, 1e-7, None) + 1.)
            return torch.mean(torch.square(first_log - second_log)).item()

    return None


def seed_everything(seed=None):
    '''固定seed
    '''
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed is None) or not (min_seed_value <= seed <= max_seed_value):
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    print(f"Global seed set to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
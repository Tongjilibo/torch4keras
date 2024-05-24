import sys
import collections
import time
import numpy as np
from datetime import datetime
import warnings
from collections import deque
import json
import copy
import os
from torch4keras.snippets import log_info, log_error, log_warn, send_email, log_warn_once
from torch4keras.snippets import set_precision, format_time, is_package_available
import math
from typing import Literal, Union, List
import shutil
import traceback
import torch


# 忽略nan的指标
IGNORE_NAN_VALUES = os.environ.get('IGNORE_NAN_VALUES', False)

# 不记录的metrics
SKIP_METRICS = os.environ.get('SKIP_METRICS', {})
SKIP_METRICS = eval(SKIP_METRICS) if isinstance(SKIP_METRICS, str) else SKIP_METRICS

# 不能平滑的指标
NO_SMOOTH_METRICS = os.environ.get('NO_SMOOTH_METRICS', {'lr'})
NO_SMOOTH_METRICS = eval(NO_SMOOTH_METRICS) if isinstance(NO_SMOOTH_METRICS, str) else NO_SMOOTH_METRICS

# 指标的精度
ROUND_PRECISION = int(os.environ.get('ROUND_PRECISION', 4))
_round_precision = eval(f"1e-{ROUND_PRECISION-1}")


def round(v, mode='f'):
    if abs(v) < _round_precision:
        mode = 'e'
    return f"%.{ROUND_PRECISION}{mode}" % v


class CallbackList(object):
    '''把原来在model.py中的callback_fun移植出来, 参考Keras的CallbackList重构
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

    def on_epoch_begin(self, global_step:int, epoch:int, logs:dict=None):
        # 如果是分布式DDP训练, 则仅masker_rank可以callback
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): continue
            callback.on_epoch_begin(global_step, epoch, logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): continue
            callback.on_epoch_end(global_step, epoch, logs)

    def on_batch_begin(self, global_step:int, local_step:int, logs:dict=None):
        if not self.run_callbacks: return
        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): continue
            callback.on_batch_begin(global_step, local_step, logs)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_begin)
        if (self._delta_t_batch > 0. and delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1):
            warnings.warn(f'Method on_batch_begin() is slow compared to the batch update {delta_t_median}. Check your callbacks.')
        self._t_enter_batch = time.time()

    def on_batch_end(self, global_step:int, local_step:int, logs:dict=None):
        if not self.run_callbacks: return
        logs = logs or {}
        if not hasattr(self, '_t_enter_batch'):
            self._t_enter_batch = time.time()
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): continue
            callback.on_batch_end(global_step, local_step, logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if (self._delta_t_batch > 0. and (delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1)):
            warnings.warn(f'Method on_batch_end() is slow compared to the batch update {delta_t_median}. Check your callbacks.')

    def on_train_begin(self, logs:dict=None):
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): continue
            callback.on_train_begin(logs)

    def on_train_end(self, logs:dict=None):
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): continue
            callback.on_train_end(logs)

    def on_dataloader_end(self, logs:dict=None):
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): continue
            callback.on_dataloader_end(logs)

    def on_train_step_end(self, logs:int=None):
        if not self.run_callbacks: return
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'run_callback') and (not callback.run_callback): continue
            callback.on_train_step_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    '''Callback基类'''
    def __init__(self, run_callback=True, **kwargs):
        self.trainer = None  # trainer
        self.model = None  # nn.Module模型, 或者包含Trainer的nn.Module
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
    def on_train_begin(self, logs:dict=None):
        pass
    def on_train_end(self, logs:dict=None):
        pass
    def on_epoch_begin(self, global_step:int, epoch:int, logs:dict=None):
        pass
    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        pass
    def on_batch_begin(self, global_step:int, local_step:int, logs:dict=None):
        pass
    def on_batch_end(self, global_step:int, local_step:int, logs:dict=None):
        pass
    def on_dataloader_end(self, logs:dict=None):
        pass
    def on_train_step_end(self, logs:int=None):
        pass


class SmoothMetric:
    '''指标平滑
    
    :param interval: int, 平滑时候使用的的step个数
    :param stateful_metrics: list, 以状态量记录指标的格式
    :param seen_so_far: int, 平滑起始点
    '''
    def __init__(self, interval:int=None, stateful_metrics:Union[str, set, tuple, list]=None, seen_so_far=0) -> None:
        self.interval = interval
        self.stateful_metrics = self._process_stateful_metrics(stateful_metrics)
        self._values = collections.OrderedDict()  # OrderedDict([('loss', [11.60657262802124, 5]), ('acc', [0.25, 5])])
        if self.interval is not None:
            self._oversize_values = deque(maxlen=self.interval)
        self.seen_so_far = seen_so_far
    
    @staticmethod
    def _process_stateful_metrics(stateful_metrics:Union[str, set, tuple, list]=None):
        '''对stateful_metrics进行处理'''
        if stateful_metrics is None:
            stateful_metrics_new = set()
        elif isinstance(stateful_metrics, str):
            stateful_metrics_new = {stateful_metrics}
        elif isinstance(stateful_metrics, (set, tuple, list)):
            stateful_metrics_new = set(stateful_metrics)
        else:
            raise ValueError('Args `stateful_metrics` only support `int/set/tuple/list` format')
        stateful_metrics_new.update(NO_SMOOTH_METRICS)
        return stateful_metrics_new

    def update(self, current:int, logs:dict):
        if self.interval is not None:
            if len(self._oversize_values) >= self.interval:
                self._oversize_value = self._oversize_values.popleft()
            self._oversize_values.append({k:v for k,v in logs.items()})
        
        for k, v in logs.items():
            if k in SKIP_METRICS:
                continue
            elif IGNORE_NAN_VALUES and np.isnan(v):
                log_error(f'Value `{k}` at {current} step is nan')
                continue
            elif (k in NO_SMOOTH_METRICS) or (k in self.stateful_metrics):
                self._values[k] = [v, 1]
            elif k not in self._values:
                self._values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]                    
            else:
                self._values[k][0] += v * (current - self.seen_so_far)
                self._values[k][1] += (current - self.seen_so_far)
                # 滑动平均, 把超出interval的的指标减去
                if hasattr(self, '_oversize_value') and k in self._oversize_value:
                    self._values[k][0] -= self._oversize_value[k]
                    self._values[k][1] -= 1  # 这里理论上应该也是current - self._seen_so_far
            
        self.seen_so_far = current
        return self._values

    def reset(self):
        self._values = collections.OrderedDict()

    def get_smooth_logs(self, logs:dict=None):
        smooth_logs = {k: v[0]/v[1] for k, v in self._values.items() if k not in SKIP_METRICS}
        
        if logs is not None:
            for k, v in logs.items() :
                if (k in SKIP_METRICS) or (k in smooth_logs):
                    continue
                smooth_logs[k] = v
        return smooth_logs

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)


class SmoothMetricsCallback(Callback):
    '''指标平滑的callback, 会inplace修改log, 影响后续的callback中log
    1) 适用情形: 希望Logger, Tensorboard, Wandb, EarlyStopping等callbacks中使用的累计平滑的指标
    2) 使用方法: 初始化后, 放在fit()中靠前的位置来对log进行修改
    3) step的平滑是全局来看的 (跨epoch不中断) , epoch平滑是对每个epoch分别累计计算

    :param interval: int, 平滑时候使用的的step个数
    :param stateful_metrics: list, 以状态量记录指标的格式
    '''
    def __init__(self, interval:int=100, stateful_metrics:Union[str, set, tuple, list]=None, seen_so_far:int=0, verbose:int=0, **kwargs):
        super(SmoothMetricsCallback, self).__init__(**kwargs)
        self.interval = interval
        self.stateful_metrics = stateful_metrics
        self.seen_so_far = seen_so_far
        self.smooth_metric_step = SmoothMetric(interval=self.interval, stateful_metrics=self.stateful_metrics, seen_so_far=self.seen_so_far)
        if verbose != 0:
            log_info(f'SmoothMetricCallback calculate {interval} steps average metrics')

    def on_epoch_begin(self, global_step:int, epoch:int, logs:dict=None):
        self.smooth_metric_epoch = SmoothMetric(stateful_metrics=self.stateful_metrics)

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        self.smooth_metric_epoch.get_smooth_logs(logs)
        smooth_logs = self.smooth_metric_epoch.get_smooth_logs()
        logs.update(smooth_logs)

    def on_batch_end(self, global_step:int, local_step:int, logs:dict=None):
        self.smooth_metric_step.update(global_step+1, logs)
        self.smooth_metric_epoch.update(local_step+1, logs)
        smooth_logs = self.smooth_metric_step.get_smooth_logs()
        logs.update(smooth_logs)


class Progbar(object):
    '''进度条, 直接从keras引入'''
    def __init__(self, target:int, width:int=30, verbose:int=1, time_interval:float=0.05):
        '''
        :param target: 进度条的step数量
        :param width: 进度条的宽度
        :param verbose: 是否展示进度条
        :param time_interval: 更新进度条的最短时间间隔
        '''
        self.target = target
        self.width = width
        self.verbose = verbose
        self.time_interval = time_interval
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()) or 'ipykernel' in sys.modules)
        self._total_width = 0
        self._start = time.time()
        self._last_update = 0

    def update(self, current:int, values:dict=None):
        '''Updates the progress bar.'''
        now = time.time()
        if self.verbose == 1:
            if (now - self._last_update < self.time_interval and self.target is not None and current < self.target):
                # 训练每个step太快了, 则不更新进度条, 累计达到一定的时间间隔再更新进度条
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
                info = ' - ETA: %s' % format_time(now - self._start) + '<' + format_time(eta)
            else:
                info = ' - %s' % format_time(now - self._start)
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k, v in values.items():
                info += ' - %s:' % k
                info += f' %.{ROUND_PRECISION}f' % v if abs(v) > 1e-3 else f' %.{ROUND_PRECISION}e' % v
            info += ' '  # 最后加个空格, 防止中途有别的打印
            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k, v in values.items():
                    info += ' - %s:' % k
                    info += f' %.{ROUND_PRECISION}f' % v if abs(v) > 1e-3 else f' %.{ROUND_PRECISION}e' % v
                info += '\n'
                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now


class KerasProgbar(Callback):
    ''' keras进度条 '''
    def __init__(self, width:int=30, **kwargs):
        super(KerasProgbar, self).__init__(**kwargs)
        self.width = width

    def add_metrics(self, metrics:Union[str, list], add_position:bool=None):
        '''在指定位置插入metrics指标
        '''
        if add_position is None:
            add_position = len(self.params['metrics'])
        metrics = [metrics] if isinstance(metrics, str) else metrics

        add_metrics = []
        for metric in metrics:
            if metric not in self.params['metrics']:
                add_metrics.append(metric)
        self.params['metrics'] = self.params['metrics'][:add_position] + add_metrics + self.params['metrics'][add_position:]

    def on_train_begin(self, logs:dict=None):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']
        if self.verbose:
            time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('%s - Start Training' % (time_start))

    def on_train_end(self, logs:dict=None):
        if self.verbose:
            time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('\n%s - Finish Training' % (time_start))

    def on_epoch_begin(self, global_step=None, epoch=None, logs=None):
        if self.verbose:
            time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('\n%s - Start  Epoch: %d/%d' % (time_start, epoch+1, self.epochs))
            self.target = self.params['steps']
            self.progbar = Progbar(target=self.target, width=self.width, verbose=self.verbose)

    def on_epoch_end(self, global_step: int, epoch: int, logs: dict = None):
        # 打印该Epoch的参数
        logs = logs or {}
        log_values = {k:logs[k] for k in self.params['metrics'] if k in logs}
        if self.verbose:
            time_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print_str = '%s - Finish Epoch: %d/%d' % (time_end, epoch+1, self.epochs)
            for k, v in log_values.items():
                v = f' %.{ROUND_PRECISION}f' % v if abs(v) > 1e-3 else f' %.{ROUND_PRECISION}e' % v
                print_str += f' - {k}:{v}'
            print(print_str)

    def on_batch_end(self, global_step:int=None, local_step:int=None, logs:dict=None):
        logs = logs or {}  # 这里的logs是当前batch的指标
        log_values = {k:logs[k] for k in self.params['metrics'] if k in logs}  # 限定仅打印metrics中
        if self.verbose:  # 这里打印的是过去interval区间内的平均指标
            self.progbar.update(local_step+1, log_values)
    
        
class TqdmProgbar(KerasProgbar):
    ''' Tqdm进度条 '''
    def __init__(self, width:int=None, **kwargs):
        super().__init__(width, **kwargs)

    def on_epoch_begin(self, global_step=None, epoch=None, logs=None):
        if self.verbose:
            from tqdm import tqdm
            time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('\n%s - Start  Epoch: %d/%d' % (time_start, epoch+1, self.epochs))
            self.target = self.params['steps']
            self.progbar = tqdm(total=self.params['steps'], desc='Training', dynamic_ncols=False, file=sys.stdout, smoothing=0, ncols=self.width)

    def on_batch_end(self, global_step:int=None, local_step:int=None, logs:dict=None):
        logs_new = self.smooth_values(local_step+1, logs)
        log_values = [(k, logs_new[k]) for k in self.params['metrics'] if k in logs_new]

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose:
            self.progbar.n = local_step+1
            self.progbar.refresh()
            self.progbar.set_postfix(log_values)

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        self.progbar.close()
        KerasProgbar.on_epoch_end(self, global_step, epoch, logs)
    
    def smooth_values(self, current:int, values:dict=None):
        '''从Progbar迁移过来'''
        logs = collections.OrderedDict()
        for k, v in values.items():
            logs[k] = f' %.{ROUND_PRECISION}f' % v if abs(v) > 1e-3 else f' %.{ROUND_PRECISION}e' % v
        return logs


class ProgressBar2Progbar(TqdmProgbar):
    ''' progressbar2进度条 '''
    def on_epoch_begin(self, global_step:int=None, epoch:int=None, logs:dict=None):
        if self.verbose:
            import progressbar
            time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('\n%s - Start  Epoch: %d/%d' % (time_start, epoch+1, self.epochs))
            self.target = self.params['steps']
            widgets = [progressbar.SimpleProgress(format='%(value_s)s/%(max_value_s)s'), progressbar.Bar(marker='=', left='[', right=']'), ' ', 
                       progressbar.AdaptiveETA(format='ETA: %(eta)s', format_finished='Time: %(elapsed)s'), ' - ']
            for i, param in enumerate(self.params['metrics']):
                widgets.append(progressbar.Variable(param, precision=7))
                if i < len(self.params['metrics'])-1:
                    widgets.append(' - ')
            self.progbar = progressbar.bar.ProgressBar(min_value=0, max_value=self.params['steps'], widgets=widgets, 
                                                       redirect_stdout=True, redirect_stderr=True)

    def on_batch_end(self, global_step:int=None, local_step:int=None, logs:dict=None):
        logs_new = self.smooth_values(local_step+1, logs or {})
        logs_new = {k:logs_new[k].strip() for k in self.params['metrics'] if k in logs_new}

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose:
            self.progbar.update(local_step+1, **logs_new)

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        self.progbar.finish()
        KerasProgbar.on_epoch_end(self, global_step, epoch, logs)


class TerminateOnNaN(Callback):
    '''Loss出现NAN停止训练'''
    def on_batch_end(self, global_step:int, local_step:int, logs:dict=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                log_error('Step %d: Invalid loss, terminating training' % global_step)
                self.trainer.stop_training = True


class History(Callback):
    '''指标历史, 默认是fit的返回项, 这里仅记录epoch_end的指标'''
    def on_train_begin(self, logs:dict=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        logs = logs or {}
        self.epoch.append(epoch+1)  # 这里和keras相比+1了
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def plot(self, ncols:int=4, subplot_size:tuple=(4,3), save_path:str=None, show:bool=True, plot_text:bool=True):
        import matplotlib.pyplot as plt
        import traceback

        nrows = math.ceil(len(self.history) / ncols)
        if nrows <= 1:
            ncols = len(self.history)
        figsize = (subplot_size[0]*ncols, subplot_size[1]*nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=False, squeeze=False, figsize=figsize)
        for i, fea in enumerate(self.history):
            try:
                ax = axes[int(i/ncols), i%ncols]
                ax.plot(self.epoch, self.history[fea], 'bo--', clip_on=False)
                ax.set_title(fea)
                ax.set_xlabel('epoch')
                ax.set_ylabel(fea)
                if plot_text:
                    for x, y in zip(self.epoch, self.history[fea]):
                        ax.text(x, y, f'{set_precision(y)}', ha='center', va='bottom', color='red')
            except:
                log_error(f'Plot `{fea}` error: ' + traceback.format_exc())
        plt.tight_layout()

        if save_path:
            # 保存文件
            save_dir = '/'.join(save_path.split('/')[:-1])
            save_file = save_path.split('/')[-1].split('.')[0] + '.jpg'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, save_file), dpi=100, bbox_inches='tight')
        if show:
            plt.show()
            plt.close()


class EarlyStopping(Callback):
    '''Stop training策略, 从keras中移植
       使用时候需要保证monitor在logs中, 因此如果以valid指标评估需要在EarlyStopping前进行评估并设置logs[monitor]

       :param monitor: str, 监控指标, 需要在logs中, 默认为'loss', 
       :param min_delta: float, 最小变动, 默认为0 
       :param patience: int, 最长等候的次数, 默认为0
       :param verbose: int, 是否打印, 默认为0表示不打印
       :param min_max: str, 控制监控指标monitor的大小方向, 默认为'auto', 可选{'auto', 'min', 'max'}
       :param epoch_or_step: str, 控制是按照epoch还是step来计算, 默认为'epoch', 可选{'step', 'epoch'}
       :param baseline: None/float, 基线, 默认为None 
       :param restore_best_weights: bool, stopping时候是否恢复最优的权重, 默认为False

       Examples:
       ```python
       >>> # 如果连续3个epoch, test_acc还没有继续增长则停止训练
       >>> early_stop = EarlyStopping(monitor='test_acc', verbose=1, epoch_or_step='epoch', patience=3, min_max='max')

       >>> # 如果连续100个steps, loss还未继续下降则停止训练
       >>> early_stop = EarlyStopping(monitor='loss', verbose=1, epoch_or_step='step', patience=100, min_max='min')
       ```

    '''
    def __init__(self, monitor:str='perf', min_delta:float=0, patience:int=0, verbose:int=0, min_max:Literal['auto', 'min', 'max']='auto', 
                 epoch_or_step:Literal['epoch', 'step']='epoch', baseline:float=None, restore_best_weights:bool=False, **kwargs):
        super(EarlyStopping, self).__init__(**kwargs)
        assert epoch_or_step in {'step', 'epoch'}, 'Args `epoch_or_step` only support `step` or `epoch`'
        self.epoch_or_step = epoch_or_step  # 默认的epoch和原版一样
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience  # epoch_or_step=step时候表示最多wait的训练步数
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_iteration = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if min_max not in {'auto', 'min', 'max'}:
            warnings.warn('EarlyStopping `min_max` %s is unknown, fallback to auto.' % min_max, RuntimeWarning)
            min_max = 'auto'

        if min_max == 'min':
            self.monitor_op = np.less
        elif min_max == 'max':
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.greater if 'acc' in self.monitor else np.less
        self.min_delta = self.min_delta if self.monitor_op == np.greater else -self.min_delta

    def on_train_begin(self, logs:dict=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_iteration = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_batch_end(self, global_step:int, local_step:int, logs:dict=None):
        if self.epoch_or_step == 'step':
            return self.process(global_step, logs)

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        if self.epoch_or_step == 'epoch':
            self.process(epoch, logs)

    def process(self, iteration:int, logs:dict=None):
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

    def on_train_end(self, logs:dict=None):
        if self.stopped_iteration > 0 and self.verbose > 0:
            print(f'Iteration {self.stopped_iteration+1}: early stopping\n')

    def get_monitor_value(self, logs:dict):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn('Early stopping conditioned on metric `%s` which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)
        return monitor_value


class ReduceLROnPlateau(Callback):
    '''当monitor指标不下降时候, 降低学习率

    :param monitor: str, 监控指标, 需要在logs中, 默认为'loss'
    :param factor: float, 权重衰减系数, 取值范围(0, 1), 默认为0.1
    :param patience: int, 最长等候的次数, 默认为0
    :param epoch_or_step: str, 控制是按照epoch还是step来计算, 默认为'epoch', 可选{'step', 'epoch'}
    :param verbose: int, 是否打印, 默认为0表示不打印
    :param min_max: str, 控制监控指标monitor的大小方向, 默认为'auto', 可选{'auto', 'min', 'max'}
    :param min_delta: float, 最小变动, 默认为0 
    :param cooldown: float
    :param min_lr: float, 最小学习率
    '''
    def __init__(self, monitor:str='loss', factor:float=0.1, patience:int=10, epoch_or_step:Literal['epoch', 'step']='epoch', 
                 verbose:int=0, min_max:Literal['auto', 'min', 'max']='auto', min_delta:float=1e-4, cooldown:float=0, 
                 min_lr:float=0, **kwargs):
        super(ReduceLROnPlateau, self).__init__(**kwargs)
        assert epoch_or_step in {'step', 'epoch'}, 'Args `epoch_or_step` only support `step` or `epoch`'
        self.epoch_or_step = epoch_or_step  # 默认的epoch和原版一样
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('`epsilon` argument is deprecated and will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience  # epoch_or_step=step时候表示最多wait的训练步数
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.min_max = min_max
        self.monitor_op = None
        self._reset()

    def _reset(self):
        '''Resets wait counter and cooldown counter.
        '''
        if self.min_max not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing min_max %s is unknown, fallback to auto.' % (self.min_max), RuntimeWarning)
            self.min_max = 'auto'
        if (self.min_max == 'min' or
           (self.min_max == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs:dict=None):
        self._reset()

    def on_batch_end(self, global_step:int, local_step:int, logs:dict=None):
        if self.epoch_or_step == 'step':
            return self.process(global_step, logs)

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        if self.epoch_or_step == 'epoch':
            self.process(epoch, logs)

    def process(self, iteration:int, logs:dict=None):
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
    '''Callback used to stream events to a server.

    :param root: str, url+port
    :param path: str, router
    :param field: str, 字段名
    :param headers: header
    :param send_as_json: bool, 是否以json形式请求, 默认为False
    '''
    def __init__(self, root:str='http://localhost:9000', path:str='/publish/epoch/end/', field:str='data',
                 headers=None, send_as_json:bool=False, **kwargs):
        super(RemoteMonitor, self).__init__(**kwargs)
        self.root = root
        self.path = path
        self.field = field
        self.headers = headers
        self.send_as_json = send_as_json
        import requests
        self.requests = requests

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
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
    '''保存Checkpoint, 可以每个epoch或者每隔一定的steps保存, 也可以保存最近/最优的ckpt(weights/optimizer/scheduler/steps_params)

    :param save_dir: str, 保存的文件夹, 只定义即可按照默认的文件名保存
    :param epoch_or_step: str, 按照轮次保存还是按照步数保存, 默认为'epoch'表示每个epoch保存一次, 可选['epoch', 'step'] 
    :param interval: int, epoch_or_step设置为'step'时候指定每隔多少步数保存模型, 默认为100表示每隔100步保存一次
    :param max_save_count: int, 最大保存的权重的个数
    :param max_save_count_path: str, 历史保存的ckpt的路径和指标记录, 如果是断点续训可以开启, 减少保存的冗余ckpt数量
    :param monitor: str, 监控的指标, 当max_save_count_path不为None且monitor为None表示最近的ckpt, monitor不为None表示最优的ckpt
    :param min_max: str, 指示指标的优化方向
    :param save_on_train_end: bool, 训练结束后是否保存ckpt

    > 可选参数, 用于具体指定每个文件保存, 一般推荐直接指定save_dir即可
    :param model_path: str, 模型保存路径(含文件名), 可以使用{epoch}和{step}占位符
    :param optimizer_path: str, 优化器保存路径(含文件名), 可以使用{epoch}和{step}占位符, 默认为None表示不保存
    :param scheduler_path: str, scheduler保存路径(含文件名), 可以使用{epoch}和{step}占位符, 默认为None表示不保存
    :param steps_params_path: str, 模型训练进度保存路径(含文件名), 可以使用{epoch}和{step}占位符, 默认为None表示不保存
    
    Examples:
    ```python
    >>> # 每个epoch结束时保存
    >>> ckpt = Checkpoint(save_dir='./ckpt/{epoch}', epoch_or_step='epoch')
    
    >>> # 每隔1000个steps保存
    >>> ckpt = Checkpoint(save_dir='./ckpt/{step}', epoch_or_step='step', interval=1000)
    
    >>> # 保存loss最小的3个权重
    >>> ckpt = Checkpoint(save_dir='./ckpt/{epoch}', epoch_or_step='epoch', monitor='loss', min_max='min', max_save_count=3)
    
    >>> # 保存最近的3个权重
    >>> ckpt = Checkpoint(save_dir='./ckpt/{epoch}', epoch_or_step='epoch', monitor=None, max_save_count=3)
    
    >>> # 保存文件夹名称中含指标名, 方便查看
    >>> ckpt = Checkpoint(save_dir='./ckpt/{epoch}_{loss}')
    ```
    '''
    def __init__(self, save_dir:str=None, epoch_or_step:Literal['epoch', 'step']='epoch', interval:int=100, monitor:str=None, 
                 min_max:Literal['max', 'min']='min', verbose:int=0, max_save_count:int=None, max_save_count_path:str=None, 
                 save_on_train_end:bool=False, **kwargs):
        super().__init__(**kwargs)
        assert epoch_or_step in {'step', 'epoch'}, 'Args `epoch_or_step` only support `step` or `epoch`'
        self.epoch_or_step = epoch_or_step
        self.save_dir = save_dir  # 保存文件夹（推荐）
        self.save_paths = {}  # 具体指定各自保存的路径
        for i in ['model_path', 'optimizer_path', 'scheduler_path', 'steps_params_path']:
            if i in kwargs:
                self.save_paths[i] = kwargs.pop(i)
        self.interval = interval  # epoch_or_step='step'时候生效
        self.verbose = verbose
        self.monitor = monitor
        self.min_max = min_max
        self.max_save_count = max_save_count  # 最大保存的权重的个数
        self.max_save_count_path = max_save_count_path
        if self.max_save_count is not None:
            # 记录保存的文件路径历史，用于保存最近/最优的ckpt
            # 保存成文件目的：如果训练中间断掉，后续重启后还能保存断之前的状态，否则会重新保存最优/最近的几个文件
            if self.max_save_count_path is not None and os.path.exists(self.max_save_count_path):
                tmp = torch.load(self.max_save_count_path)
                self.save_history = tmp['save_history']
                self.save_history_monitor = tmp['save_history_monitor']
                log_info(f'Resume save_history from {self.max_save_count_path}')
            else:
                self.save_history = []
                self.save_history_monitor = []
        self.save_on_train_end = save_on_train_end
        self.kwargs = kwargs
    
    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        logs = logs or {}
        if self.epoch_or_step == 'epoch':
            self.process(epoch+1, logs)

    def on_batch_end(self, global_step:int, local_step:int, logs:dict=None):
        logs = logs or {}
        if (self.epoch_or_step == 'step') and ((global_step+1) % self.interval == 0):
            self.process(global_step+1, logs)

    @staticmethod
    def replace_placeholder(filepath:Union[str, list, dict], epoch_suffix, step_suffix, **logs):
        if filepath is None:
            pass
        elif isinstance(filepath, str):
            filepath = filepath.format(epoch=epoch_suffix, step=step_suffix, **logs)
        elif isinstance(filepath, list):
            filepath = [i.format(epoch=epoch_suffix, step=step_suffix, **logs) for i in filepath]
        elif isinstance(filepath, dict):
            filepath = {k:v.format(epoch=epoch_suffix, step=step_suffix, **logs) for k, v in filepath.items()}
        return filepath
    
    def on_train_end(self, logs: dict = None):
        if self.save_on_train_end:
            self.trainer.save_to_checkpoint(self.replace_placeholder(self.save_dir, epoch_suffix='final', step_suffix='final', **logs), 
                                            verbose=self.verbose, **self.kwargs,
                                            **self.replace_placeholder(self.save_paths, epoch_suffix='final', step_suffix='final', **logs))

    def process(self, suffix:int, logs:dict):
        save_dir = self.replace_placeholder(self.save_dir, epoch_suffix=suffix, step_suffix=suffix, **logs)
        save_paths = self.replace_placeholder(self.save_paths, epoch_suffix=suffix, step_suffix=suffix, **logs)
        self.trainer.save_to_checkpoint(save_dir, verbose=self.verbose, **self.kwargs, **save_paths)

        # 删除超出size的文件
        if self.max_save_count is not None:
            file_paths = tuple([save_dir] + list(save_paths.values()))
            if file_paths not in self.save_history:
                self.save_history.append(file_paths)
                if self.monitor is not None:
                    if self.monitor in logs:
                        self.save_history_monitor.append(logs.get(self.monitor))
                    else:
                        log_warn_once(f'Args `monitor`={self.monitor} not in logs')
            self.remove_oversize_checkpoints()
            # 保存历史ckpt的记录
            if self.max_save_count_path is not None:
                os.makedirs(os.path.dirname(self.max_save_count_path), exist_ok=True)
                torch.save({'save_history': self.save_history, 'save_history_monitor': self.save_history_monitor}, self.max_save_count_path)
    
    def remove_oversize_checkpoints(self):
        '''删除超出size的文件'''
        if len(self.save_history) > self.max_save_count:
            # 删除oldest的ckpt
            split_index = len(self.save_history) - self.max_save_count
            if len(self.save_history_monitor) == 0:
                drop_list = list(range(0, split_index))
            # 删除指标最差的ckpt
            else:
                sorted_idx = np.argsort(self.save_history_monitor)  # 从小到大
                if self.min_max == 'max':
                    drop_list = sorted_idx[:split_index]  # 删除最小的
                else:
                    drop_list = sorted_idx[::-1][:split_index]  # 删除指标最大的

            for item in [self.save_history[d_i] for d_i in drop_list]:
                for i in item:
                    if i is None:
                        continue
                    try:
                        if os.path.isdir(i):
                            shutil.rmtree(i)
                        elif os.path.isfile(i):
                            os.remove(i)
                    except:
                        log_warn(f'Remove {i} error')
            self.save_history = [v for i,v in enumerate(self.save_history) if i not in drop_list]
            self.save_history_monitor = [v for i,v in enumerate(self.save_history_monitor) if i not in drop_list]


class Evaluator(Checkpoint):
    '''评估: 可以每个epoch或者每隔一定的steps进行评估, 并可保存最优Checkpoint, 一般需要继承使用

    :param monitor: str, 监控指标, 需要在logs中, 默认为'perf'
    :param verbose: int, 是否打印, 默认为2表示打印
    :param min_max: str, 控制监控指标monitor的大小方向, 默认为'auto', 可选{'auto', 'min', 'max'}
    :param epoch_or_step: str, 控制是按照epoch还是step来计算, 默认为'epoch', 可选{'step', 'epoch'}
    :param baseline: None/float, 基线, 默认为None 
    :param save_dir: str, 保存的文件夹, 只定义即可按照默认的文件名保存
    :param interval: int, epoch_or_step设置为'step'时候指定每隔多少步数保存模型, 默认为100表示每隔100步保存一次
    
    > 可选参数
    :param model_path: str, 模型保存路径(含文件名), 可以使用{epoch}和{step}占位符
    :param optimizer_path: str, 优化器保存路径(含文件名), 可以使用{epoch}和{step}占位符, 默认为None表示不保存
    :param scheduler_path: str, scheduler保存路径(含文件名), 可以使用{epoch}和{step}占位符, 默认为None表示不保存
    :param steps_params_path: str, 模型训练进度保存路径(含文件名), 可以使用{epoch}和{step}占位符, 默认为None表示不保存
    
    Examples:
    ```python
    >>> class MyEvaluator(Evaluator):
    >>> def evaluate(self):
    >>>     test_acc = random.random()  # 计算逻辑示例
    >>>     return {'test_acc': test_acc}
    
    >>> # 每个epoch进行评估, 并保存test_acc最大的ckpt权重
    >>> evaluator = MyEvaluator(monitor='test_acc', save_dir='./ckpt/best/', min_max='max', epoch_or_step='epoch')
    
    >>> # 每隔1000个steps进行评估, 并保存test_acc最大的ckpt权重
    >>> evaluator = MyEvaluator(monitor='test_acc', save_dir='./ckpt/best/', min_max='max', epoch_or_step='step', interval=1000)
    ```
    '''
    def __init__(self, monitor:str='perf', min_max:Literal['max', 'min']='max', verbose:int=2, 
                 save_dir:str=None, epoch_or_step:Literal['epoch', 'step']='epoch', interval:int=100, **kwargs):
        super().__init__(save_dir, epoch_or_step, interval, **kwargs)
        self.monitor = monitor
        assert min_max in {'max', 'min'}, 'Compare performance only support `max/min`'
        self.min_max = min_max
        self.verbose = verbose
        self.best_perf = np.inf if min_max == 'min' else -np.inf

    def process(self, suffix:int, logs:dict):
        perf = self.evaluate()
        # 如果evaluate返回的是字典则使用字典, 如果返回的是数值则套上{'perf': perf}
        if perf is None:
            perf = logs.copy()
        else:
            perf = perf if isinstance(perf, dict) else {'perf': perf}
            logs.update(perf)  # 评估的指标后续可能会用到
        
        # 不存在
        if perf.get(self.monitor) is None:
            warnings.warn('Evaluator callback conditioned on metric `%s` which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)
            return

        # 满足条件
        if ((self.min_max == 'max') and (perf[self.monitor] >= self.best_perf)) or ((self.min_max == 'min') and (perf[self.monitor] <= self.best_perf)):
            self.best_perf = perf[self.monitor]
            # 保存ckpt
            super().process(suffix, logs)

        if self.verbose != 0:
            print_str = ', '.join([f'{k}: {round(v)}' for k, v in perf.items()])
            print(print_str + f'. best_{self.monitor}: {round(self.best_perf)}')
        
    # 定义评价函数
    def evaluate(self) -> Union[int, float, dict]:
        # 需要返回一个字典, 且self.monitor在字典key中
        # 如果返回的是一个数值型, 则默认使用'perf'作为指标名
        return None


class Logger(Callback):
    '''默认logging
    对于valid/dev和test的日志需要在evaluate之后对log进行赋值, 如log['dev_f1']=f1, 并在Evaluator之后调用
    若每隔一定steps对验证集评估, 则Logger的interval设置成和Evaluater一致或者约数, 保证日志能记录到

    :param log_path: str, log文件的保存路径
    :param interval: int, 保存log的间隔
    :param mode: str, log保存的模式, 默认为'a'表示追加
    :param separator: str, 指标间分隔符
    :param level: str, DEBUG/INFO/WARNING/ERROR/CRITICAL, 指定log的level

    Examples:
    ```python
    >>> # 每隔100个step记录下当前指标
    >>> logger = Logger('./ckpt/log.log', interval=100)
    ```
    '''
    def __init__(self, log_path:str, interval:int=100, mode:Literal['a', 'w']='a', separator:str='\t', 
                 level:Literal['DEBUG','INFO','WARNING','ERROR','CRITICAL']='DEBUG', name:str='root', **kwargs):
        super(Logger, self).__init__(**kwargs)
        self.log_path = log_path
        self.interval = interval
        self.mode = mode
        self.sep = separator
        self.name = name
        self.level = level

    def on_train_begin(self, logs:dict=None):
        import logging
        level_dict = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 
                      'ERROR': logging.ERROR, 'CRITICAL':logging.CRITICAL}
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(level_dict[self.level])
        save_dir = os.path.dirname(self.log_path)
        if save_dir != '':
            os.makedirs(save_dir, exist_ok=True)
        
        fh = logging.FileHandler(self.log_path, self.mode)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info('Start Training'.center(40, '='))

    def on_train_end(self, logs:dict=None):
        self.logger.info('Finish Training'.center(40, '=') + '\n')

    def on_epoch_begin(self, global_step:int, epoch:int, logs:dict=None):
        self.logger.info(f'Epoch {epoch+1}'.center(40, '='))

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        log_str = f'{self.sep}'.join([f'{k}={round(v)}' for k, v in logs.items() if k not in SKIP_METRICS])
        self.logger.info(f'epoch={epoch+1}{self.sep}{log_str}')

    def on_batch_end(self, global_step:int, local_step:int, logs:dict=None):
        if (global_step+1) % self.interval == 0:
            log_str = f'{self.sep}'.join([f'{k}={round(v)}' for k, v in logs.items() if k not in SKIP_METRICS])
            self.logger.info(f'step={global_step+1}{self.sep}{log_str}')


class Tensorboard(Callback):
    '''默认Tensorboard
    对于valid/dev和test的Tensorboard需要在evaluate之后对log进行赋值, 如log['dev/f1']=f1, 并在Evaluator之后调用
    赋值需要分栏目的用'/'进行分隔
    若每隔一定steps对验证集评估, 则Tensorboard的interval设置成和Evaluater一致或者约数, 保证Tensorboard能记录到

    :param log_dir: str, tensorboard文件的保存路径
    :param interval: int, 保存tensorboard的间隔
    :param prefix: str, tensorboard分栏的前缀, 默认为'train'

    Examples:
    ```python
    >>> ts_board = Tensorboard('./ckpt/tensorboard')
    ```
    '''
    def __init__(self, log_dir:str, interval:int=100, prefix:str='Train', **kwargs):
        super(Tensorboard, self).__init__(**kwargs)
        self.log_dir = log_dir
        self.interval = interval
        self.prefix_step = prefix+'/' if len(prefix.strip()) > 0 else ''  # 控制默认的前缀, 用于区分栏目
        self.prefix_epoch = prefix+'_epoch/' if len(prefix.strip()) > 0 else 'Epoch/'  # 控制默认的前缀, 用于区分栏目

    def on_train_begin(self, logs:dict=None):
        try:
            from tensorboardX import SummaryWriter
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))  # prepare summary writer
        except ImportError:
            log_warn('Tensorboard callback requires tensorboardX to be installed. Run `pip install tensorboardX`; Skip this callback instead.')
            self.run_callback = False
        except:
            log_warn_once(traceback.format_exc())
            self.run_callback = False

    def on_train_end(self, logs: dict = None):
        try:
            self.writer.close()
        except:
            log_warn_once(traceback.format_exc())
    
    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        if hasattr(self.trainer, 'epochs') and self.trainer.epochs > 1:
            self.process(epoch+1, logs, self.prefix_epoch)

    def on_batch_end(self, global_step:int, local_step:int, logs:dict=None):
        if (global_step+1) % self.interval == 0:
            self.process(global_step+1, logs, self.prefix_step)

    def process(self, iteration:int, logs:dict, prefix:str):
        logs = logs or {}
        for k, v in logs.items():
            if k in SKIP_METRICS:
                continue
            index = k if '/' in k else f"{prefix}{k}"
            self.writer.add_scalar(index, v, iteration)


class SystemStateCallback(Tensorboard):
    '''监控system的状态

    :param log_dir: str, tensorboard文件的保存路径
    :param interval: int, 保存tensorboard的间隔
    :param gpu_id_list: List[int], 需要记录的gpuid列表
    :param pids: int/List[int], 需要记录的

    Examples:
    ```python
    >>> syscallback = SystemStateCallback('./ckpt/tensorboard/system')
    ```
    '''
    def __init__(self, log_dir:str, interval:int=100, gpu_id_list:List[int]=None, pids:Union[int,List[int]]=None, **kwargs):
        super(SystemStateCallback, self).__init__(log_dir, interval, **kwargs)
        self.gpu_id_list = gpu_id_list
        self.pids = pids or os.getpid()
        self.pids = [self.pids] if isinstance(self.pids, int) else self.pids

    def on_train_begin(self, logs:dict=None):
        try:
            import pynvml
            self.pynvml = pynvml
            self.pynvml.nvmlInit()
            import psutil
            self.psutil = psutil
            self.pre_read = {pid:psutil.Process(pid).io_counters().read_bytes for pid in self.pids}
            self.pre_write = {pid:psutil.Process(pid).io_counters().write_bytes for pid in self.pids}
            self.pre_time = time.time()
            super().on_train_begin()
        except ImportError:
            if not is_package_available('pynvml'):
                log_warn_once("SystemStateCallback requires pynvml to be installed. Run `pip install pynvml`; Skip this callback instead.")
            if not is_package_available('psutil'):
                log_warn_once("SystemStateCallback requires psutil to be installed. Run `pip install psutil`; Skip this callback instead.")
            self.run_callback = False
        except:
            log_warn_once(traceback.format_exc())
            self.run_callback = False

    def on_train_end(self, logs: dict = None):
        try:
            super().on_train_end(logs)
            self.pynvml.nvmlShutdown()
        except:
            log_warn_once(traceback.format_exc())

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        # 不记录
        pass

    def process(self, iteration:int, logs:dict, prefix:str):
        G = 1024*1024*1024
        logs = {}
        
        now_time = time.time()
        for pid in self.pids:
            p = self.psutil.Process(pid)
            # CPU使用情况
            logs[f"Pid_{pid}/CPU-Percent"] = p.cpu_percent(interval=0.5)

            # 内存使用情况
            logs[f"Pid_{pid}/Memory-Percent"] = p.memory_percent()
            logs[f"Pid_{pid}/Memory-RSS-G_byte"] = p.memory_info().rss/G
            logs[f"Pid_{pid}/Memory-VMS-G_byte"] = p.memory_info().vms/G

            # 进程IO信息
            data = p.io_counters()
            logs[f"Pid_{pid}/IO-read-M_byte"] = data.read_bytes/1024
            logs[f"Pid_{pid}/IO-write-M_byte"] = data.write_bytes/1024
            logs[f"Pid_{pid}/IO-readRate-M_byte_s"] = (data.read_bytes - self.pre_read[pid])/(now_time-self.pre_time)/1024
            logs[f"Pid_{pid}/IO-writeRate-M_byte_s"] = (data.write_bytes - self.pre_write[pid])/(now_time-self.pre_time)/1024
            self.pre_read[pid] = data.read_bytes
            self.pre_write[pid] = data.write_bytes
        self.pre_time = now_time
        
        # gpu使用情况       
        tesorboard_info_list = []
        if self.gpu_id_list is None:
            deviceCount = self.pynvml.nvmlDeviceGetCount()
            device_list = [i for i in range(deviceCount)]
        else:
            device_list = self.gpu_id_list

        tesorboard_info_list = []
        for i in device_list:
            tesorboard_info = {}
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
            tesorboard_info['gpu_name'] = self.pynvml.nvmlDeviceGetName(handle)
            meminfo = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            tesorboard_info['gpu_mem_used'] = meminfo.used/G
            UTIL = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            tesorboard_info['gpu_gpu_util'] = UTIL.gpu
            tesorboard_info['gpu_mem_util'] = UTIL.memory
            tesorboard_info_list.append(copy.deepcopy(tesorboard_info))
        for tesorboard_info,i in zip(tesorboard_info_list, device_list):
            logs[f'GPU/Gpu{i}-mem_used-unit_G'] = tesorboard_info['gpu_mem_used']
            logs[f'GPU/Gpu{i}-gpu_util'] = tesorboard_info['gpu_gpu_util']
            logs[f'GPU/Gpu{i}-mem_util'] = tesorboard_info['gpu_mem_util']

        for k, v in logs.items():
            self.writer.add_scalar(k, v, iteration)


class WandbCallback(Callback):
    '''从transformers迁移过来
    A :class:`~transformers.TrainerCallback` that sends the logs to `Weight and Biases <https://www.wandb.com/>`__.

    :param interval: int, log的的step间隔
    :param watch (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
        Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
        logging or :obj:`"all"` to log gradients and parameters.
    :param project: str, wandb的project name, 默认为bert4torch
    
    Examples:
    ```python
    >>> wandb = WandbCallback(save_code=True)
    ```
    '''
    def __init__(self, project:str='bert4torch', trial_name:str=None, run_name:str=None, watch:str='gradients', 
                 interval:int=100, save_code:bool=False, config:dict=None):
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            log_warn("WandbCallback requires wandb to be installed. Run `pip install wandb`; Skip this callback instead.")
            self._wandb = None
            self.run_callback = False
        except:
            log_warn(traceback.format_exc())
            self._wandb = None
            self.run_callback = False

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

    def define_metric(self, step_metric:str, logs:dict=None):
        if getattr(self._wandb, "define_metric", None):
            for m in logs.keys():
                if m not in self.metrics:
                    self._wandb.define_metric(name=m, step_metric=step_metric, hidden=True if m in SKIP_METRICS else False)
                    self.metrics.add(m)

    def adjust_logs(self, logs:dict, **kwargs):
        logs_new = {**logs, **kwargs}
        return {k:v for k,v in logs_new.items() if k not in SKIP_METRICS}

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        if self._wandb is None:
            return
        if hasattr(self.trainer, 'epochs') and self.trainer.epochs > 1:
            self.define_metric('epoch', logs)
            self._wandb.log(self.adjust_logs(logs, epoch=epoch+1))

    def on_batch_end(self, global_step:int, local_step:int, logs:dict=None):
        if self._wandb is None:
            return
        
        if (global_step+1) % self.interval == 0:
            self.define_metric('step', logs)
            self._wandb.log(self.adjust_logs(logs, step=global_step+1))
        
    def on_train_begin(self, logs:dict=None):
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

    def on_train_end(self, logs:dict=None):
        if self._wandb is None:
            return

        # transformer中的on_log
        self._wandb.finish()
        self._initialized = False


class LambdaCallback(Callback):
    '''lambda表达式
    '''
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
    def on_train_begin(self, logs:dict=None):
        try:
            from torchinfo import summary
            print()
            summary(self.model, input_data=next(iter(self.trainer.train_dataloader))[0])
            print()
        except ImportError:
            log_warn("Summary callback requires torchinfo to be installed. Run `pip install torchinfo`; Skip this callback instead")
            self.run_callback = False
        except:
            log_warn(traceback.format_exc())
            self.run_callback = False

class EmailCallback(Callback):
    '''发送Email

    :param mail_receivers: str/list, 收件人邮箱
    :param epoch_or_step: str, 控制是按照epoch还是step来发送邮件, 默认为'epoch', 可选{'step', 'epoch'}
    :param interval: int, 发送邮件的的step间隔
    :param mail_host: str, 发件服务器host
    :param mail_user: str, 发件人
    :param mail_pwd: str, smtp的第三方密码
    :param mail_sender: str, 发件人邮箱

    Examples:
    ```python
    >>> # 每个epoch结束后, 使用默认邮箱把对应的指标发送到指定邮箱
    >>> email = EmailCallback(mail_receivers='tongjilibo@163.com', epoch_or_step='epoch')
    
    >>> # 每个epoch结束后, 使用自定义邮箱把对应的指标发送到指定邮箱
    >>> email = EmailCallback(mail_receivers='tongjilibo@163.com', 
    ...                       epoch_or_step='epoch', 
    ...                       mail_host='smtp.163.com', 
    ...                       mail_user='bert4torch', 
    ...                       mail_pwd='VDSGQEHFXDZOCVEH', 
    ...                       mail_sender='bert4torch@163.com')
    ```
    '''
    def __init__(self, mail_receivers:Union[str,list], mail_subject:str='', epoch_or_step:Literal['epoch', 'step']='epoch', interval:int=100, 
                 mail_host:str=None, mail_user:str=None, mail_pwd:str=None, mail_sender:str=None, **kwargs):
        super(EmailCallback, self).__init__(**kwargs)
        self.epoch_or_step = epoch_or_step
        self.interval = interval
        self.mail_receivers = mail_receivers
        self.mail_subject = mail_subject
        self.mail_host = mail_host
        self.mail_user = mail_user
        self.mail_pwd = mail_pwd
        self.mail_sender = mail_sender

    def on_epoch_end(self, global_step:int, epoch:int, logs:dict=None):
        if self.epoch_or_step == 'epoch':
            msg = json.dumps({k:f'{round(v)}' for k,v in logs.items() if k not in SKIP_METRICS}, indent=2, ensure_ascii=False)
            subject = f'[INFO] Epoch {epoch+1} performance'
            if self.mail_subject != '':
                subject = self.mail_subject + ' | ' + subject
            self._email(subject, msg)

    def on_batch_end(self, global_step:int, local_step:int, logs:dict=None):
        if (self.epoch_or_step == 'step') and ((global_step+1) % self.interval == 0):
            msg = json.dumps({k:f'{round(v)}' for k,v in logs.items() if k not in SKIP_METRICS}, indent=2, ensure_ascii=False)
            subject = f'[INFO] Step {global_step} performance'
            if self.mail_subject != '':
                subject = self.mail_subject + ' | ' + subject
            self._email(subject, msg)

    def on_train_end(self, logs:dict=None):
        msg = json.dumps({k:f'{round(v)}' for k,v in logs.items() if k not in SKIP_METRICS}, indent=2, ensure_ascii=False)
        subject = f'[INFO] Finish training'
        if self.mail_subject != '':
            subject = self.mail_subject + ' | ' + subject
        self._email(subject, msg)

    def _email(self, subject:str, msg:str):
        send_email(self.mail_receivers, mail_subject=subject, mail_msg=msg, mail_host=self.mail_host,
                   mail_user=self.mail_user, mail_pwd=self.mail_pwd, mail_sender=self.mail_sender)

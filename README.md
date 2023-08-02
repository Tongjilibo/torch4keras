![torch4keras](https://github.com/Tongjilibo/torch4keras/blob/master/docs/pics/torch4keras.png)


[![licence](https://img.shields.io/github/license/Tongjilibo/torch4keras.svg?maxAge=3600)](https://github.com/Tongjilibo/torch4keras/blob/master/LICENSE) 
[![GitHub release](https://img.shields.io/github/release/Tongjilibo/torch4keras.svg?maxAge=3600)](https://github.com/Tongjilibo/torch4keras/releases) 
[![PyPI](https://img.shields.io/pypi/v/torch4keras?label=pypi%20package)](https://pypi.org/project/torch4keras/) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/torch4keras)](https://pypistats.org/packages/torch4keras)
[![GitHub stars](https://img.shields.io/github/stars/Tongjilibo/torch4keras?style=social)](https://github.com/Tongjilibo/torch4keras)
[![GitHub Issues](https://img.shields.io/github/issues/Tongjilibo/torch4keras.svg)](https://github.com/Tongjilibo/torch4keras/issues)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Tongjilibo/torch4keras/issues)

[Documentation](https://torch4keras.readthedocs.io) |
[Bert4torch](https://github.com/Tongjilibo/bert4torch) |
[Examples](https://github.com/Tongjilibo/torch4keras/blob/master/examples) |
[Source code](https://github.com/Tongjilibo/torch4keras)

## 1. 下载安装
安装稳定版
```shell
pip install torch4keras
```
安装最新版
```shell
pip install git+https://github.com/Tongjilibo/torch4keras.git
```

## 2. 功能
- 简述：抽象出来的Trainer，适用于一般神经网络的训练，仅需关注网络结构代码
- 特色：进度条展示训练过程，自定义metric，自带Evaluator, Checkpoint, Tensorboard, Logger等Callback，也可自定义Callback
- 初衷：前期功能是作为[bert4torch](https://github.com/Tongjilibo/bert4torch)和[rec4torch](https://github.com/Tongjilibo/rec4torch)的Trainer
- 训练：

    ```text
    2022-10-28 23:16:10 - Start Training
    2022-10-28 23:16:10 - Epoch: 1/5
    5000/5000 [==============================] - 13s 3ms/step - loss: 0.1351 - acc: 0.9601
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 798.09it/s] 
    test_acc: 0.98045. best_test_acc: 0.98045

    2022-10-28 23:16:27 - Epoch: 2/5
    5000/5000 [==============================] - 13s 3ms/step - loss: 0.0465 - acc: 0.9862
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 635.78it/s] 
    test_acc: 0.98280. best_test_acc: 0.98280

    2022-10-28 23:16:44 - Epoch: 3/5
    5000/5000 [==============================] - 15s 3ms/step - loss: 0.0284 - acc: 0.9915
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 673.60it/s] 
    test_acc: 0.98365. best_test_acc: 0.98365

    2022-10-28 23:17:03 - Epoch: 4/5
    5000/5000 [==============================] - 15s 3ms/step - loss: 0.0179 - acc: 0.9948
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 692.34it/s] 
    test_acc: 0.98265. best_test_acc: 0.98365

    2022-10-28 23:17:21 - Epoch: 5/5
    5000/5000 [==============================] - 14s 3ms/step - loss: 0.0129 - acc: 0.9958
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 701.77it/s] 
    test_acc: 0.98585. best_test_acc: 0.98585

    2022-10-28 23:17:37 - Finish Training
    ```

## 3. 快速上手
- 参考[bert4torch](https://github.com/Tongjilibo/bert4torch)的训练过程
- 简单示例: [turorials_mnist](https://github.com/Tongjilibo/torch4keras/blob/master/examples/turorials_mnist.py)

## 4. 版本说明
- **v0.1.0.post2**: 20230725 修复v0.1.0的bug，主要是进度条和log的标签平滑的问题
- **v0.1.0**: 20230724 允许调整进度条的显示参数, 进度条和日志同步（如果进度条平滑了则日志也平滑）, 自动把tensor转到model.device上, 允许打印第一个batch来检查样本
- **v0.0.9**：20230716 增加auto_set_cuda_devices自动选择显卡，增加log_info，log_warn, log_error等小函数
- **v0.0.8**：20230625 增加EmailCallback和WandbCallback, 增加AccelerateTrainer和DeepSpeedTrainer, grad_accumulation_steps内算一个batch，修改Trainer中部分成员函数
- **v0.0.7.post3**: 20230517 修复保存scheduler
- **v0.0.7.post2**: 20230517 Checkpoint Calback增加保存scheduler, save_weights可自行创建目录，Logger, Tensorboard模块加入lr, 修改predict和add_trainer
- **v0.0.7**：20230505 独立出callbacks.py文件, fit允许输入形式为字典，load_weights支持list输入，save_weights支持仅保存可训练参数
- **v0.0.6**：20230212 增加resume_from_checkpoint和save_to_checkpoint；增加add_trainer方法，重构了Trainer(BaseModel)的实现(增加几个成员变量、增加initilize()、删除对forward参数个数的判断、dp和ddp不解析module、修改use_amp参数为mixed_precision)，增加了AccelerateCallback
- **v0.0.5**：20221217 增加Summary的Callback, 增加Tqdm的进度条展示，保留原有BaseModel的同时，增加Trainer(不从nn.Module继承), 从bert4torch的snippets迁移部分通用函数
- **v0.0.4**：20221127 为callback增加on_train_step_end方法, 修复BaseModel(net)方式的bug
- **v0.0.3.post2**：20221107 修复DDP下打印的bug
- **v0.0.3**：20221106 参考Keras修改了callback的逻辑
- **v0.0.2**：20221023 增加Checkpoint, Evaluator等自带Callback, 修改BaseModel(net)方式，修复DP和DDP的__init__()
- **v0.0.1**：20221019 初始版本

## 5. 更新：
- **20230802**: 增加指标平滑的SmoothMetricCallback，统一管理指标平滑的问题, 增加SKIP_METRICS，NO_SMOOTH_METRICS，ROUND_PRECISION，默认对指标会进行平滑，修改tensorboard和wandb的callback, 允许跳过nan的指标, Tensorboard可以记录gpu等系统信息
- **20230725**: 修复v0.1.0的bug，主要是进度条和log的标签平滑的问题
- **20230724**: 允许调整进度条的显示参数, 进度条和日志同步（如果进度条平滑了则日志也平滑）, 自动把tensor转到model.device上, 允许打印第一个batch来检查样本
- **20230716**：增加auto_set_cuda_devices自动选择显卡，增加log_info，log_warn, log_error等小函数
- **20230625**：增加EmailCallback和WandbCallback, 增加AccelerateTrainer和DeepSpeedTrainer, grad_accumulation_steps内算一个batch，修改Trainer中部分成员函数
- **20230517**：Checkpoint Calback增加保存scheduler, save_weights可自行创建目录，Logger, Tensorboard模块加入lr, 修改predict和add_trainer
- **20230505**：独立出callbacks.py文件, fit允许输入形式为字典，load_weights支持list输入，save_weights支持仅保存可训练参数
- **20230212**：增加hf的accelerator测试用例, ddp需要外部控制执行callback, 混合精度支持bf16
- **20230116**：增加resume_from_checkpoint和save_to_checkpoint，动态为nn.Module增加Trainer的方法add_trainer
- **20221217**：保留原有BaseModel的同时，增加Trainer(不从nn.Module继承), 从bert4torch的snippets迁移部分通用函数
- **20221203**：增加Summary的Callback, 增加Tqdm的进度条展示
- **20221127**：为callback增加on_train_step_end方法, 修复BaseModel(net)方式的bug
- **20221107**：修复DDP下打印的bug，metrics中加入detach和auc
- **20221106**：默认的Tensorboard的global_step+1, 参考Keras修改了callback的逻辑
- **20221020**：增加Checkpoint, Evaluator等自带Callback, 修改BaseModel(net)方式，修复DP和DDP的__init__()
- **20221019**：初版提交

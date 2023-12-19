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

## 4. 版本历史
|更新日期| 版本 | 版本说明 |
|------| ----------------- |----------- |
|20231219|v0.1.7           | 增加SimpleStreamFileLogger和LoggerHandler, 修改Logger的格式|
|20231208|v0.1.6.post2     |监控fit过程，有报错则发送邮件提醒; 解决torch2.0的compile冲突问题; 修复clip_grad_norm的bug|
|20230928|v0.1.5     |进度条中显示已经训练的时间|

[更多版本](https://github.com/Tongjilibo/torch4keras/blob/master/docs/Update.md)

## 5. 更新历史：
- **20231219**: 增加SimpleStreamFileLogger和LoggerHandler, 修改Logger的格式
- **20231208**: 监控fit过程，有报错则发送邮件提醒; 解决torch2.0的compile冲突问题; 修复clip_grad_norm的bug
- **20230928**: 进度条中显示已经训练的时间

[更多历史](https://github.com/Tongjilibo/torch4keras/blob/master/docs/History.md)

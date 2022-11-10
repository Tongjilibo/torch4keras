# torch4keras
**Use torch like keras**

[![licence](https://img.shields.io/github/license/Tongjilibo/torch4keras.svg?maxAge=3600)](https://github.com/Tongjilibo/torch4keras/blob/master/LICENSE) 
[![GitHub release](https://img.shields.io/github/release/Tongjilibo/torch4keras.svg?maxAge=3600)](https://github.com/Tongjilibo/torch4keras/releases) 
[![PyPI](https://img.shields.io/pypi/v/torch4keras?label=pypi%20package)](https://pypi.org/project/torch4keras/) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/torch4keras)](https://pypistats.org/packages/torch4keras)
[![GitHub stars](https://img.shields.io/github/stars/Tongjilibo/torch4keras?style=social)](https://github.com/Tongjilibo/torch4keras)
[![GitHub Issues](https://img.shields.io/github/issues/Tongjilibo/torch4keras.svg)](https://github.com/Tongjilibo/torch4keras/issues)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Tongjilibo/torch4keras/issues)

## 1. 下载安装
安装稳定版
```shell
pip install torch4keras
```
安装最新版
```shell
pip install git+https://www.github.com/Tongjilibo/torch4keras.git
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
- **v0.0.3post2**：20221107 修复DDP下打印的bug
- **v0.0.3**：20221106 参考Keras修改了callback的逻辑
- **v0.0.2**：20221023 增加Checkpoint, Evaluator等自带Callback, 修改BaseModel(net)方式，修复DP和DDP的__init__()
- **v0.0.1**：20221019 初始版本

## 5. 更新：
- **20221107**：修复DDP下打印的bug，metrics中加入detach
- **20221106**：默认的Tensorboard的global_step+1, 参考Keras修改了callback的逻辑
- **20221020**：增加Checkpoint, Evaluator等自带Callback, 修改BaseModel(net)方式，修复DP和DDP的__init__()
- **20221019**：初版提交

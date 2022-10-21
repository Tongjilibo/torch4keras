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
- 简述：抽象出来的Trainer，适用于一般神经网络的训练
- 特色：进度条展示训练过程，自定义metric，自带Evaluator, Checkpoint, Tensorboard, Logger等Callback，也可自定义Callback
- 初衷：前期功能是作为[bert4torch](https://github.com/Tongjilibo/bert4torch)和[rec4torch](https://github.com/Tongjilibo/rec4torch)的Trainer

## 3. 快速上手
- 参考[bert4torch](https://github.com/Tongjilibo/bert4torch)的训练过程
- 简单示例[turorials_mnist](https://github.com/Tongjilibo/torch4kerass/blob/master/examples/turorials_mnist.py)

## 4. 版本说明
- **v0.0.1**：20221019 初始版本

## 5. 更新：
- **20221020**：增加Checkpoint, Evaluator等自带Callback, 修改BaseModel为Model，支持Model(net)方式
- **20221019**：初版提交
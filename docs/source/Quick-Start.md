# torch4keras
**Use torch like keras**

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
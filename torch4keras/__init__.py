from torch import nn

# torch2.0以后nn.Module自带compile方法
# 和model.compile()冲突, 这里把nn.Module.compile重命名为compile_torch
if hasattr(nn.Module, 'compile'):
    nn.Module.compile_torch = nn.Module.compile
    del nn.Module.compile
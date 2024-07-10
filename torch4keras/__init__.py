from .snippets import is_torch_available, log_warn_once

if is_torch_available():
    from torch import nn

    # torch2.0以后nn.Module自带compile方法
    # 和model.compile()冲突, 这里把nn.Module.compile重命名为compile_torch
    if hasattr(nn.Module, 'compile'):
        nn.Module.compile_torch = nn.Module.compile
        del nn.Module.compile
else:
    log_warn_once("PyTorch have not been found. Models won't be available and only snippets can be used.")

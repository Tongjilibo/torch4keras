import importlib.util
from typing import Any, Tuple, Union
from packaging import version
from functools import lru_cache
from packaging.version import Version, parse
from .log import log_warn_once, log_error_once
import os
import sys
import operator as op
from contextlib import contextmanager
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}


def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Compares a library version to some requirement using a given operation.

    Args:
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib.metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        # print(f"Detected {pkg_name} version {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


def is_torch_available():
    return is_package_available("torch")


def is_torch_version(operation: str, version: str):
    """
    Compares the current PyTorch version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    torch_version = parse(importlib.metadata.version("torch"))
    return compare_versions(torch_version, operation, version)


def is_safetensors_available():
    return is_package_available("safetensors")


def is_sklearn_available():
    return is_package_available("sklearn")


def is_deepspeed_available():
    package_exists = importlib.util.find_spec("deepspeed") is not None

    # Check we're not importing a "deepspeed" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    if package_exists:
        try:
            _ = importlib_metadata.metadata("deepspeed")
            return True
        except importlib_metadata.PackageNotFoundError:
            return False


def is_accelerate_available(check_partial_state=False):
    '''是否可以使用accelerate'''
    accelerate_available = importlib.util.find_spec("accelerate") is not None
    if accelerate_available:
        if check_partial_state:
            return version.parse(importlib_metadata.version("accelerate")) >= version.parse("0.17.0")
        else:
            return True
    else:
        return False


@lru_cache
def is_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch
    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache
def is_xpu_available(check_device=False):
    "check if user disables it explicitly"

    def str_to_bool(value) -> int:
        """
        Converts a string representation of truth to `True` (1) or `False` (0).

        True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
        """
        value = value.lower()
        if value in ("y", "yes", "t", "true", "on", "1"):
            return 1
        elif value in ("n", "no", "f", "false", "off", "0"):
            return 0
        else:
            raise ValueError(f"invalid truth value {value}")
    
    def parse_flag_from_env(key, default=False):
        """Returns truthy value for `key` from the env if available else the default."""
        value = os.environ.get(key, str(default))
        return str_to_bool(value) == 1  # As its name indicates `str_to_bool` actually returns an int...


    if not parse_flag_from_env("ACCELERATE_USE_XPU", default=True):
        return False
    "Checks if `intel_extension_for_pytorch` is installed and potentially if a XPU is in the environment"
    if is_ipex_available():
        import torch

        if is_torch_version("<=", "1.12"):
            return False
    else:
        return False

    import intel_extension_for_pytorch  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no XPU  is found
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_mps_available():
    import torch
    return is_torch_version(">=", "1.12") and torch.backends.mps.is_available() and torch.backends.mps.is_built()


@contextmanager
def safe_import(pkg_name: str = None):
    '''import some module safely
    - 部分包中使用到某些函数，import出错也不影响其他函数的运行
    - 如果下面函数入参类型声明使用到了这个包，这种方式也会报错，比如定义`def func(df:pd.DataFrame)`，但是不存在pandas包

    :param pkg_name: str, 包名，用于检查module是否存在，默认为None表示不检查

    ### Example
    ```python
    from torch4keras.snippets import safe_import
    with safe_import('pptx') as si:
        if si:  # 包存在才执行
            from pptx import Presentation

    # 不验证包是否存在, import出错不影响运行
    with safe_import():
        import fitz
    '''
    try:
        if pkg_name is None:  # 未指定包名
            yield True
        elif not is_package_available(pkg_name): # 指定包名但不存在
            log_warn_once(f"No module named '{pkg_name}'")
            yield False
        else:  # 指定报名且存在
            yield True
    except Exception as e:
        log_warn_once(f"{e}")
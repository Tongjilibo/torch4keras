import importlib.util
from typing import Any, Tuple, Union
from packaging import version
import sys
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


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
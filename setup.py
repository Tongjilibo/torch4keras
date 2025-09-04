#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
import re


with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

def get_version() -> str:
    with open(os.path.join("torch4keras", "cli.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        version = re.findall(pattern, file_content)[0]
        return version

def get_console_scripts() -> list[str]:
    console_scripts = [
        "torch4keras = torch4keras.cli:main",
        "t4k = torch4keras.cli:main"
        ]
    return console_scripts


setup(
    name='torch4keras',
    version=get_version(),
    description='Use torch like keras',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License',
    url='https://github.com/Tongjilibo/torch4keras',
    author='Tongjilibo',
    install_requires=['numpy', 'packaging'],
    packages=find_packages(),
    entry_points={"console_scripts": get_console_scripts()},
)
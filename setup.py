#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='torch4keras',
    version='v0.2.9.post2',
    description='Use torch like keras',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License',
    url='https://github.com/Tongjilibo/torch4keras',
    author='Tongjilibo',
    install_requires=['numpy', 'packaging'],
    packages=find_packages()
)
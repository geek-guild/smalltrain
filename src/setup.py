# setup.py
#
# Copyright 2020 Geek Guild Co., Ltd.
#
import setuptools
from distutils.core import setup

setup(
    name='smalltrain',
    version='0.2.1.0',
    description='SmallTrain',
    author='Geek Guild Co., Ltd.',
    author_email='labo@geek-guild.jp',
    url='https://www.geek-guild.jp',
    packages=setuptools.find_packages(
        exclude=['configs', 'log']
    ),
    package_dir={
        'smalltrain': './smalltrain'
    }
)

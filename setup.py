##
## @file        setup.py
## @brief       My code for Chainer
## @author      Keitetsu
## @date        2020/05/24
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


from setuptools import setup, find_packages


setup(
    name='keitetsuchainerlib',
    version='0.1',
    description='My code for Chainer',
    author='Keitetsu',
    author_email='keitetsu@gmail.com',
    url='https://github.com/KeitetsuWorks/chainerlib',
    packages=find_packages(exclude=['examples'])
)

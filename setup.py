from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "Require Python 3.6 or greater."

setup(
    name='redq',
    py_modules=['redq'],
    version='0.0.1',
    install_requires=[
        'numpy',
        'joblib',
        'gym>=0.17.2'
    ],
    description="REDQ algorithm PyTorch implementation",
    author="Xinyue Chen, Che Wang, Zijian Zhou, Keith Ross",
)

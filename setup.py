from distutils.core import setup
from setuptools import find_packages

setup(
    name='ACT_plus_plus',
    version='0.0.0',
    packages=find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
    entry_points={
        # <命令名> = <模块路径>:<函数名>
        'console_scripts': [
            'detr = ACT_plus_plus.detr.setup:setup',
            'robomimic = ACT_plus_plus.robomimic.setup:setup',
            'mobile-aloha = ACT_plus_plus.mobile-aloha.setup:setup',
        ]
    }
)

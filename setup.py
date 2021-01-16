#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='OCR-Captcha-Cracker',
    version='0.1.0',
    include_package_data=True,
    packages=find_packages(exclude=["examples"]),
    install_requires=[
        'pandas==1.1.2',
        'numpy==1.19.1',
        'matplotlib>=2.2.0',
        'jupyter',
        'torch==1.6.0',
        'torchvision==0.7.0',
        'tqdm==4.48.2',
        'Pillow==6.2.2',
        'pytorch-crf==0.7.2',
        'matplotlib>=3.3.0',
        'tensorboard>=2.4.0',
        'captcha==0.3',
        'editdistance==0.3.1',
    ],
    python_requires=">=3.6",
)

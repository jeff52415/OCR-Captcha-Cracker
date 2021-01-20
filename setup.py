#!/usr/bin/env python
from setuptools import find_packages, setup


dev_requires = [
    "jupyter",
    "matplotlib>=3.3.0",
    "pandas==1.1.2",
]

serving_requires = ["flask==1.1.2"]

lint_requires = [
    "black==19.10b0",
    "pylint==2.5.2",
]

test_requires = [
    "coverage==5.1",
    "pytest==5.4.2",
    "pytest-cov==2.9.0",
    "pytest-html==2.1.1",
    "tox==3.15.1",
]

torch_requires = [
    "torch>=1.6.0",
    "torchvision>=0.7.0",
]

setup(
    name="OCR-Captcha-Cracker",
    version="0.1.0",
    include_package_data=True,
    packages=find_packages(exclude=["examples"]),
    install_requires=[
        "captcha==0.3",
        "editdistance==0.5.3",
        "numpy==1.19.1",
        "Pillow==6.2.2",
        "pytorch-crf==0.7.2",
        "tensorboard>=2.4.0",
        "tqdm==4.48.2",
    ],
    python_requires=">=3.6",
    extras_require={
        "serving": serving_requires + torch_requires,
        "dev": dev_requires + serving_requires + lint_requires + test_requires + torch_requires,
        "lint": lint_requires + torch_requires,
        "test": serving_requires + test_requires + torch_requires,
        "torch": torch_requires,
    },
)

#!/usr/bin/env python
from setuptools import find_packages, setup
import platform
import subprocess
import warnings
import re

def get_cuda_win_pkg_url(package_name, version):
    "get dowload url for torch/torchvision package on windows system"

    if package_name == "torch":
        base_link = {
            "gpu": "torch@https://download.pytorch.org/whl/cu<cuda_ver>/torch-<version>-cp36-cp36m-win_amd64.whl",
            "cpu": "https://download.pytorch.org/whl/cpu/torch-<version>%2Bcpu-cp36-cp36m-win_amd64.whl",
        }
    elif package_name == "torchvision":
        base_link = {
            "gpu": "https://download.pytorch.org/whl/cu<cuda_ver>/torchvision-<version>%2Bcu<cuda_ver>-cp36-cp36m-win_amd64.whl",
            "cpu": "https://download.pytorch.org/whl/cpu/torchvision-<version>%2Bcpu-cp36-cp36m-linux_x86_64.whl",
        }
        
    url = base_link["cpu"]
    url = url.replace("<version>", version)
    return url

def torch_urls(version):
    "get torch version based on system type"
    
    platform_system = platform.system()
    if platform_system == "Windows":
        return get_cuda_win_pkg_url("torch", version)
    return f"torch=={version}"


def torchvision_urls(version):
    "get torchvision version based on system type"
    
    platform_system = platform.system()
    if platform_system == "Windows":
        return get_cuda_win_pkg_url("torchvision", version)
    return f"torchvision=={version}"

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

setup(
    name="OCR-Captcha-Cracker",
    version="0.1.0",
    include_package_data=True,
    packages=find_packages(exclude=["examples"]),
    install_requires=[
        torch_urls("1.6.0"),
        torchvision_urls("0.7.0"),
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
        "serving": serving_requires,
        "dev": dev_requires + serving_requires + lint_requires + test_requires,
        "lint": lint_requires,
        "test": serving_requires + test_requires,
    },
)

"""
Package installation file
"""

from setuptools import find_packages, setup

setup(
    name="amazon_seg_project",
    version="1.0.0",
    packages=find_packages(exclude=["amazon_seg_project_tests"]),
    install_requires=[
        "albumentations >= 2.0, < 2.1",
        "boto3 >= 1.38, < 1.39",
        "dagster >= 1.10, < 1.11",
        "dagster-aws >= 0.26, < 0.27",
        "dagster-webserver >= 1.10, < 1.11",
        "matplotlib >= 3.10, < 3.11",
        "moto >= 5.1, < 5.2",
        "numpy >= 1.26, < 2.0",
        "pandas >= 2.2, < 2.3",
        "pydantic >= 2.11, < 2.12",
        "pytest >= 8.3, < 8.4",
        "python-dotenv >= 1.1, < 1.2",
        "rioxarray >= 0.19, < 0.20",
        "segmentation_models_pytorch >= 0.5, < 0.6",
        "torch >= 2.2, < 2.3",
        "torchvision >= 0.17, < 0.18",
        "wandb >= 0.19, < 0.20",
        "xarray >= 2025.4, < 2025.5",
    ],
    extras_require={
        "dev": [
            "torchinfo >= 1.8, < 1.9",
        ]
    },
    author="Akshay Suresh",
    author_email="akshay721@gmail.com",
    url="https://github.com/akshaysuresh1/amazonforest_segmentation",
    license="MIT",
)

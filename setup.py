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
        "boto3 >= 1.34, < 1.35",
        "dagster >= 1.9, < 1.10",
        "dagster-aws >= 0.25, < 0.26",
        "dagster-webserver >= 1.9, < 1.10",
        "moto >= 5.0, < 5.1",
        "numpy >= 1.26, < 2.0",
        "pandas >= 2.2, < 2.3",
        "pydantic >= 2.10, < 2.11",
        "pytest >= 8.2, < 8.3",
        "python-dotenv >= 1.0, < 1.1",
        "rioxarray >= 0.17, < 0.18",
        "segmentation_models_pytorch >= 0.4, < 0.5",
        "torch >= 2.2, < 2.3",
        "torchvision >= 0.17, < 0.18",
        "xarray >= 2024.6, < 2024.7",
    ],
    extras_require={
        "dev": [
            "matplotlib >= 3.10, < 3.11",
            "torchinfo >= 1.8, < 1.9",
        ]
    },
    author="Akshay Suresh",
    author_email="akshay721@gmail.com",
    url="https://github.com/akshaysuresh1/amazonforest_segmentation",
    license="MIT",
)

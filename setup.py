"""
Package installation file
"""

from setuptools import find_packages, setup

setup(
    name="amazon_seg_project",
    version="1.0.0",
    packages=find_packages(exclude=["amazon_seg_project_tests"]),
    install_requires=[
        "numpy >= 1.26, < 2.0",
        "pandas >= 2.2, < 2.3",
        "boto3 >= 1.34, < 1.35",
        "dagster >= 1.9, < 1.10",
        "dagster-aws >= 0.25, < 0.26",
        "rioxarray >= 0.17, < 0.18",
        "xarray >= 2024.6, < 2024.7",
        "python-dotenv >= 1.0, < 1.1",
        "torch >= 2.2, < 2.3",
        "torchvision >= 0.17, < 0.18",
        "segmentation_models_pytorch >= 0.4, < 0.5",
        "albumentations >= 2.0, < 2.1",
    ],
    extras_require={
        "dev": [
            "dagster-webserver >= 1.9, < 1.10",
            "pytest >= 8.2, < 8.3",
            "moto >= 5.0, < 5.1",
        ]
    },
    author="Akshay Suresh",
    author_email="akshay721@gmail.com",
    url="https://github.com/akshaysuresh1/amazonforest_segmentation",
    license="MIT",
)

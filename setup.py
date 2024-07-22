from setuptools import find_packages, setup

setup(
    name="amazon_seg_project",
    packages=find_packages(exclude=["amazon_seg_project_tests"]),
    install_requires=[
        "boto3==1.34.145",
        "rioxarray==0.17.0",
        "xarray==2024.6.0",
    ],
    extras_require={"dev": ["pytest==8.2.2"]},
)

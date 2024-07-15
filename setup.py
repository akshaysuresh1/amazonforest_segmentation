from setuptools import find_packages, setup

setup(
    name="amazon_seg_project",
    packages=find_packages(exclude=["amazon_seg_project_tests"]),
    install_requires=["python-dotenv"],
    extras_require={"dev": ["pytest==8.2.2"]},
)

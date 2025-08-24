from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="PRODUCT RECOMMENDER SYSTEM",
    version="0.1",
    author="Deepanshu",
    packages=find_packages(),
    install_requires = requirements,
)
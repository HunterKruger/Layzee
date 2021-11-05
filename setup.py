import importlib
from setuptools import setup,find_packages
import json

with open("config.json","r") as f:
    config = json.load(f)

with open("requirements.txt",'r') as f:
    install_requires = f.readlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

package = importlib.import_module(config["project_name"])

setup(
    name=config["project_name"],
    version=package.__version__,
    url=config['project_url'],
    packages=find_packages(),
    install_requires=install_requires,
    author="Yuan Feng, Mashimaro",
    author_email="yuanfeng9606@gmail.com",
    description="A library for lazy data scientists ^_^",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
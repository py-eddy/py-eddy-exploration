#!/usr/bin/env python
from setuptools import setup, find_packages
import versioneer

min_version = (3, 10)

with open("requirements.txt", "r") as fh:
    requirements = fh.read().split("\n")

setup(
    name="pyEddyExplorer",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Py-Eddy-Tracker exporation",
    author="Antoine Delepoulle",
    author_email="delepoulle.a@gmail.com",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=f'>={".".join(str(i) for i in min_version)}',
)
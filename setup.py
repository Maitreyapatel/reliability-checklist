#!/usr/bin/env python

from setuptools import find_packages, setup
import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if "yaml" in filename:
                paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('reliability_score/configs/')

setup(
    name="reliability-score",
    version="0.0.1",
    description="A suite of reliability tests on NLP models.",
    author="Maitreya Patel",
    author_email="patel.maitreya57@gmail.com",
    url="https://github.com/Maitreyapatel/reliability-score",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    package_data={"reliability_score": extra_files},
    entry_points={"console_scripts": ["rs=reliability_score.eval:main"]},
    include_package_data=True,
)

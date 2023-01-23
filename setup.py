#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="reliability-score",
    version="0.0.1",
    description="A suite of reliability tests on NLP models.",
    author="Maitreya Patel",
    author_email="patel.maitreya57@gmail.com",
    url="https://github.com/Maitreyapatel/reliability-score",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    entry_points={"console_scripts": ["rs=reliability_score.eval:main"]},
)

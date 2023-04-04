#!/usr/bin/env python

import os

from setuptools import find_packages, setup


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if "yaml" in filename:
                paths.append(os.path.join("..", path, filename))
    return paths


extra_files = package_files("reliability_checklist/configs/")

setup(
    name="reliability-checklist",
    version="0.0.3",
    description="A suite of reliability tests on NLP models.",
    author="Maitreya Patel",
    author_email="patel.maitreya57@gmail.com",
    url="https://github.com/Maitreyapatel/reliability-checklist",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    package_data={"reliability_checklist": extra_files},
    entry_points={"console_scripts": ["recheck=reliability_checklist.eval:main"]},
    include_package_data=True,
)

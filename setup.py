from setuptools import setup, find_packages
import os

# Read README if it exists, otherwise use description
title = "tabsyn Syntheas"
long_description = title
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="tabsyn_zenml",
    version="0.1.0",
    packages=find_packages(include=['tabsyn_zenml', 'tabsyn_zenml.*']),
    install_requires=[],
    author="virsel",
    author_email="your.email@example.com",
    description=title,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11,<3.12",
)
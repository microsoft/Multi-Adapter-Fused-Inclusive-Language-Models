"""Setups the project."""

from setuptools import find_packages, setup

# Uses the readme as the description on PyPI
with open("README.md") as fh:
    long_description = ""
    header_count = 0
    for line in fh:
        if line.startswith("#"):
            header_count += 1
        if header_count < 2:
            long_description += line
        else:
            break

# Get requirements
with open("requirements.txt") as fh:
    reqs = fh.read().split("\n")
    if reqs[-1] == "":
        reqs = reqs[:-1]


setup(
    author="Sunyana Sitaram",
    author_email="sunayana.sitaram@microsoft.com",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="Inclusivity Toolkit: An easy to use tool to evaluate biases in LLMs",
    install_requires=reqs,
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="inclusivity_toolkit",
    python_requires=">=3.8",
    version="0.2",
    packages=[
        package
        for package in find_packages()
        if package.startswith("inclusivity_toolkit")
    ],
    include_package_data=True,
)

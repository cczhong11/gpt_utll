# setup.py for gpt_util
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
with open("README.md", "r") as f:
    long_description = f.read()
setup(
    name="gpt_util",
    version="0.1",
    description="A utility package for LLM",
    author="tc zhong",
    license="MIT",
    long_description=long_description,
    author_email="me@tczhong.com",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

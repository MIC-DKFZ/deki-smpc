from pathlib import Path

from setuptools import find_packages, setup


def read_readme() -> str:
    return Path("README.md").read_text(encoding="utf-8")


setup(
    name="deki_smpc",
    version="0.1",
    author="Benjamin Hamm",
    author_email="benjamin.hamm@dkfz-heidelberg.de",
    description="A lightweight client for Secure Multi-Party Computation (SMPC)-based Federated Learning (FL) framework, enabling seamless integration of privacy-preserving aggregation into custom training workflows.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "requests==2.32.5",
        "torch==2.8.0",
        "lz4==4.4.4",
        "urllib3==2.5.0",
        "numpy==2.3.3",
        "pydantic==2.12.2",
        "httpx==0.28.1",
        "tqdm==4.67.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)

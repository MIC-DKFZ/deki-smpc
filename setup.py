from setuptools import find_packages, setup

setup(
    name="deki_smpc",
    version="0.1",
    author="Benjamin Hamm",
    author_email="benjamin.hamm@dkfz-heidelberg.de",
    description="A lightweight client for Secure Multi-Party Computation (SMPC)-based Federated Learning (FL) framework, enabling seamless integration of privacy-preserving aggregation into custom training workflows.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["requests", "torch", "lz4", "urllib3"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

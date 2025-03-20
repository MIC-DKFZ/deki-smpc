from setuptools import find_packages, setup

setup(
    name="deki_smpc",
    version="0.1",
    author="Benjamin Hamm",
    author_email="benjamin.hamm@dkfz-heidelberg.de",
    description="A sample Python package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        # List dependencies here, e.g. "numpy", "requests"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

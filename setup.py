from setuptools import setup, find_packages

setup(
    name="pattern_formation",
    version="0.1.0",
    author="Your Team Name",
    description="Turing pattern formation via reaction-diffusion systems",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "dask>=2023.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
        ]
    },
)

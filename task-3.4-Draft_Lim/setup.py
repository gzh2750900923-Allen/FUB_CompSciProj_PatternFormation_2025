from setuptools import setup, find_packages

setup(
    name="turing_pattern_formation",
    version="0.1.0",
    author="Samuel Kim",
    author_email="samuel.kim@fu-berlin.de",  # Your university email
    description="Numerical solvers for Turing pattern formation in reaction-diffusion systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/pattern-formation", # GitLab/GitHub URL
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "dask[complete]",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
    ],
)

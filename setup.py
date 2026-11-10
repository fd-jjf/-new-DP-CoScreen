from setuptools import setup, find_packages

setup(
    name="dp-coscreen",
    version="1.0.0",
    description="Differentially Private Orthogonal Feature Screening for Industrial IoT Data",
    author="DP-CoScreen Authors",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "threadpoolctl>=3.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
)

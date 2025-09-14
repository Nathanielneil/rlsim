"""
Setup script for DVLN Baseline package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dvln-baseline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Visual-Language Navigation Baseline for UAV using AirSim",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dvln_baseline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Microsoft :: Windows :: Windows 11",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dvln-train-ppo=experiments.train_ppo:main",
            "dvln-train-dqn=experiments.train_dqn:main",
            "dvln-train-sac=experiments.train_sac:main",
            "dvln-evaluate=experiments.evaluate:main",
            "dvln-compare=experiments.compare_algorithms:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dvln_baseline": [
            "config/*.json",
            "config/*.yaml",
            "config/*.yml",
        ],
    },
    zip_safe=False,
)
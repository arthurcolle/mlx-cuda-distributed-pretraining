from setuptools import setup, find_packages

setup(
    name="mlx-pretrain",
    version="0.1.0",
    description="MLX-based pretraining for language models",
    author="MLX Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "mlx",
        "numpy",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
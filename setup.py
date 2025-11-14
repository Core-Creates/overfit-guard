from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="overfit-guard",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A micro-library to detect and correct overfitting in machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/overfit-guard",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "pytorch": ["torch>=1.9.0"],
        "tensorflow": ["tensorflow>=2.6.0"],
        "sklearn": ["scikit-learn>=1.0.0"],
        "all": [
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
            "scikit-learn>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
)

from setuptools import setup, find_packages

setup(
    name="SOFTS",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # List your dependencies here
        "scikit-learn",
        "numpy",
        "pandas",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            # Define command-line scripts here
        ],
    },
    author="Xu-Yang Chen,Lu Han",
    author_email="your.email@example.com",
    description='Official implement for "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion" in PyTorch.',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Secilia-Cxy/SOFTS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

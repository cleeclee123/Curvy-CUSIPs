from pathlib import Path

import setuptools

VERSION = "0.0.5"  

NAME = "CurvyCUSIPs" 

INSTALL_REQUIRES = [
    "aiohttp>=3.10.10",
    "httpx>=0.27.2",
    "matplotlib>=3.9.2",
    "numpy>=2.1.3",
    "pandas>=2.2.3",
    "plotly>=5.24.1",
    "polars>=1.12.0",
    "QuantLib>=1.36",
    "QuantLib>=1.36",
    "rateslib>=1.5.0",
    "Requests>=2.32.3",
    "scikit_learn>=1.5.2",
    "scipy>=1.14.1",
    "seaborn>=0.13.2",
    "statsmodels>=0.14.4",
    "termcolor>=2.5.0",
    "tqdm>=4.67.0",
    "ujson>=5.10.0",
]


setuptools.setup(
    name=NAME,
    version=VERSION,
    description="UST Research Tools",
    url="https://github.com/yieldcurvemonkey/Curvy-CUSIPs",
    project_urls={
        "Source Code": "https://github.com/yieldcurvemonkey/Curvy-CUSIPs",
    },
    author="Nonquantitative Quant",
    author_email="yieldcurvemonkey@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src/"),
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
)

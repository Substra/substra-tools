import os
from codecs import open

from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), "r", "utf-8") as fp:
    readme = fp.read()

about = {}
with open(os.path.join(here, "substratools", "__version__.py"), "r", "utf-8") as fp:
    exec(fp.read(), about)


setup(
    name="substratools",
    version=about["__version__"],
    description="Python tools to submit functions on the Substra platform",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Substra/substra-tools",
    keywords=["substra"],
    author="Owkin, Inc.",
    author_email="fldev@owkin.com",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    extras_require={
        "test": [
            "flake8",
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "numpy",
        ],
    },
    zip_safe=False,
)

from setuptools import setup, find_packages


setup(
    name='substratools',
    version='0.0',
    description='Python tools to submit algo on the Substra platform',
    author='Camille Marini',
    packages=find_packages(),
    install_requires=[
        'click',
        'numpy',
        'sklearn'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
    zip_safe=False
)

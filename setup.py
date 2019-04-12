from setuptools import setup, find_packages


setup(
    name='substratools',
    version='0.0',
    description='Python tools to submit algo on the Substra platform',
    author='Camille',
    packages=find_packages(),
    install_requires=['numpy', 'sklearn'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
    zip_safe=False
)

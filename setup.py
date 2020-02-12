from codecs import open
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), 'r', 'utf-8') as fp:
    readme = fp.read()

about = {}
with open(os.path.join(here, 'substratools', '__version__.py'),
          'r', 'utf-8') as fp:
    exec(fp.read(), about)


setup(
    name='substratools',
    version=about['__version__'],
    description='Python tools to submit algo on the Substra platform',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/SubstraFoundation/substra-tools',
    keywords=['substra'],
    author='Owkin',
    author_email='fldev@owkin.com',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['flake8', 'pytest', 'pytest-cov', 'pytest-mock'],
    zip_safe=False
)

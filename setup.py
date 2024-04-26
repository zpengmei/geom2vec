# an setup file for the geom2vec package

from setuptools import setup, find_packages

setup(
    name='geom2vec',
    version='0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'deeptime',
    ]
)


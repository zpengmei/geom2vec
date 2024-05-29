from setuptools import setup, find_packages

setup(
    name='geom2vec',
    version='0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'deeptime',
        'torch>=2.0',
        'torch_geometric',
        'torch_scatter',
    ]
)


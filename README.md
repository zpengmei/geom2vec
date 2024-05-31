# Geom2vec

![scheme](.figs/scheme.jpg)

Geom2Vec (Geometry-to-vector) is a framework for compute vector representation of molecular conformations using 
pretrained graph neural networks (GNNs). The resulting vectors can be used for dimensionality reduction, committer
function estimation, and in principle any other learnable task for dynamic analysis of molecular simulations.
By avoiding the need to retrain the GNNs for each new simulation ana analysis pipeline, 
the framework allows for efficient exploration of the dynamics data, which is usually vast in both system size 
and timescale. Comparing to other current or possible (not-yet appearing) graph-based methods, Geom2Vec guarantees 
orders of magnitude advantage in terms of computational efficiency and scalability in both time and memory.

## Installation
The package is based on PyTorch and PyTorch Geometric. 
Follow the instructions on 
the [PyTorch Geometric website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) 
to install the relevant packages.

Clone the repository and install the package using pip:
```bash
pip install -e .
```

## Tutorials


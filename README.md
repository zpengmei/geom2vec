# Geom2vec

![scheme](figs/scheme.jpg)

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
## Package structure
The package is organized as follows:
- `geom2vec` contains the main classes and functions for the framework.
- `checkpoints` contains the pretrained GNNs with different architectures.
- `tutorial` contains basic tutorials for using the package.

Under `geom2vec`:
- `geom2vec.data` contains the data-relevant class and processing utils.
- `geom2vec.downstream_models` contains models for downstream tasks, e.g., committer function estimation.
- `geom2vec.layers` contains building blocks (MLPs and Token mixing layers) for the general network architecture.
Instead, users should directly use the `geom2vec.downstream_models.lobe.lobe` class for best performance and convenience.
- `geom2vec.pretrain` contains dataset classes and training scripts for pretraining the GNNs 
in case users want to train their own models.
- `geom2vec.representation_models` contains the main classes various GNN architectures 
that can be used for representation learning.

## Tutorials


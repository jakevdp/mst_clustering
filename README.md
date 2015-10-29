# Minimum Spanning Tree Clustering

This package implements a simple scikit-learn style estimator for clustering
with a minimum spanning tree.

## Example

For an explanation of the algorithm and an example of it in action, see the [MST Clustering Notebook](http://nbviewer.ipython.org/github/jakevdp/mst_clustering/blob/master/MSTClustering.ipynb).

## Installation & Requirements

Requirements:

- [numpy](http://numpy.org)
- [scipy](http://scipy.org)
- [scikit-learn](http://scikit-learn.org)

I'd recommend installing these requirements with [Anaconda](https://www.continuum.io/downloads) or [miniconda](http://conda.pydata.org/miniconda.html). Once this is installed, you can type

```
$ conda update conda
$ conda install numpy scipy scikit-learn
```

This package is still unreleased; you can install it from source by running

```
$ python setup.py install
```
# Minimum Spanning Tree Clustering

[![build status](http://img.shields.io/travis/jakevdp/mst_clustering/master.svg?style=flat)](https://travis-ci.org/jakevdp/mst_clustering)
[![version status](http://img.shields.io/pypi/v/mst_clustering.svg?style=flat)](https://pypi.python.org/pypi/mst_clustering)
[![license](http://img.shields.io/badge/license-BSD-blue.svg?style=flat)](https://github.com/jakevdp/mst_clustering/blob/master/LICENSE)

This package implements a simple scikit-learn style estimator for clustering
with a minimum spanning tree.

## Example

For an explanation of the algorithm and an example of it in action, see the [MST Clustering Notebook](http://nbviewer.jupyter.org/github/jakevdp/mst_clustering/blob/master/MSTClustering.ipynb).

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

You can install the current release of this package from the Python Package Index using ``pip``:

```
$ conda install pip  # if using conda
$ pip install mst_clustering
```

To install from source, you can run

```
$ python setup.py install
```

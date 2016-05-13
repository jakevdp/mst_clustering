# Minimum Spanning Tree Clustering

[![build status](http://img.shields.io/travis/jakevdp/mst_clustering/master.svg?style=flat)](https://travis-ci.org/jakevdp/mst_clustering)
[![version status](http://img.shields.io/pypi/v/mst_clustering.svg?style=flat)](https://pypi.python.org/pypi/mst_clustering)
[![license](http://img.shields.io/badge/license-BSD-blue.svg?style=flat)](https://github.com/jakevdp/mst_clustering/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.50995.svg)](http://dx.doi.org/10.5281/zenodo.50995)


This package implements a simple scikit-learn style estimator for clustering
with a minimum spanning tree.

## Motivation

Automated clustering can be an important means of identifying structure in data,
but many of the more popular clustering algorithms do not perform well in the
presence of background noise. The clustering algorithm implemented here, based
on a trimmed Euclidean Minimum Spanning Tree, can be useful in this case.

## Example

The API of the ``mst_clustering`` code is designed for compatibility with
the [scikit-learn](http://scikit-learn.org) project.

```python
from mst_clustering import MSTClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# create some data with four clusters
X, y = make_blobs(200, centers=4, random_state=42)

# predict the labels with the MST algorithm
model = MSTClustering(cutoff_scale=2)
labels = model.fit_predict(X)

# plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow');
```

![Simple Clustering Plot](https://raw.githubusercontent.com/jakevdp/mst_clustering/master/images/SimpleClustering.png)

For a detailed explanation of the algorithm and a more interesting example of it in action, see the [MST Clustering Notebook](http://nbviewer.jupyter.org/github/jakevdp/mst_clustering/blob/master/MSTClustering.ipynb).

## Installation & Requirements

The ``mst_clustering`` package itself is fairly lightweight. It is tested on
Python 2.7 and 3.4-3.5, and depends on the following packages:

- [numpy](http://numpy.org)
- [scipy](http://scipy.org)
- [scikit-learn](http://scikit-learn.org)

Using the cross-platform [conda](http://conda.pydata.org/miniconda.html)
package manager, these requirements can be installed as follows:

```
$ conda install numpy scipy scikit-learn
```

Finally, the current release of ``mst_clustering`` can be installed using ``pip``:
```
$ conda install pip  # if using conda
$ pip install mst_clustering
```

To install ``mst_clustering`` from source, first download the source repository and then run
```
$ python setup.py install
```

## Contributing & Reporting Issues
Bug reports, questions, suggestions, and contributions are welcome.
For these, please make use the
[Issues](https://github.com/jakevdp/mst_clustering/issues)
or [Pull Requests](https://github.com/jakevdp/mst_clustering/pulls)
associated with this repository.

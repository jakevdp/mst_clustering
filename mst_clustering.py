"""
Minimum Spanning Tree Clustering
"""
from __future__ import division

import numpy as np

from scipy import sparse
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.utils import check_array

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import kneighbors_graph



class MSTClustering(BaseEstimator, ClusterMixin):
    """Minimum Spanning Tree Clustering

    Parameters
    ----------
    cutoff : float, int, optional
        either the number of edges to cut (if cutoff >= 1) or the fraction of
        edges to cut (if 0 < cutoff < 1). See also the ``cutoff_scale``
        parameter.
    cutoff_scale : float, optional
        minimum size of edges. All edges larger than cutoff_scale will be
        removed (see also ``cutoff`` parameter).
    min_cluster_size : int (default 1)
        minimum number of points per cluster. Points belonging to smaller
        clusters will be assigned to the background.
        all clusters will be kept.
    n_neighbors : int, optional (default 20)
        number of neighbors of each point used for approximate Euclidean
        minimum spanning tree (MST) algorithm.  See Notes below.

    Attributes
    ----------
    full_tree_ : sparse array, shape (n_samples, n_samples)
        Full minimum spanning tree over the fit data
    T_trunc_ : sparse array, shape (n_samples, n_samples)
        Non-connected graph over the final clusters
    labels_: array, length n_samples
        Labels of each point

    Notes
    -----
    This routine uses an approximate Euclidean minimum spanning tree (MST)
    to perform hierarchical clustering.  A true Euclidean minimum spanning
    tree naively costs O[N^3].  Graph traversal algorithms only help so much,
    because all N^2 edges must be used as candidates.  In this approximate
    algorithm, we use k << N edges from each point, so that the cost is only
    O[Nk log(Nk)]. For k = N, the approximation is exact; in practice for
    well-behaved data sets, the result is exact for k << N.
    """
    def __init__(self, cutoff=None, cutoff_scale=None,
                 min_cluster_size=1, n_neighbors=20,
                 metric='euclidean', metric_params=None):
        self.cutoff = cutoff
        self.cutoff_scale = cutoff_scale
        self.min_cluster_size = min_cluster_size
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, X, y=None):
        """Fit the clustering model

        Parameters
        ----------
        X : array_like
            the data to be clustered: shape = [n_samples, n_features]
        """
        X = check_array(X)

        # validate n_neighbors
        n_neighbors = min(self.n_neighbors, X.shape[0] - 1)

        # generate a sparse graph using the k nearest neighbors of each point
        G = kneighbors_graph(X, n_neighbors=n_neighbors,
                             mode='distance',
                             metric=self.metric,
                             metric_params=self.metric_params)

        # Compute the minimum spanning tree of this graph
        self.full_tree_ = minimum_spanning_tree(G, overwrite=True)

        # Determine cutoff scale from ``cutoff`` and ``cutoff_scale``
        if self.cutoff is None:
            cutoff_frac = 0
        elif self.cutoff >= 1:
            cutoff_frac = max(1.0, self.cutoff / X.shape[0])
        elif 0 <= self.cutoff < 1:
            cutoff_frac = self.cutoff
        else:
            raise ValueError('self.cutoff must be positive, not {0}'
                             ''.format(self.cutoff))

        cutoff_value = np.percentile(self.full_tree_.data,
                                     100 * (1 - cutoff_frac))

        if self.cutoff_scale is not None:
            cutoff_value = min(cutoff_value, self.cutoff_scale)

        # Trim the tree
        T_trunc = self.full_tree_.copy()
        mask = T_trunc.data > cutoff_value

        # Eliminate zeros from T_trunc for efficiency.
        # We want to do this:
        #    T_trunc.data[mask] = 0
        #    T_trunc.eliminate_zeros()
        # but there could be explicit zeros in our data!
        # So we call eliminate_zeros() with a stand-in data array,
        # then replace the data when we're finished.
        original_data = T_trunc.data
        T_trunc.data = np.arange(1, len(T_trunc.data) + 1)
        T_trunc.data[mask] = 0
        T_trunc.eliminate_zeros()
        T_trunc.data = original_data[T_trunc.data.astype(int) - 1]

        # find connected components
        n_components, labels = connected_components(T_trunc, directed=False)

        # remove clusters with fewer than min_cluster_size
        counts = np.bincount(labels)
        to_remove = np.where(counts < self.min_cluster_size)[0]

        for i in to_remove:
            labels[labels == i] = -1

        if len(to_remove) > 0:
            _, labels = np.unique(labels, return_inverse=True)
            labels -= 1  # keep -1 labels the same

        # update T_trunc by eliminating non-clusters
        # operationally, this means zeroing-out rows & columns where
        # the label is negative.
        I = sparse.eye(len(labels))
        I.data[0, labels < 0] = 0

        # we could just do this:
        #   T_trunc = I * T_trunc * I
        # but we want to be able to eliminate the zeros, so we use the same
        # trick as above
        original_data = T_trunc.data
        T_trunc.data = np.arange(1, len(T_trunc.data) + 1)
        T_trunc = I * T_trunc * I
        T_trunc.eliminate_zeros()
        T_trunc.data = original_data[T_trunc.data.astype(int) - 1]

        self.labels_ = labels
        self.cluster_graph_ = T_trunc
        return self

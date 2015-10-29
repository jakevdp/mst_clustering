import numpy as np
from numpy.testing import assert_equal, assert_allclose

from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

from .. import MSTClustering


def test_simple_blobs():
    X, y = make_blobs(100, random_state=42)

    def _check_params(kwds):
        y_pred = MSTClustering(n_neighbors=100, **kwds).fit_predict(X)
        assert_equal(len(np.unique(y_pred)), 3)
        assert_allclose([np.std(y[y == i]) for i in range(3)], 0)

    for kwds in [dict(cutoff=2), dict(cutoff=0.02), dict(cutoff_scale=2.5)]:
        yield _check_params, kwds


def test_n_clusters():
    N = 30
    rng = np.random.RandomState(42)
    X = rng.rand(N, 3)

    def _check_n(n):
        y_pred = MSTClustering(cutoff=n).fit_predict(X)
        assert_equal(len(np.unique(y_pred)), n + 1)

    for n in range(30):
        yield _check_n, n


def test_explicit_zeros():
    N = 30
    rng = np.random.RandomState(42)
    X = rng.rand(N, 3)
    X[-5:] = X[:5]

    def _check_n(n):
        y_pred = MSTClustering(cutoff=n).fit_predict(X)
        assert_equal(len(np.unique(y_pred)), n + 1)

    for n in range(30):
        yield _check_n, n


def test_precomputed_metric():
    N = 30
    n_neighbors = 10
    rng = np.random.RandomState(42)
    X = rng.rand(N, 3)

    G_sparse = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance')
    G_dense = G_sparse.toarray()
    G_dense[G_dense == 0] = np.nan

    kwds = dict(cutoff=0.1)
    y1 = MSTClustering(n_neighbors=n_neighbors, **kwds).fit_predict(X)
    y2 = MSTClustering(metric='precomputed', **kwds).fit_predict(G_sparse)
    y3 = MSTClustering(metric='precomputed', **kwds).fit_predict(G_dense)

    assert_allclose(y1, y2)
    assert_allclose(y2, y3)


def test_precomputed_metric_with_duplicates():
    N = 30
    n_neighbors = N - 1
    rng = np.random.RandomState(42)

    # make data with duplicate points
    X = rng.rand(N, 3)
    X[-5:] = X[:5]

    # compute sparse distances
    G_sparse = kneighbors_graph(X, n_neighbors=n_neighbors,
                                mode='distance')

    # compute dense distances
    G_dense = pairwise_distances(X, X)

    kwds = dict(cutoff=0.1)
    y1 = MSTClustering(n_neighbors=n_neighbors, **kwds).fit_predict(X)
    y2 = MSTClustering(metric='precomputed', **kwds).fit_predict(G_sparse)
    y3 = MSTClustering(metric='precomputed', **kwds).fit_predict(G_dense)

    assert_allclose(y1, y2)
    assert_allclose(y2, y3)

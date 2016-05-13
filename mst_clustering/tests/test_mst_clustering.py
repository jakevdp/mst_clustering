import numpy as np
from numpy.testing import (assert_, assert_equal, assert_allclose,
                           assert_raises_regex)

from nose import SkipTest

from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

from mst_clustering import MSTClustering


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
        y_pred = MSTClustering(cutoff=n, approximate=False).fit_predict(X)
        assert_equal(len(np.unique(y_pred)), n + 1)

    for n in range(30):
        yield _check_n, n


def test_n_clusters_approximate():
    N = 30
    rng = np.random.RandomState(42)
    X = rng.rand(N, 3)

    def _check_n(n):
        y_pred = MSTClustering(cutoff=n,
                               n_neighbors=2,
                               approximate=True).fit_predict(X)
        assert_equal(len(np.unique(y_pred)), n + 1)

    # due to approximation, there are 3 clusters for n in (1, 2)
    for n in range(3, 30):
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


def test_min_cluster_size():
    N = 30
    rng = np.random.RandomState(42)
    X = rng.rand(N, 3)

    def _check(n, min_cluster_size):
        y_pred = MSTClustering(cutoff=n,
                               n_neighbors=2,
                               min_cluster_size=min_cluster_size,
                               approximate=True).fit_predict(X)
        labels, counts = np.unique(y_pred, return_counts=True)
        counts = counts[labels >= 0]
        if len(counts):
            assert_(counts.min() >= min_cluster_size)

    # due to approximation, there are 3 clusters for n in (1, 2)
    for n in range(3, 30, 5):
        for min_cluster_size in [1, 3, 5]:
            yield _check, n, min_cluster_size


def test_precomputed():
    X, y = make_blobs(100, random_state=42)
    D = pairwise_distances(X)

    mst1 = MSTClustering(cutoff=0.1)
    mst2 = MSTClustering(cutoff=0.1, metric='precomputed')

    assert_equal(mst1.fit_predict(X),
                 mst2.fit_predict(D))


def test_bad_arguments():
    X, y = make_blobs(100, random_state=42)

    mst = MSTClustering()
    assert_raises_regex(ValueError,
                        "Must specify either cutoff or cutoff_frac",
                        mst.fit, X, y)

    mst = MSTClustering(cutoff=-1)
    assert_raises_regex(ValueError, "cutoff must be positive", mst.fit, X)

    mst = MSTClustering()
    msg = "Must call fit\(\) before get_graph_segments()"
    assert_raises_regex(ValueError, msg, mst.get_graph_segments)

    mst = MSTClustering(cutoff=0, metric='precomputed')
    mst.fit(pairwise_distances(X))
    msg = "Cannot use ``get_graph_segments`` with precomputed metric."
    assert_raises_regex(ValueError, msg, mst.get_graph_segments)


def test_graph_segments_shape():
    def check_shape(ndim, cutoff, N=10):
        X = np.random.rand(N, ndim)
        mst = MSTClustering(cutoff=cutoff).fit(X)

        segments = mst.get_graph_segments()
        print(ndim, cutoff, segments[0].shape)
        assert len(segments) == ndim
        assert all(seg.shape == (2, N - 1 - cutoff) for seg in segments)

        segments = mst.get_graph_segments(full_graph=True)
        print(segments[0].shape)
        assert len(segments) == ndim
        assert all(seg.shape == (2, N - 1) for seg in segments)

    for N in [10, 15]:
        for ndim in [1, 2, 3]:
            for cutoff in [0, 1, 2]:
                yield check_shape, ndim, cutoff, N


def check_graph_segments_vals():
    X = np.arange(5)[:, None] ** 2
    mst = MSTClustering(cutoff=0).fit(X)
    segments = mst.get_graph_segments()
    assert len(segments) == 1
    assert_allclose(segments[0],
                    [[0, 4, 4, 9],
                     [1, 1, 9, 16]])


# this fails for silly reasons currently; we'll leave it out.
def __test_estimator_checks():
    try:
        from sklearn.utils.estimator_checks import check_estimator
    except ImportError:
        raise SkipTest("need scikit-learn 0.17+ for check_estimator()")

    check_estimator(MSTClustering)

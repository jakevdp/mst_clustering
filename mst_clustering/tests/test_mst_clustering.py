import numpy as np
from numpy.testing import assert_equal, assert_allclose

from sklearn.datasets import make_blobs

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

    for n in range(25):
        yield _check_n, n
    
    

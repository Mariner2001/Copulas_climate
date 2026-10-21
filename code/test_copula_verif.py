"""Sanity tests for copula_verif.py (run: python -m pytest -q test_copula_verif.py)."""
import numpy as np
from scipy.stats import norm, kstest
import copula_verif as cv


def test_transplant_marginals_and_ranks():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 7))
    B = rng.standard_t(3, size=(50, 7)) @ rng.normal(size=(7, 7))
    T = cv.transplant(A, B)
    assert np.allclose(np.sort(T, 0), np.sort(A, 0))          # same marginal multisets
    assert (cv.ranks(T) == cv.ranks(B)).all()                   # same ranks = same copula


def test_empirical_cdf_identity():
    # H_hat(y) = C_hat_B(F_hat_A1(y1), ..., F_hat_Ad(yd)) exactly (no ties)
    rng = np.random.default_rng(1)
    M, d = 40, 3
    A, B = rng.normal(size=(M, d)), rng.normal(size=(M, d)) @ np.array([[1, .8, .3], [0, .6, .2], [0, 0, 1.]])
    T = cv.transplant(A, B)
    rB = cv.ranks(B) + 1                                        # 1..M
    for _ in range(200):
        y = rng.normal(size=d) * 1.5
        H = np.mean(np.all(T <= y, axis=1))
        Fa = np.array([np.sum(A[:, j] <= y[j]) for j in range(d)])   # M * F_hat_Aj
        C = np.mean(np.all(rB <= Fa[None, :], axis=1))
        assert np.isclose(H, C)


def test_crps_fair_bruteforce():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(20, 5)), rng.normal(size=5)
    M = X.shape[0]
    bf = np.mean(np.abs(X - y), 0) - np.abs(X[:, None, :] - X[None, :, :]).sum((0, 1)) / (2 * M * (M - 1))
    assert np.allclose(cv.crps_fair(X, y), bf)


def test_coppit_uniform_for_ideal_forecast():
    rng = np.random.default_rng(3)
    R = np.array([[1, .7], [.7, 1]])
    L = np.linalg.cholesky(R)
    u = []
    for _ in range(3000):
        ens = rng.normal(size=(50, 2)) @ L.T
        y = rng.normal(size=2) @ L.T
        u.append(cv.coppit(ens, y, rng))
    assert kstest(u, "uniform").pvalue > 0.01


def test_coppit_pairs_matches_scalar_distribution():
    rng = np.random.default_rng(4)
    R = np.array([[1, .5], [.5, 1]]); L = np.linalg.cholesky(R)
    u1, u2 = [], []
    for _ in range(2000):
        ens = rng.normal(size=(50, 2)) @ L.T
        y = rng.normal(size=2) @ L.T
        u1.append(cv.coppit(ens, y, rng))
        u2.append(cv.coppit_pairs(ens, y, np.array([[0, 1]]), rng)[0])
    assert kstest(u1, "uniform").pvalue > 0.01 and kstest(u2, "uniform").pvalue > 0.01


def test_coppit_detects_underdependence():
    rng = np.random.default_rng(7)
    Lt = np.linalg.cholesky(np.array([[1, .9], [.9, 1]]))
    u = []
    for _ in range(2000):
        ens = rng.normal(size=(50, 2))            # independence copula, correct margins
        y = rng.normal(size=2) @ Lt.T             # strongly dependent truth
        u.append(cv.coppit(ens, y, rng))
    assert kstest(u, "uniform").pvalue < 1e-6


def test_shapley_exact():
    rng = np.random.default_rng(5)
    s = rng.random((4, 100))
    out = cv.shapley_2x2(*s)
    assert np.allclose(out["total"], out["marg"] + out["cop"])


def test_energy_score_fair():
    rng = np.random.default_rng(6)
    X, y = rng.normal(size=(15, 4)), rng.normal(size=4)
    M = 15
    bf = np.mean(np.linalg.norm(X - y, axis=1)) - np.linalg.norm(X[:, None] - X[None], axis=2).sum() / (2 * M * (M - 1))
    assert np.isclose(cv.energy_score_fair(X, y), bf)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)

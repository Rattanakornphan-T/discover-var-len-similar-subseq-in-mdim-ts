import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def _squared_euclidean_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # x: (M,), y: (N,)
    M = x.shape[0]
    N = y.shape[0]
    out = np.empty((M, N), dtype=np.float64)
    for i in prange(M):
        xi = x[i]
        for j in range(N):
            d = xi - y[j]
            out[i, j] = d * d
    return out

@njit(cache=True, fastmath=True)
def _squared_euclidean_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # x: (M,D), y:(N,D)
    M, D = x.shape
    N = y.shape[0]
    out = np.empty((M, N), dtype=np.float64)
    for i in prange(M):
        for j in range(N):
            s = 0.0
            for k in range(D):
                d = x[i, k] - y[j, k]
                s += d * d
            out[i, j] = s
    return out

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pointwise Euclidean distance between x and y (broadcasted)."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("x and y must have at least 2 dimensions")
    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions")
    return np.sqrt(np.sum((x - y) ** 2, axis=-1))

def squared_euclidean_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fast squared-Euclidean cost matrix between two time series.

    Parameters
    ----------
    x : np.ndarray
        Shape (M,) or (M, D)
    y : np.ndarray
        Shape (N,) or (N, D)

    Returns
    -------
    np.ndarray
        Squared Euclidean distance matrix, shape (M, N), dtype float64.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        return _squared_euclidean_1d(x, y)
    if x.ndim == 2:
        return _squared_euclidean_2d(x, y)
    raise ValueError("x must be 1D or 2D")

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("x and y must have at least 2 dimensions")
    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions")
    return np.sum(np.abs(x - y), axis=-1)

import numpy as np

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


# ----------------------------
# NumPy implementations
# ----------------------------
def znorm_numpy(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Z-normalize along time axis (axis=0).
    - x shape (t,)  -> returns (t,)
    - x shape (t,n) -> returns (t,n) (each column normalized separately)
    Constant series/columns become all zeros.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x_f = x.astype(np.float64, copy=False)
        mu = x_f.mean()
        sigma = x_f.std()  # ddof=0
        if not np.isfinite(sigma) or sigma <= eps:
            return np.zeros_like(x_f)
        return (x_f - mu) / sigma

    if x.ndim == 2:
        x_f = x.astype(np.float64, copy=False)
        mu = x_f.mean(axis=0)
        sigma = x_f.std(axis=0)
        # avoid divide-by-zero: constant cols -> zeros
        sigma_safe = np.where((~np.isfinite(sigma)) | (sigma <= eps), 1.0, sigma)
        y = (x_f - mu) / sigma_safe
        # set constant/invalid cols to 0 exactly
        bad = (~np.isfinite(sigma)) | (sigma <= eps)
        if np.any(bad):
            y[:, bad] = 0.0
        return y

    raise ValueError(f"Expected 1D (t,) or 2D (t,n); got shape {x.shape}")


# ----------------------------
# Numba implementations (fast)
# ----------------------------
if _HAVE_NUMBA:
    @njit(cache=True)
    def _znorm_1d_nb(x: np.ndarray, eps: float) -> np.ndarray:
        t = x.shape[0]
        y = np.empty(t, dtype=np.float64)

        # mean
        s = 0.0
        for i in range(t):
            s += x[i]
        mu = s / t

        # variance
        v = 0.0
        for i in range(t):
            d = x[i] - mu
            v += d * d
        v /= t
        sigma = np.sqrt(v)

        if (not np.isfinite(sigma)) or sigma <= eps:
            for i in range(t):
                y[i] = 0.0
            return y

        inv = 1.0 / sigma
        for i in range(t):
            y[i] = (x[i] - mu) * inv
        return y


    @njit(parallel=True, cache=True)
    def _znorm_2d_nb(x: np.ndarray, eps: float) -> np.ndarray:
        t, n = x.shape
        y = np.empty((t, n), dtype=np.float64)

        for j in prange(n):
            # mean for column j
            s = 0.0
            for i in range(t):
                s += x[i, j]
            mu = s / t

            # variance for column j
            v = 0.0
            for i in range(t):
                d = x[i, j] - mu
                v += d * d
            v /= t
            sigma = np.sqrt(v)

            if (not np.isfinite(sigma)) or sigma <= eps:
                for i in range(t):
                    y[i, j] = 0.0
            else:
                inv = 1.0 / sigma
                for i in range(t):
                    y[i, j] = (x[i, j] - mu) * inv

        return y


def znorm_numba(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Numba-accelerated z-normalization along time axis (axis=0).
    - x shape (t,)  -> returns (t,)
    - x shape (t,n) -> returns (t,n) (each column normalized separately)
    Output dtype is float64.
    """
    if not _HAVE_NUMBA:
        # fallback
        return znorm_numpy(x, eps=eps)

    x = np.asarray(x)
    if x.ndim == 1:
        x_f = np.ascontiguousarray(x.astype(np.float64, copy=False))
        return _znorm_1d_nb(x_f, float(eps))

    if x.ndim == 2:
        x_f = np.ascontiguousarray(x.astype(np.float64, copy=False))
        return _znorm_2d_nb(x_f, float(eps))

    raise ValueError(f"Expected 1D (t,) or 2D (t,n); got shape {x.shape}")


# ----------------------------
# Convenience wrapper
# ----------------------------
def znorm(x: np.ndarray, eps: float = 1e-12, engine: str = "numba") -> np.ndarray:
    """
    engine: "numba" or "numpy"
    """
    if engine == "numba":
        return znorm_numba(x, eps=eps)
    if engine == "numpy":
        return znorm_numpy(x, eps=eps)
    raise ValueError("engine must be 'numba' or 'numpy'")

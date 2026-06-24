import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def sum_pooling_1d(arr: np.ndarray, size: int) -> np.ndarray:
    n = arr.shape[0]
    out_n = n - size + 1
    out = np.empty(out_n, dtype=np.float64)
    # prefix sum
    ps = 0.0
    # warmup first window
    for i in range(size):
        ps += arr[i]
    out[0] = ps
    for i in range(size, n):
        ps += arr[i] - arr[i - size]
        out[i - size + 1] = ps
    return out

@njit(cache=True, fastmath=True)
def max_pooling_1d(arr: np.ndarray, size: int) -> np.ndarray:
    n = arr.shape[0]
    out_n = n - size + 1
    out = np.empty(out_n, dtype=np.float64)

    dq = np.empty(n, dtype=np.int32)
    head = 0
    tail = 0

    for i in range(n):
        ai = arr[i]
        while tail > head and arr[dq[tail - 1]] <= ai:
            tail -= 1
        dq[tail] = i
        tail += 1

        # remove out-of-window
        if dq[head] <= i - size:
            head += 1

        if i >= size - 1:
            out[i - size + 1] = arr[dq[head]]

    return out

@njit(cache=True, fastmath=True)
def sum_pooling(arr: np.ndarray, size0: int, size1: int) -> np.ndarray:
    s0, s1 = arr.shape
    out_s0 = s0 - size0 + 1
    out_s1 = s1 - size1 + 1

    # horizontal sums
    tmp = np.empty((s0, out_s1), dtype=np.float64)
    if size1 > 1:
        for i in prange(s0):
            tmp[i, :] = sum_pooling_1d(arr[i, :], size1)
    else:
        tmp[:, :] = arr

    # vertical sums
    out = np.empty((out_s0, out_s1), dtype=np.float64)
    if size0 > 1:
        for j in prange(out_s1):
            out[:, j] = sum_pooling_1d(tmp[:, j], size0)
    else:
        out[:, :] = tmp

    return out

@njit(cache=True, fastmath=True)
def max_pooling(arr: np.ndarray, size0: int, size1: int) -> np.ndarray:
    s0, s1 = arr.shape
    out_s0 = s0 - size0 + 1
    out_s1 = s1 - size1 + 1

    tmp = np.empty((s0, out_s1), dtype=np.float64)
    if size1 > 1:
        for i in prange(s0):
            tmp[i, :] = max_pooling_1d(arr[i, :], size1)
    else:
        tmp[:, :] = arr

    out = np.empty((out_s0, out_s1), dtype=np.float64)
    if size0 > 1:
        for j in prange(out_s1):
            out[:, j] = max_pooling_1d(tmp[:, j], size0)
    else:
        out[:, :] = tmp

    return out

@njit(cache=True, fastmath=True)
def region_max_pooling(arr: np.ndarray, size0: int, size1: int, region: np.ndarray) -> np.ndarray:
    s0, s1 = arr.shape
    out_s0 = s0 - size0 + 1
    out_s1 = s1 - size1 + 1
    out = np.empty((out_s0, out_s1), dtype=np.float64)

    for i in prange(out_s0):
        for j in range(out_s1):
            max_val = -np.inf
            for pi in range(size0):
                start = region[0, pi]
                end = region[1, pi]
                for pj in range(start, end):
                    val = arr[i + pi, j + pj]
                    if val > max_val:
                        max_val = val
            out[i, j] = max_val
    return out

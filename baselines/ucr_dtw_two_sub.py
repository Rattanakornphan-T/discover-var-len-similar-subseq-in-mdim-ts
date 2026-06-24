"""
UCR Suite DTW nearest-neighbor search (Python port of UCR_DTW.cpp).

Acceleration:
  - If NumPy + Numba are available, you can use a numba-jitted inner loop
    (engine='numba') for a large speedup.
  - If Numba isn't available, it falls back to the pure-Python implementation.
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from typing import Deque, List, Tuple

INF = 1e20  # matches the C++ code

# Optional acceleration
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    # Numba can fail to import in some environments due to dependency conflicts.
    from numba import njit  # type: ignore

    _NUMBA_OK = True
except Exception:  # pragma: no cover
    njit = None  # type: ignore
    _NUMBA_OK = False


def _dist(x: float, y: float) -> float:
    d = x - y
    return d * d


def z_normalize(x: List[float]) -> Tuple[List[float], float, float]:
    """Return z-normalized copy of x, along with mean and std used."""
    m = len(x)
    if m == 0:
        return [], 0.0, 1.0
    ex = sum(x)
    ex2 = sum(v * v for v in x)
    mean = ex / m
    var = ex2 / m - mean * mean
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0.0:
        std = 1e-12
    return [(v - mean) / std for v in x], mean, std


def lower_upper_lemire(t: List[float], length: int, r: int) -> Tuple[List[float], List[float]]:
    """Compute Lemire lower/upper envelopes for LB_Keogh (pure Python)."""
    if length <= 0:
        return [], []
    if r < 0:
        raise ValueError("r must be >= 0")

    u = [0.0] * length
    l = [0.0] * length

    du: Deque[int] = deque()
    dl: Deque[int] = deque()

    du.append(0)
    dl.append(0)

    for i in range(1, length):
        if i > r:
            u[i - r - 1] = t[du[0]]
            l[i - r - 1] = t[dl[0]]

        if t[i] > t[i - 1]:
            while du and t[i] > t[du[-1]]:
                du.pop()
        else:
            while dl and t[i] < t[dl[-1]]:
                dl.pop()

        du.append(i)
        dl.append(i)

        if i == (2 * r + 1) + du[0]:
            du.popleft()
        if i == (2 * r + 1) + dl[0]:
            dl.popleft()

    for i in range(length, length + r + 1):
        u[i - r - 1] = t[du[0]]
        l[i - r - 1] = t[dl[0]]

        if i - du[0] >= 2 * r + 1:
            du.popleft()
        if i - dl[0] >= 2 * r + 1:
            dl.popleft()

    return l, u


def lb_kim_hierarchy(
    t2m: List[float],
    q: List[float],
    j: int,
    m: int,
    mean: float,
    std: float,
    bsf: float = INF,
) -> float:
    """LB_Kim hierarchy (pure Python)."""
    x0 = (t2m[j] - mean) / std
    y0 = (t2m[j + m - 1] - mean) / std
    lb = _dist(x0, q[0]) + _dist(y0, q[m - 1])
    if lb >= bsf:
        return lb

    x1 = (t2m[j + 1] - mean) / std
    d = min(_dist(x1, q[0]), _dist(x0, q[1]))
    d = min(d, _dist(x1, q[1]))
    lb += d
    if lb >= bsf:
        return lb

    y1 = (t2m[j + m - 2] - mean) / std
    d = min(_dist(y1, q[m - 1]), _dist(y0, q[m - 2]))
    d = min(d, _dist(y1, q[m - 2]))
    lb += d
    if lb >= bsf:
        return lb

    x2 = (t2m[j + 2] - mean) / std
    d = min(_dist(x0, q[2]), _dist(x1, q[2]))
    d = min(d, _dist(x2, q[2]))
    d = min(d, _dist(x2, q[1]))
    d = min(d, _dist(x2, q[0]))
    lb += d
    if lb >= bsf:
        return lb

    y2 = (t2m[j + m - 3] - mean) / std
    d = min(_dist(y0, q[m - 3]), _dist(y1, q[m - 3]))
    d = min(d, _dist(y2, q[m - 3]))
    d = min(d, _dist(y2, q[m - 2]))
    d = min(d, _dist(y2, q[m - 1]))
    lb += d

    return lb


def lb_keogh_cumulative(
    order: List[int],
    t2m: List[float],
    uo: List[float],
    lo: List[float],
    cb: List[float],
    j: int,
    m: int,
    mean: float,
    std: float,
    best_so_far: float = INF,
) -> float:
    """LB_Keogh using the query envelope (pure Python)."""
    lb = 0.0
    for i in range(m):
        if lb >= best_so_far:
            break
        idx = order[i]
        x = (t2m[j + idx] - mean) / std
        d = 0.0
        if x > uo[i]:
            d = _dist(x, uo[i])
        elif x < lo[i]:
            d = _dist(x, lo[i])
        lb += d
        cb[idx] = d
    return lb


def lb_keogh_data_cumulative(
    order: List[int],
    tz: List[float],
    qo: List[float],
    cb: List[float],
    l_buff: List[float],
    u_buff: List[float],
    base: int,
    m: int,
    mean: float,
    std: float,
    best_so_far: float = INF,
) -> float:
    """LB_Keogh2 using the data envelope (pure Python)."""
    lb = 0.0
    for i in range(m):
        if lb >= best_so_far:
            break
        idx = order[i]
        uu = (u_buff[base + idx] - mean) / std
        ll = (l_buff[base + idx] - mean) / std

        d = 0.0
        qv = qo[i]
        if qv > uu:
            d = _dist(qv, uu)
        elif qv < ll:
            d = _dist(qv, ll)
        lb += d
        cb[idx] = d
    return lb


def dtw(A: List[float], B: List[float], cb: List[float], m: int, r: int, bsf: float = INF) -> float:
    """DTW with Sakoe-Chiba band and early abandoning (pure Python)."""
    band = 2 * r + 1
    cost = [INF] * band
    cost_prev = [INF] * band

    for i in range(m):
        k = max(0, r - i)
        min_cost = INF

        j_start = max(0, i - r)
        j_end = min(m - 1, i + r)

        for j in range(j_start, j_end + 1):
            if i == 0 and j == 0:
                cost[k] = _dist(A[0], B[0])
                min_cost = cost[k]
                k += 1
                continue

            y = cost[k - 1] if (j - 1 >= 0 and k - 1 >= 0) else INF
            x = cost_prev[k + 1] if (i - 1 >= 0 and (k + 1) < band) else INF
            z = cost_prev[k] if (i - 1 >= 0 and j - 1 >= 0) else INF

            cost[k] = min(x, y, z) + _dist(A[i], B[j])
            if cost[k] < min_cost:
                min_cost = cost[k]
            k += 1

        if i + r < m - 1:
            if min_cost + cb[i + r + 1] >= bsf:
                return min_cost + cb[i + r + 1]

        cost, cost_prev = cost_prev, cost
        for kk in range(band):
            cost[kk] = INF

    k_last = max(0, r - (m - 1)) + (m - 1 - max(0, (m - 1) - r))
    return cost_prev[k_last]

# -------------------- Numba-accelerated kernels (optional) --------------------

if _NUMBA_OK and np is not None:  # pragma: no cover

    @njit(cache=True)
    def _lower_upper_lemire_nb(t: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
        n = t.shape[0]
        u = np.empty(n, dtype=np.float64)
        l = np.empty(n, dtype=np.float64)
        du = np.empty(n, dtype=np.int64)
        dl = np.empty(n, dtype=np.int64)

        du_head = 0
        du_tail = 0
        dl_head = 0
        dl_tail = 0
        du[0] = 0
        dl[0] = 0

        for i in range(1, n):
            if i > r:
                u[i - r - 1] = t[du[du_head]]
                l[i - r - 1] = t[dl[dl_head]]

            if t[i] > t[i - 1]:
                while du_tail >= du_head and t[i] > t[du[du_tail]]:
                    du_tail -= 1
            else:
                while dl_tail >= dl_head and t[i] < t[dl[dl_tail]]:
                    dl_tail -= 1

            du_tail += 1
            du[du_tail] = i
            dl_tail += 1
            dl[dl_tail] = i

            if i == (2 * r + 1) + du[du_head]:
                du_head += 1
            if i == (2 * r + 1) + dl[dl_head]:
                dl_head += 1

        for i in range(n, n + r + 1):
            u[i - r - 1] = t[du[du_head]]
            l[i - r - 1] = t[dl[dl_head]]

            if i - du[du_head] >= 2 * r + 1:
                du_head += 1
            if i - dl[dl_head] >= 2 * r + 1:
                dl_head += 1

        return l, u

    @njit(cache=True)
    def _dist_nb(x: float, y: float) -> float:
        d = x - y
        return d * d

    @njit(cache=True)
    def _lb_kim_window_nb(buf: np.ndarray, I: int, q: np.ndarray, m: int, mean: float, std: float, bsf: float) -> float:
        x0 = (buf[I] - mean) / std
        y0 = (buf[I + m - 1] - mean) / std
        lb = _dist_nb(x0, q[0]) + _dist_nb(y0, q[m - 1])
        if lb >= bsf:
            return lb

        x1 = (buf[I + 1] - mean) / std
        d = _dist_nb(x1, q[0])
        v = _dist_nb(x0, q[1])
        if v < d:
            d = v
        v = _dist_nb(x1, q[1])
        if v < d:
            d = v
        lb += d
        if lb >= bsf:
            return lb

        y1 = (buf[I + m - 2] - mean) / std
        d = _dist_nb(y1, q[m - 1])
        v = _dist_nb(y0, q[m - 2])
        if v < d:
            d = v
        v = _dist_nb(y1, q[m - 2])
        if v < d:
            d = v
        lb += d
        if lb >= bsf:
            return lb

        x2 = (buf[I + 2] - mean) / std
        d = _dist_nb(x0, q[2])
        v = _dist_nb(x1, q[2])
        if v < d:
            d = v
        v = _dist_nb(x2, q[2])
        if v < d:
            d = v
        v = _dist_nb(x2, q[1])
        if v < d:
            d = v
        v = _dist_nb(x2, q[0])
        if v < d:
            d = v
        lb += d
        if lb >= bsf:
            return lb

        y2 = (buf[I + m - 3] - mean) / std
        d = _dist_nb(y0, q[m - 3])
        v = _dist_nb(y1, q[m - 3])
        if v < d:
            d = v
        v = _dist_nb(y2, q[m - 3])
        if v < d:
            d = v
        v = _dist_nb(y2, q[m - 2])
        if v < d:
            d = v
        v = _dist_nb(y2, q[m - 1])
        if v < d:
            d = v
        lb += d

        return lb

    @njit(cache=True)
    def _lb_keogh_window_nb(
        buf: np.ndarray,
        I: int,
        order: np.ndarray,
        uo: np.ndarray,
        lo: np.ndarray,
        cb_out: np.ndarray,
        m: int,
        mean: float,
        std: float,
        bsf: float,
    ) -> float:
        lb = 0.0
        for ii in range(m):
            if lb >= bsf:
                break
            idx = order[ii]
            x = (buf[I + idx] - mean) / std
            d = 0.0
            if x > uo[ii]:
                d = _dist_nb(x, uo[ii])
            elif x < lo[ii]:
                d = _dist_nb(x, lo[ii])
            lb += d
            cb_out[idx] = d
        return lb

    @njit(cache=True)
    def _lb_keogh2_window_nb(
        qo: np.ndarray,
        order: np.ndarray,
        l_buff: np.ndarray,
        u_buff: np.ndarray,
        I: int,
        cb_out: np.ndarray,
        m: int,
        mean: float,
        std: float,
        bsf: float,
    ) -> float:
        lb = 0.0
        for ii in range(m):
            if lb >= bsf:
                break
            idx = order[ii]
            uu = (u_buff[I + idx] - mean) / std
            ll = (l_buff[I + idx] - mean) / std
            qv = qo[ii]
            d = 0.0
            if qv > uu:
                d = _dist_nb(qv, uu)
            elif qv < ll:
                d = _dist_nb(qv, ll)
            lb += d
            cb_out[idx] = d
        return lb

    @njit(cache=True)
    def _dtw_nb(A: np.ndarray, B: np.ndarray, cb: np.ndarray, m: int, r: int, bsf: float) -> float:
        band = 2 * r + 1
        cost = np.empty(band, dtype=np.float64)
        cost_prev = np.empty(band, dtype=np.float64)
        for k in range(band):
            cost[k] = INF
            cost_prev[k] = INF

        for i in range(m):
            k = r - i
            if k < 0:
                k = 0
            min_cost = INF

            j_start = i - r
            if j_start < 0:
                j_start = 0
            j_end = i + r
            if j_end > m - 1:
                j_end = m - 1

            for j in range(j_start, j_end + 1):
                if i == 0 and j == 0:
                    cost[k] = _dist_nb(A[0], B[0])
                    if cost[k] < min_cost:
                        min_cost = cost[k]
                    k += 1
                    continue

                y = cost[k - 1] if (j - 1 >= 0 and k - 1 >= 0) else INF
                x = cost_prev[k + 1] if (i - 1 >= 0 and (k + 1) < band) else INF
                z = cost_prev[k] if (i - 1 >= 0 and j - 1 >= 0) else INF

                v = x
                if y < v:
                    v = y
                if z < v:
                    v = z
                cost[k] = v + _dist_nb(A[i], B[j])

                if cost[k] < min_cost:
                    min_cost = cost[k]
                k += 1

            if i + r < m - 1:
                if min_cost + cb[i + r + 1] >= bsf:
                    return min_cost + cb[i + r + 1]

            # swap + clear
            tmp = cost_prev
            cost_prev = cost
            cost = tmp
            for kk in range(band):
                cost[kk] = INF

        k_last = (r - (m - 1))
        if k_last < 0:
            k_last = 0
        j_start = (m - 1) - r
        if j_start < 0:
            j_start = 0
        k_last = k_last + (m - 1 - j_start)
        return cost_prev[k_last]

    @njit(cache=True)
    def _scan_chunk_nb(
        buf: np.ndarray,
        l_buff: np.ndarray,
        u_buff: np.ndarray,
        q: np.ndarray,
        qo: np.ndarray,
        uo: np.ndarray,
        lo: np.ndarray,
        order: np.ndarray,
        m: int,
        r: int,
        bsf: float,
    ) -> Tuple[float, int, int, int, int]:
        ep = buf.shape[0]
        maxI = ep - m
        if maxI < 0:
            return bsf, -1, 0, 0, 0

        tz = np.empty(m, dtype=np.float64)
        cb = np.empty(m, dtype=np.float64)
        cb1 = np.empty(m, dtype=np.float64)
        cb2 = np.empty(m, dtype=np.float64)
        for k in range(m):
            cb1[k] = 0.0
            cb2[k] = 0.0
            cb[k] = 0.0

        best_loc = -1
        kim = 0
        keogh = 0
        keogh2 = 0

        ex = 0.0
        ex2 = 0.0
        for k in range(m):
            v = buf[k]
            ex += v
            ex2 += v * v

        for I in range(maxI + 1):
            if I > 0:
                old = buf[I - 1]
                new = buf[I + m - 1]
                ex += new - old
                ex2 += new * new - old * old

            mean = ex / m
            var = ex2 / m - mean * mean
            std = math.sqrt(var) if var > 0.0 else 0.0
            if std == 0.0:
                std = 1e-12

            if m < 6:
                lb_kim = 0.0   # skip Kim hierarchy for tiny m
            else:
                lb_kim = _lb_kim_window_nb(buf, I, q, m, mean, std, bsf)
                
            if lb_kim < bsf:
                lb_k = _lb_keogh_window_nb(buf, I, order, uo, lo, cb1, m, mean, std, bsf)
                if lb_k < bsf:
                    for k in range(m):
                        tz[k] = (buf[I + k] - mean) / std

                    lb_k2 = _lb_keogh2_window_nb(qo, order, l_buff, u_buff, I, cb2, m, mean, std, bsf)
                    if lb_k2 < bsf:
                        src = cb1
                        if lb_k <= lb_k2:
                            src = cb2

                        cb[m - 1] = src[m - 1]
                        for k in range(m - 2, -1, -1):
                            cb[k] = cb[k + 1] + src[k]

                        dist2 = _dtw_nb(tz, q, cb, m, r, bsf)
                        if dist2 < bsf:
                            bsf = dist2
                            best_loc = I
                    else:
                        keogh2 += 1
                else:
                    keogh += 1
            else:
                kim += 1

        return bsf, best_loc, kim, keogh, keogh2

def _as_1d_float_array(x, name: str):
    """Convert input to a contiguous 1D float64 numpy array."""
    if np is None:
        raise ImportError("NumPy is required.")
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    # Reject NaNs/Infs early; the pruning + DTW math assumes finite values.
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf; please clean or impute before running DTW search.")
    return np.ascontiguousarray(arr)


def _z_norm_np(x):
    """Z-normalize a 1D float64 numpy array (ddof=0)."""
    mean = float(x.mean())
    std = float(x.std())
    if std == 0.0:
        std = 1e-12
    return (x - mean) / std


def _prepare_query_array(query: "np.ndarray", m: int, r: int, use_numba_env: bool):
    """Prepare (z-norm) query, envelopes, and ordering for LB_Keogh."""
    q = _z_norm_np(query)

    if use_numba_env and _NUMBA_OK and (np is not None):
        l_q, u_q = _lower_upper_lemire_nb(q, r)
    else:
        l_list, u_list = lower_upper_lemire(q.tolist(), m, r)
        l_q = np.asarray(l_list, dtype=np.float64)
        u_q = np.asarray(u_list, dtype=np.float64)

    order = np.argsort(np.abs(q))[::-1].astype(np.int64)
    qo = q[order]
    uo = u_q[order]
    lo = l_q[order]

    return q, order, qo, uo, lo


def _lb_kim_window_py(buf: list[float], I: int, q: list[float], m: int, mean: float, std: float, bsf: float) -> float:
    """LB_Kim for an in-memory buffer window starting at I."""
    # Requires m >= 3 for the 3-point checks.
    x0 = (buf[I] - mean) / std
    y0 = (buf[I + m - 1] - mean) / std
    lb = _dist(x0, q[0]) + _dist(y0, q[m - 1])
    if lb >= bsf:
        return lb

    x1 = (buf[I + 1] - mean) / std
    d = min(_dist(x1, q[0]), _dist(x0, q[1]), _dist(x1, q[1]))
    lb += d
    if lb >= bsf:
        return lb

    y1 = (buf[I + m - 2] - mean) / std
    d = min(_dist(y1, q[m - 1]), _dist(y0, q[m - 2]), _dist(y1, q[m - 2]))
    lb += d
    if lb >= bsf:
        return lb

    # 3-point look-ahead / look-behind
    x2 = (buf[I + 2] - mean) / std
    d = min(_dist(x0, q[2]), _dist(x1, q[2]), _dist(x2, q[2]), _dist(x2, q[1]), _dist(x2, q[0]))
    lb += d
    if lb >= bsf:
        return lb

    y2 = (buf[I + m - 3] - mean) / std
    d = min(_dist(y0, q[m - 3]), _dist(y1, q[m - 3]), _dist(y2, q[m - 3]), _dist(y2, q[m - 2]), _dist(y2, q[m - 1]))
    lb += d

    return lb


def _lb_keogh_window_py(
    buf: list[float],
    I: int,
    order: list[int],
    uo: list[float],
    lo: list[float],
    cb_out: list[float],
    m: int,
    mean: float,
    std: float,
    bsf: float,
) -> float:
    """LB_Keogh (query envelope) for an in-memory buffer window starting at I."""
    lb = 0.0
    for ii in range(m):
        if lb >= bsf:
            break
        idx = order[ii]
        x = (buf[I + idx] - mean) / std
        d = 0.0
        if x > uo[ii]:
            d = _dist(x, uo[ii])
        elif x < lo[ii]:
            d = _dist(x, lo[ii])
        lb += d
        cb_out[idx] = d
    return lb


def _scan_chunk_py(
    buf: list[float],
    l_buff: list[float],
    u_buff: list[float],
    q: list[float],
    qo: list[float],
    uo: list[float],
    lo: list[float],
    order: list[int],
    m: int,
    r: int,
    bsf: float,
):
    """Scan one in-memory chunk (pure Python). Returns (bsf2, best_I, kim, keogh, keogh2)."""
    ep = len(buf)
    maxI = ep - m
    if maxI < 0:
        return bsf, -1, 0, 0, 0

    tz = [0.0] * m
    cb = [0.0] * m
    cb1 = [0.0] * m
    cb2 = [0.0] * m

    best_I = -1
    kim = 0
    keogh = 0
    keogh2 = 0

    ex = 0.0
    ex2 = 0.0
    for k in range(m):
        v = buf[k]
        ex += v
        ex2 += v * v

    for I in range(maxI + 1):
        if I > 0:
            old = buf[I - 1]
            new = buf[I + m - 1]
            ex += new - old
            ex2 += new * new - old * old

        mean = ex / m
        var = ex2 / m - mean * mean
        std = math.sqrt(var) if var > 0.0 else 0.0
        if std == 0.0:
            std = 1e-12

        if m < 6:
            lb_kim = 0.0   # skip Kim hierarchy for tiny m
        else:
            lb_kim = _lb_kim_window_py(buf, I, q, m, mean, std, bsf)

        if lb_kim >= bsf:
            kim += 1
            continue

        lb_k = _lb_keogh_window_py(buf, I, order, uo, lo, cb1, m, mean, std, bsf)
        if lb_k >= bsf:
            keogh += 1
            continue

        for k in range(m):
            tz[k] = (buf[I + k] - mean) / std

        lb_k2 = lb_keogh_data_cumulative(order, tz, qo, cb2, l_buff, u_buff, I, m, mean, std, bsf)
        if lb_k2 >= bsf:
            keogh2 += 1
            continue

        # Use the tighter (larger) lower bound to build the cumulative bound for DTW early abandoning.
        src = cb1 if lb_k > lb_k2 else cb2
        cb[m - 1] = src[m - 1]
        for k in range(m - 2, -1, -1):
            cb[k] = cb[k + 1] + src[k]

        dist2 = dtw(tz, q, cb, m, r, bsf)
        if dist2 < bsf:
            bsf = dist2
            best_I = I

    return bsf, best_I, kim, keogh, keogh2


def ucr_dtw_search_array(
    data: "np.ndarray",
    query: "np.ndarray",
    R: float,
    *,
    epoch: int = 100000,
    show_progress: bool = False,
    engine: str = "auto",
    bsf: float = INF,
    return_squared: bool = False,
):
    """UCR Suite DTW subsequence search on in-memory arrays.

    Find best match of `query` inside `data`.

    Parameters
    ----------
    data:
        1D array; the longer series to be searched.
    query:
        1D array; the query subsequence (length m).
    R:
        Warping window: ratio (<=1) or absolute (>1). Same semantics as UCR_DTW.cpp.
    epoch:
        Chunk size for scanning. For small arrays you can set epoch=len(data).
    show_progress:
        If True, prints '.' periodically to stderr.
    engine:
        'auto' (default), 'numba', or 'python'.
    bsf:
        Initial best-so-far *squared* distance (used for pruning). Use INF for no prior bound.
    return_squared:
        If True, returns squared DTW distance instead of sqrt distance.

    Returns
    -------
    loc:
        0-based starting index of best match in `data` (or -1 if no match).
    dist:
        DTW distance (sqrt) by default, or squared distance if return_squared=True.
    data_scanned:
        Number of data points scanned (==len(data)).
    stats:
        Dict with pruning counts and timing.
    """
    if np is None:
        raise ImportError("NumPy is required for ucr_dtw_search_array.")

    data = _as_1d_float_array(data, "data")
    query = _as_1d_float_array(query, "query")

    m = int(query.size)
    if m < 3:
        raise ValueError("Query length m must be >= 3 (LB_Kim uses 3-point checks).")
    if data.size < m:
        # No subsequence of length m exists in data.
        dist_out = float("inf")
        if return_squared:
            dist_out = INF
        return -1, dist_out, int(data.size), {
            "engine": "python" if engine == "python" else ("numba" if (engine != "python" and _NUMBA_OK) else "python"),
            "r": 0,
            "kim_prunes": 0,
            "keogh_prunes": 0,
            "keogh2_prunes": 0,
            "elapsed_sec": 0.0,
        }

    if R <= 1:
        r = int(math.floor(R * m))
    else:
        r = int(math.floor(R))
    if r < 0:
        r = 0

    if engine not in {"auto", "numba", "python"}:
        raise ValueError("engine must be one of: auto, numba, python")

    use_numba = False
    if engine == "python":
        use_numba = False
    elif engine in {"auto", "numba"}:
        use_numba = _NUMBA_OK and (np is not None)

    # Prepare query structures
    q_np, order_np, qo_np, uo_np, lo_np = _prepare_query_array(query, m, r, use_numba_env=use_numba)

    # Decide epoch / step
    if epoch < m:
        epoch = m
    n = int(data.size)
    epoch = int(min(epoch, n))
    step = epoch - m + 1
    if step <= 0:
        step = 1

    kim = 0
    keogh = 0
    keogh2 = 0
    loc = -1

    t_start = time.perf_counter()

    if use_numba:
        # Ensure types
        q_np = np.ascontiguousarray(q_np, dtype=np.float64)
        order_np = np.ascontiguousarray(order_np, dtype=np.int64)
        qo_np = np.ascontiguousarray(qo_np, dtype=np.float64)
        uo_np = np.ascontiguousarray(uo_np, dtype=np.float64)
        lo_np = np.ascontiguousarray(lo_np, dtype=np.float64)

        dot_every = max(1, 1000000 // max(1, step))
        it = 0
        chunk_start = 0
        while True:
            chunk_end = chunk_start + epoch
            if chunk_end > n:
                chunk_end = n
            buf_view = data[chunk_start:chunk_end]
            ep = int(buf_view.size)
            if ep <= m - 1:
                break

            l_buff, u_buff = _lower_upper_lemire_nb(buf_view, r)

            if show_progress and (it % dot_every == 0):
                print(".", end="", file=sys.stderr, flush=True)

            bsf, best_I, k1, k2, k3 = _scan_chunk_nb(
                buf_view,
                l_buff,
                u_buff,
                q_np,
                qo_np,
                uo_np,
                lo_np,
                order_np,
                m,
                r,
                bsf,
            )
            kim += int(k1)
            keogh += int(k2)
            keogh2 += int(k3)
            if int(best_I) >= 0:
                loc = chunk_start + int(best_I)

            if chunk_end == n:
                break
            it += 1
            chunk_start += step

        engine_used = "numba"

    else:
        # Convert query structures to Python lists for fast indexing
        q = q_np.tolist()
        order = order_np.tolist()
        qo = qo_np.tolist()
        uo = uo_np.tolist()
        lo = lo_np.tolist()

        dot_every = max(1, 1000000 // max(1, step))
        it = 0
        chunk_start = 0
        while True:
            chunk_end = chunk_start + epoch
            if chunk_end > n:
                chunk_end = n
            buf = data[chunk_start:chunk_end].tolist()
            ep = len(buf)
            if ep <= m - 1:
                break

            l_buff, u_buff = lower_upper_lemire(buf, ep, r)

            if show_progress and (it % dot_every == 0):
                print(".", end="", file=sys.stderr, flush=True)

            bsf, best_I, k1, k2, k3 = _scan_chunk_py(
                buf,
                l_buff,
                u_buff,
                q,
                qo,
                uo,
                lo,
                order,
                m,
                r,
                bsf,
            )
            kim += int(k1)
            keogh += int(k2)
            keogh2 += int(k3)
            if int(best_I) >= 0:
                loc = chunk_start + int(best_I)

            if chunk_end == n:
                break
            it += 1
            chunk_start += step

        engine_used = "python"

    t_end = time.perf_counter()

    dist_out = bsf if return_squared else (math.sqrt(bsf) if bsf < INF else float("inf"))

    stats = {
        "engine": engine_used,
        "r": r,
        "kim_prunes": kim,
        "keogh_prunes": keogh,
        "keogh2_prunes": keogh2,
        "elapsed_sec": t_end - t_start,
    }

    return loc, dist_out, n, stats


def find_closest_subsequences(
    ts1: "np.ndarray",
    ts2: "np.ndarray",
    *,
    subseq_len: int | None = None,
    R: float = 1,
    epoch: int = 100000,
    engine: str = "auto",
):
    """Find the closest pair of same-length subsequences between two series.

    The shorter series is chosen as the source of queries; we slide a window of
    length `subseq_len` across it. Each window is searched in the longer series
    using the UCR Suite DTW subsequence search. Returns the closest pair of subsequences.

    Parameters
    ----------
    ts1, ts2:
        1D numpy arrays.
    subseq_len:
        Length of the subsequences to compare.
        - If None (default), uses min(len(ts1), len(ts2)), which degenerates to a
          single query window on the shorter series.
    R, epoch, engine:
        Passed through to ucr_dtw_search_array.

    Returns
    -------
    start1, start2, dist:
        start indices (0-based) in ts1 and ts2 for the best-matching subsequences,
        and the DTW distance between them (in z-normalized space).

    Notes
    -----
    - Distances are computed after z-normalizing the query and each candidate
      subsequence (matching UCR_DTW.cpp behavior).
    - If a non-degenerate sliding search is needed, pass an explicit `subseq_len`
      smaller than min(len(ts1), len(ts2)).
    """
    if np is None:
        raise ImportError("NumPy is required for find_closest_subsequences.")

    a = _as_1d_float_array(ts1, "ts1")
    b = _as_1d_float_array(ts2, "ts2")

    # Choose shorter series as the query source.
    swapped = False
    if a.size <= b.size:
        short = a
        long = b
    else:
        short = b
        long = a
        swapped = True

    if subseq_len is None:
        m = int(short.size)
    else:
        m = int(subseq_len)

    if m < 3:
        raise ValueError("subseq_len must be >= 3 (LB_Kim uses 3-point checks).")
    if m > int(short.size) or m > int(long.size):
        raise ValueError(
            f"subseq_len={m} is too large for the inputs: len(ts1)={int(a.size)}, len(ts2)={int(b.size)}."
        )

    best_dist2 = INF
    best_short_start = 0
    best_long_start = 0

    # Slide on the shorter series.
    n_short = int(short.size)
    for s in range(0, n_short - m + 1):
        q_win = short[s : s + m]
        loc, dist2, _, _stats = ucr_dtw_search_array(
            long,
            q_win,
            R,
            epoch=epoch,
            show_progress=False,
            engine=engine,
            bsf=best_dist2,
            return_squared=True,
        )
        if loc >= 0 and dist2 < best_dist2:
            best_dist2 = float(dist2)
            best_short_start = s
            best_long_start = int(loc)

    dist = math.sqrt(best_dist2) if best_dist2 < INF else float("inf")

    # Map back to original ts1/ts2 coordinates.
    if not swapped:
        return best_short_start, best_long_start, dist
    return best_long_start, best_short_start, dist


__all__ = [
    "find_closest_subsequences",
]

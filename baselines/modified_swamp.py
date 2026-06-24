"""
Python (NumPy + Numba) port of the reference MATLAB implementation that accompanies:

"Matrix Profile XXII - Exact Discovery of Time Series Motifs under DTW"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

# ----
# Numba import guard
# ----
#
# Some Python environments ship an incompatible `coverage` package that can
# break Numba import (Numba expects `coverage.types.Tracer`). We patch in a
# tiny stub when needed. This does *not* enable coverage; it just unblocks
# importing Numba.
try:  # pragma: no cover
    import coverage  # type: ignore

    if not hasattr(coverage, "types"):
        class _Types:  # noqa: D401
            pass

        coverage.types = _Types()  # type: ignore
    if not hasattr(coverage.types, "Tracer"):
        coverage.types.Tracer = object  # type: ignore
    if not hasattr(coverage.types, "TShouldTraceFn"):
        coverage.types.TShouldTraceFn = object  # type: ignore
except Exception:
    pass

try:
    from numba import njit
except Exception as e:  # pragma: no cover
    # Fallback: make the module importable without numba.
    def njit(*_args, **_kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        return _wrap

    _NUMBA_IMPORT_ERROR = e
else:
    _NUMBA_IMPORT_ERROR = None


@dataclass(frozen=True)
class MotifResult:
    """Top-1 DTW motif result."""

    distance: float
    idx1: int
    idx2: int
    best_so_far_history: Tuple[float, ...]
    pruning_counts: Tuple[int, ...]
    downsample_factors: Tuple[int, ...]


def _as_float1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.reshape(-1)
    return x


# --------------------------
# Rolling mean/std (ddof=0)
# --------------------------


@njit(cache=True)
def _rolling_mean_std_ddof0(ts: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    n = ts.size
    subcount = n - m + 1
    mu = np.empty(subcount, dtype=np.float64)
    sig = np.empty(subcount, dtype=np.float64)

    # prefix sums
    ps = np.empty(n + 1, dtype=np.float64)
    ps2 = np.empty(n + 1, dtype=np.float64)
    ps[0] = 0.0
    ps2[0] = 0.0
    for i in range(n):
        v = ts[i]
        ps[i + 1] = ps[i] + v
        ps2[i + 1] = ps2[i] + v * v

    inv_m = 1.0 / m
    for i in range(subcount):
        s = ps[i + m] - ps[i]
        s2 = ps2[i + m] - ps2[i]
        mean = s * inv_m
        # population variance (ddof=0)
        var = s2 * inv_m - mean * mean
        if var < 0.0:
            var = 0.0
        mu[i] = mean
        sig[i] = np.sqrt(var)
    return mu, sig


@njit(cache=True)
def _znorm_subseq(ts: np.ndarray, start: int, m: int, mu: np.ndarray, sig: np.ndarray, out: np.ndarray) -> bool:
    s = sig[start]
    mean = mu[start]

    if not np.isfinite(mean) or not np.isfinite(s) or s < 0.0:
        return False
    
    if s == 0.0:
        for k in range(m):
            out[k] = 0.0
        return True
    
    inv_s = 1.0 / s
    for k in range(m):
        out[k] = (ts[start + k] - mean) * inv_s
    return True


# --------------------------
# PAA
# --------------------------


@njit(cache=True)
def paa_updated(ts: np.ndarray, numcoeffs: int) -> np.ndarray:
    """Piecewise Aggregate Approximation (PAA)

    If N/numcoeffs is not integer, uses fractional carry logic.
    """
    n = ts.size
    if n < 1 or n < numcoeffs:
        raise ValueError("invalid input")

    per_section = n / numcoeffs
    # Integer ratio: PAA with equal-sized (integer) segments.
    if float(int(per_section)) == per_section:
        w = int(per_section)
        out = np.empty(numcoeffs, dtype=np.float64)
        idx = 0
        start = 0
        while idx < numcoeffs:
            s = 0.0
            for k in range(w):
                s += ts[start + k]
            out[idx] = s / w
            idx += 1
            start += w
        return out

    out = np.empty(numcoeffs, dtype=np.float64)
    paa_index = 0
    ts_index = 0
    carry = 1.0
    while ts_index < n and paa_index < numcoeffs:
        if carry > 0.0:
            p = carry * ts[ts_index]
            ts_index += 1
            post_carry = per_section - carry
            full = int(np.floor(post_carry))
            fractional = post_carry - full
        else:
            p = 0.0
            full = int(np.floor(per_section))
            fractional = per_section - full

        # add full points
        if ts_index + full > n:
            full = n - ts_index
        for _ in range(full):
            p += ts[ts_index]
            ts_index += 1

        if ts_index < n and fractional > 0.0:
            p += fractional * ts[ts_index]
            carry = 1.0 - fractional
        else:
            carry = 0.0

        out[paa_index] = p / per_section
        paa_index += 1

    return out


# --------------------------
# Sliding max/min for envelopes (Sakoe-Chiba band)
# --------------------------


@njit(cache=True)
def _trailing_window_max(x: np.ndarray, wlen: int, out: np.ndarray) -> None:
    n = x.size
    dq = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for i in range(n):
        xi = x[i]
        while tail > head:
            j = dq[tail - 1]
            if x[j] <= xi:
                tail -= 1
            else:
                break
        dq[tail] = i
        tail += 1
        # pop left if out of window
        left = i - wlen + 1
        if dq[head] < left:
            head += 1
        if i >= wlen - 1:
            out[i] = x[dq[head]]


@njit(cache=True)
def _trailing_window_min(x: np.ndarray, wlen: int, out: np.ndarray) -> None:
    n = x.size
    dq = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for i in range(n):
        xi = x[i]
        while tail > head:
            j = dq[tail - 1]
            if x[j] >= xi:
                tail -= 1
            else:
                break
        dq[tail] = i
        tail += 1
        left = i - wlen + 1
        if dq[head] < left:
            head += 1
        if i >= wlen - 1:
            out[i] = x[dq[head]]


@njit(cache=True)
def envelope_sakoe_chiba(q: np.ndarray, warpmax: int, U: np.ndarray, L: np.ndarray) -> None:
    """Compute U/L envelopes for LB_Keogh with Sakoe-Chiba radius warpmax."""
    m = q.size
    if warpmax <= 0:
        for i in range(m):
            U[i] = q[i]
            L[i] = q[i]
        return

    # If the radius exceeds sequence length, the envelope becomes global max/min.
    w = warpmax
    if w > m - 1:
        w = m - 1
    if w <= 0:
        for i in range(m):
            U[i] = q[i]
            L[i] = q[i]
        return
    wlen = 2 * w + 1

    # interior points can use fixed window
    tmp_max = np.empty(m, dtype=np.float64)
    tmp_min = np.empty(m, dtype=np.float64)
    for i in range(m):
        tmp_max[i] = 0.0
        tmp_min[i] = 0.0

    _trailing_window_max(q, wlen, tmp_max)
    _trailing_window_min(q, wlen, tmp_min)

    # edges: direct
    for i in range(w):
        lo = 0
        hi = min(m - 1, i + w)
        mx = q[lo]
        mn = q[lo]
        for j in range(lo + 1, hi + 1):
            v = q[j]
            if v > mx:
                mx = v
            if v < mn:
                mn = v
        U[i] = mx
        L[i] = mn

    for i in range(w, m - w):
        # window end index
        end = i + w
        U[i] = tmp_max[end]
        L[i] = tmp_min[end]

    for i in range(m - w, m):
        lo = max(0, i - w)
        hi = m - 1
        mx = q[lo]
        mn = q[lo]
        for j in range(lo + 1, hi + 1):
            v = q[j]
            if v > mx:
                mx = v
            if v < mn:
                mn = v
        U[i] = mx
        L[i] = mn


# --------------------------
# LB_Keogh and DTW (early abandon)
# --------------------------


@njit(cache=True)
def lb_keogh_ea_sq(x: np.ndarray, U: np.ndarray, L: np.ndarray, bsf_sq: float) -> float:
    dist = 0.0
    m = x.size
    for i in range(m):
        xi = x[i]
        ui = U[i]
        li = L[i]
        if xi > ui:
            d = xi - ui
            dist += d * d
        elif xi < li:
            d = xi - li
            dist += d * d
        if dist >= bsf_sq:
            return dist
    return dist


@njit(cache=True)
def lb_keogh_sq(x: np.ndarray, U: np.ndarray, L: np.ndarray) -> float:
    dist = 0.0
    m = x.size
    for i in range(m):
        xi = x[i]
        ui = U[i]
        li = L[i]
        if xi > ui:
            d = xi - ui
            dist += d * d
        elif xi < li:
            d = xi - li
            dist += d * d
    return dist


@njit(cache=True)
def dtw_distance_ea_sq(a: np.ndarray, b: np.ndarray, r: int, bsf_sq: float) -> float:
    """Band-constrained DTW with early abandoning. Returns squared distance."""
    m = a.size
    if r < 0:
        r = 0
    if r > m - 1:
        r = m - 1

    INF = 1e300
    prev = np.full(m + 1, INF, dtype=np.float64)
    curr = np.full(m + 1, INF, dtype=np.float64)
    prev[0] = 0.0

    for i in range(1, m + 1):
        # reset current row
        for j in range(m + 1):
            curr[j] = INF
        j_start = i - r
        if j_start < 1:
            j_start = 1
        j_end = i + r
        if j_end > m:
            j_end = m

        row_min = INF
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            d = ai - b[j - 1]
            cost = d * d
            # min of: left (curr[j-1]), up (prev[j]), diag (prev[j-1])
            v = cost + min(curr[j - 1], prev[j], prev[j - 1])
            curr[j] = v
            if v < row_min:
                row_min = v

        if row_min >= bsf_sq:
            return row_min

        tmp = prev
        prev = curr
        curr = tmp

    return prev[m]


# --------------------------
# Euclidean Matrix Profile
# --------------------------


@njit(cache=True)
def _mpx_v2_core(ts: np.ndarray, minlag: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    n = ts.size
    subcount = n - m + 1

    # Identify any subseq that contains non-finite values
    nanmap = np.zeros(subcount, dtype=np.bool_)
    for i in range(subcount):
        ok = True
        for k in range(m):
            v = ts[i + k]
            if not np.isfinite(v):
                ok = False
                break
        nanmap[i] = not ok

    # Replace NaN/Inf in ts with 0 for rolling stats
    ts2 = ts.copy()
    for i in range(n):
        if not np.isfinite(ts2[i]):
            ts2[i] = 0.0

    mu, _sig = _rolling_mean_std_ddof0(ts2, m)
    mus, _sigs = _rolling_mean_std_ddof0(ts2, m - 1)

    invnorm = np.empty(subcount, dtype=np.float64)
    for i in range(subcount):
        if nanmap[i]:
            invnorm[i] = np.nan
            continue
        mean = mu[i]
        ss = 0.0
        for k in range(m):
            d = ts2[i + k] - mean
            ss += d * d
        if ss <= 0.0 or not np.isfinite(ss):
            invnorm[i] = np.nan
        else:
            invnorm[i] = 1.0 / np.sqrt(ss)

    dr_bwd = np.empty(subcount, dtype=np.float64)
    dc_bwd = np.empty(subcount, dtype=np.float64)
    dr_fwd = np.empty(subcount, dtype=np.float64)
    dc_fwd = np.empty(subcount, dtype=np.float64)
    dr_bwd[0] = 0.0
    dc_bwd[0] = 0.0
    for i in range(1, subcount):
        dr_bwd[i] = ts2[i - 1] - mu[i - 1]
        dc_bwd[i] = ts2[i - 1] - mus[i]
    for i in range(subcount):
        dr_fwd[i] = ts2[i + m - 1] - mu[i]
        dc_fwd[i] = ts2[i + m - 1] - mus[i]

    mp_corr = np.full(subcount, -1.0, dtype=np.float64)
    mpi = np.full(subcount, -1, dtype=np.int64)
    for i in range(subcount):
        if nanmap[i] or not np.isfinite(invnorm[i]):
            mp_corr[i] = np.nan
            mpi[i] = -1

    # Diagonal iteration
    for diag in range(minlag, subcount):
        col0 = diag
        # initial covariance between subseq at row=0 and col=diag
        if col0 >= subcount:
            break
        if not (np.isfinite(invnorm[0]) and np.isfinite(invnorm[col0])):
            cov = 0.0
        else:
            cov = 0.0
            mean_r = mu[0]
            mean_c = mu[col0]
            for k in range(m):
                cov += (ts2[col0 + k] - mean_c) * (ts2[k] - mean_r)

        max_row = subcount - diag
        for row in range(max_row):
            col = diag + row
            if row > 0:
                cov = cov - dr_bwd[row] * dc_bwd[col] + dr_fwd[row] * dc_fwd[col]
            ir = invnorm[row]
            ic = invnorm[col]
            if not (np.isfinite(ir) and np.isfinite(ic)):
                continue
            corr = cov * ir * ic
            if corr > mp_corr[row]:
                mp_corr[row] = corr
                mpi[row] = col
            if corr > mp_corr[col]:
                mp_corr[col] = corr
                mpi[col] = row

    # Convert to distance
    mp = np.empty(subcount, dtype=np.float64)
    for i in range(subcount):
        c = mp_corr[i]
        if not np.isfinite(c):
            mp[i] = np.nan
        else:
            if c > 1.0:
                c = 1.0
            d = 2.0 * m * (1.0 - c)
            if d < 0.0:
                d = 0.0
            mp[i] = np.sqrt(d)
    return mp, mpi


def mpx_v2(ts: np.ndarray, minlag: int, subseqlen: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute z-normalized Euclidean self-join Matrix Profile"""
    ts = _as_float1d(ts)
    if subseqlen < 2 or ts.size - subseqlen + 1 < 2:
        raise ValueError("bad subsequence length")
    if minlag < 1:
        minlag = 1
    return _mpx_v2_core(ts, minlag, subseqlen)


# --------------------------
# LB_Keogh Matrix Profile
# --------------------------


@njit(cache=True)
def _lb_keogh_mp_core(ts: np.ndarray, m: int, minlag: int, warpmax: int, do_not_compute: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = ts.size
    subcount = n - m + 1

    mu, sig = _rolling_mean_std_ddof0(ts, m)

    # normalize all subseqs (subcount x m)
    subs = np.empty((subcount, m), dtype=np.float64)
    valid = np.ones(subcount, dtype=np.bool_)
    tmp = np.empty(m, dtype=np.float64)
    for i in range(subcount):
        if i < do_not_compute.size and do_not_compute[i]:
            valid[i] = False
            continue
        ok = _znorm_subseq(ts, i, m, mu, sig, tmp)
        if not ok:
            valid[i] = False
            continue
        for k in range(m):
            subs[i, k] = tmp[k]

    # envelopes
    U = np.empty((subcount, m), dtype=np.float64)
    L = np.empty((subcount, m), dtype=np.float64)
    tmpU = np.empty(m, dtype=np.float64)
    tmpL = np.empty(m, dtype=np.float64)
    for i in range(subcount):
        if not valid[i]:
            for k in range(m):
                U[i, k] = np.nan
                L[i, k] = np.nan
            continue
        envelope_sakoe_chiba(subs[i], warpmax, tmpU, tmpL)
        for k in range(m):
            U[i, k] = tmpU[k]
            L[i, k] = tmpL[k]

    mp = np.full(subcount, np.inf, dtype=np.float64)
    mpi = np.full(subcount, -1, dtype=np.int64)

    # Iterate only over valid indices to reduce overhead
    active = np.empty(subcount, dtype=np.int64)
    ac = 0
    for i in range(subcount):
        if valid[i]:
            active[ac] = i
            ac += 1

    for ai in range(ac):
        i = active[ai]
        # find first active j >= i+minlag
        for aj in range(ai, ac):
            j = active[aj]
            if j < i + minlag:
                continue
            # compute symmetric LBKeogh
            d1 = lb_keogh_sq(subs[j], U[i], L[i])
            d2 = lb_keogh_sq(subs[i], U[j], L[j])
            d = d1 if d1 > d2 else d2
            if d < mp[i]:
                mp[i] = d
                mpi[i] = j
            if d < mp[j]:
                mp[j] = d
                mpi[j] = i

    # sqrt
    for i in range(subcount):
        v = mp[i]
        if np.isfinite(v):
            if v < 0.0:
                v = 0.0
            mp[i] = np.sqrt(v)
    return mp, mpi


def lb_keogh_mp_updated(ts: np.ndarray, subseqlen: int, minlag: int, warpmax: int, do_not_compute: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ts = _as_float1d(ts)
    dnc = np.asarray(do_not_compute)
    if dnc.ndim != 1:
        dnc = dnc.reshape(-1)
    # treat nonzero as True
    dnc_bool = (dnc != 0)
    if subseqlen < 4 or minlag < 1:
        raise ValueError("bad parameters")
    if ts.size - subseqlen + 1 < 2:
        raise ValueError("time series too short")
    return _lb_keogh_mp_core(ts, subseqlen, minlag, warpmax, dnc_bool)


# --------------------------
# SWAMP top-1 DTW motif discovery
# --------------------------


@njit(cache=True)
def _argmin_ignore_nan(x: np.ndarray) -> int:
    best = 1e300
    idx = -1
    for i in range(x.size):
        v = x[i]
        if np.isfinite(v) and v < best:
            best = v
            idx = i
    return idx


@njit(cache=True)
def _set_inf_exclusion(x: np.ndarray, center: int, radius: int) -> None:
    n = x.size
    a = center - radius + 1
    if a < 0:
        a = 0
    b = center + radius
    if b > n - 1:
        b = n - 1
    for i in range(a, b + 1):
        x[i] = 1e300

def swamp_top1(
    ts: np.ndarray,
    subseqlen: int,
    warpmax: int,
    paa_start_k: int = 64,
    minlag: Optional[int] = None,
) -> MotifResult:
    """Find the top-1 DTW motif pair using a SWAMP-style search.

    Parameters
    ----------
    ts:
        1D time series.
    subseqlen:
        Motif length L.
    warpmax:
        Sakoe-Chiba DTW radius w.
    paa_start_k:
        Initial number of PAA coefficients k.
    minlag:
        Exclusion zone between motif occurrences. Default is L.
    """
    ts = _as_float1d(ts)
    n = ts.size
    if minlag is None:
        minlag = subseqlen
    if subseqlen < 4 or n < subseqlen * 2:
        raise ValueError("time series too short or bad subsequence length")

    subcount = n - subseqlen + 1

    # ---------- Phase 0: ED Matrix Profile (upper bound init) ----------
    mp_ed, _ = mpx_v2(ts, minlag, subseqlen)
    best_so_far = float(np.nanmin(mp_ed))
    first = int(np.nanargmin(mp_ed))
    mp_tmp = mp_ed.copy()
    _set_inf_exclusion(mp_tmp, first, subseqlen)
    second = int(np.nanargmin(mp_tmp))

    # DTW refine on ED motifs
    mu, sig = _rolling_mean_std_ddof0(ts, subseqlen)
    a = np.empty(subseqlen, dtype=np.float64)
    b = np.empty(subseqlen, dtype=np.float64)
    ok_a = _znorm_subseq(ts, first, subseqlen, mu, sig, a)
    ok_b = _znorm_subseq(ts, second, subseqlen, mu, sig, b)
    if ok_a and ok_b:
        d_sq = dtw_distance_ea_sq(a, b, warpmax, best_so_far * best_so_far)
        d = float(np.sqrt(d_sq))
        if d < best_so_far:
            best_so_far = d

    best_i, best_j = first, second

    # ---------- Phase I: hierarchy of downsampled LBKeogh MP ----------
    dnc = np.zeros(subcount, dtype=np.int8)  # prune flag per subseq start
    hist_bsf = [best_so_far]
    hist_pruned = [int(dnc.sum())]
    hist_D = []

    # If/when we reach full resolution (D==1), keep the final 1:1 LBKeogh MP and index
    mp_lbk_full = None
    lb_index_full = None

    k = int(paa_start_k)
    while (n / k) >= 0.75:
        D = int(round(n / k))
        hist_D.append(D)

        if D == 1:
            ds = ts
            dncs = dnc.astype(np.float64)
        else:
            ds = paa_updated(ts, k)
            dncs = paa_updated(dnc.astype(np.float64), k)

        # scaled subseqlen in downsampled space
        Ld = int(round(subseqlen * (ds.size / n)))
        if Ld < 4:
            k *= 2
            continue
        # Compute LBKeogh MP at this resolution
        mp_lbk, mpi_lbk = lb_keogh_mp_updated(ds, Ld, Ld, warpmax, dncs)
        # If we're at full resolution, store for Phase II
        if D == 1 and ds.size == n and Ld == subseqlen:
            mp_lbk_full = mp_lbk
            lb_index_full = mpi_lbk

        # stretch LBMP back to original subseq-start length
        rep = int(np.floor(n / ds.size))
        stretched = np.repeat(mp_lbk, rep)
        scale = np.sqrt(n / ds.size)
        stretched = stretched * scale
        if stretched.size < subcount:
            # pad with last value
            pad = np.full(subcount - stretched.size, stretched[-1] if stretched.size else np.inf)
            stretched = np.concatenate([stretched, pad])
        else:
            stretched = stretched[:subcount]

        # If everything is Inf, we're done
        if not np.isfinite(stretched).any():
            break

        # Try improving best-so-far using the minimum LB region
        sam1 = int(np.nanargmin(stretched))
        tmp2 = stretched.copy()
        lo = max(0, sam1 - subseqlen + 1)
        hi = min(subcount, sam1 + subseqlen)
        tmp2[lo:hi] = np.inf
        sam2 = int(np.nanargmin(tmp2))

        ok_a = _znorm_subseq(ts, sam1, subseqlen, mu, sig, a)
        ok_b = _znorm_subseq(ts, sam2, subseqlen, mu, sig, b)
        if ok_a and ok_b:
            d_sq = dtw_distance_ea_sq(a, b, warpmax, best_so_far * best_so_far)
            d = float(np.sqrt(d_sq))
            if d < best_so_far:
                best_so_far = d
                best_i, best_j = sam1, sam2

        # prune
        for i in range(subcount):
            if dnc[i] == 0 and stretched[i] > best_so_far:
                dnc[i] = 1

        hist_bsf.append(best_so_far)
        hist_pruned.append(int(dnc.sum()))
        k *= 2

    # If everything pruned, return
    if int(dnc.sum()) >= subcount - 1:
        return MotifResult(best_so_far, best_i, best_j, tuple(hist_bsf), tuple(hist_pruned), tuple(hist_D))

    # ---------- Phase II: ordered brute force with LBKim + LBKeogh + DTW ----------
    # For Phase II ordering we need LBKeogh 1:1 MP and neighbor index on full resolution.
    # Prefer reusing it from the final Phase I iteration; otherwise compute once.
    if mp_lbk_full is None or lb_index_full is None:
        mp_lbk_full, lb_index_full = lb_keogh_mp_updated(ts, subseqlen, subseqlen, warpmax, dnc)

    # order candidates by increasing lower bound (LBMP)
    order = np.argsort(mp_lbk_full)

    # Pre-allocate buffers
    U = np.empty(subseqlen, dtype=np.float64)
    L = np.empty(subseqlen, dtype=np.float64)

    dtw_calls = 0
    for candid_idx in order:
        if candid_idx < 0 or candid_idx >= subcount:
            continue
        if dnc[candid_idx] == 1:
            continue

        # neighbors: candid_idx + L ... end
        neigh = np.arange(candid_idx + subseqlen, subcount, dtype=np.int64)
        if neigh.size == 0:
            continue
        # swap first with LB nearest neighbor index if present
        nn = int(lb_index_full[candid_idx])
        if nn >= 0 and nn < subcount:
            # find nn in neigh and swap to front if found
            for t in range(neigh.size):
                if neigh[t] == nn:
                    tmp = neigh[0]
                    neigh[0] = neigh[t]
                    neigh[t] = tmp
                    break

        # build normalized candidate once
        ok_a = _znorm_subseq(ts, candid_idx, subseqlen, mu, sig, a)
        if not ok_a:
            dnc[candid_idx] = 1
            continue

        envelope_sakoe_chiba(a, warpmax, U, L)

        for j in neigh:
            if dnc[j] == 1:
                continue
            ok_b = _znorm_subseq(ts, j, subseqlen, mu, sig, b)
            if not ok_b:
                dnc[j] = 1
                continue

            # LB_Kim (fast, weak)
            lb_kim = abs(a[0] - b[0])
            t = abs(a[-1] - b[-1])
            if t > lb_kim:
                lb_kim = t
            if lb_kim >= best_so_far:
                continue

            # LB_Keogh (early abandon), using envelope of a
            lb_sq = lb_keogh_ea_sq(b, U, L, best_so_far * best_so_far)
            if np.sqrt(lb_sq) >= best_so_far:
                continue

            # DTW (early abandon)
            dtw_calls += 1
            d_sq = dtw_distance_ea_sq(a, b, warpmax, best_so_far * best_so_far)
            d = float(np.sqrt(d_sq))
            if d < best_so_far:
                best_so_far = d
                best_i = int(candid_idx)
                best_j = int(j)
                # prune candidates whose LB already exceeds bsf
                for ii in range(subcount):
                    if dnc[ii] == 0 and mp_lbk_full[ii] >= best_so_far:
                        dnc[ii] = 1

    return MotifResult(best_so_far, best_i, best_j, tuple(hist_bsf), tuple(hist_pruned), tuple(hist_D))



# --------------------------
# SWAMP-style top-1 DTW motif discovery for AB-join (two time series)
# --------------------------


@dataclass(frozen=True)
class ABMotifResult:
    """Top-1 DTW motif result between two time series (AB-join).

    This finds the single closest pair of subsequences (same length) where
    the first subsequence comes from A and the second from B.
    """

    distance: float
    idx_a: int
    idx_b: int
    best_so_far_history: Tuple[float, ...]
    pruned_a_counts: Tuple[int, ...]
    downsample_factors: Tuple[int, ...]


@njit(cache=True)
def _lb_keogh_ab_profile_ds_core(
    a: np.ndarray,
    b: np.ndarray,
    m: int,
    warpmax: int,
    dnc_a: np.ndarray,
    dnc_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsampled AB lower-bound 'profile' (A annotated by B) using LB_Keogh.

    Returns mp_a (min LB_Keogh distance to any window in B for each window in A)
    and mpi_a (argmin index in B).
    """
    n_a = a.size
    n_b = b.size
    sub_a = n_a - m + 1
    sub_b = n_b - m + 1

    mu_a, sig_a = _rolling_mean_std_ddof0(a, m)
    mu_b, sig_b = _rolling_mean_std_ddof0(b, m)

    # Precompute normalized B subsequences (reused for every A window)
    subs_b = np.empty((sub_b, m), dtype=np.float64)
    valid_b = np.ones(sub_b, dtype=np.bool_)
    tmp = np.empty(m, dtype=np.float64)
    for j in range(sub_b):
        if j < dnc_b.size and dnc_b[j]:
            valid_b[j] = False
            continue
        ok = _znorm_subseq(b, j, m, mu_b, sig_b, tmp)
        if not ok:
            valid_b[j] = False
            continue
        for k in range(m):
            subs_b[j, k] = tmp[k]

    mp_a = np.full(sub_a, np.inf, dtype=np.float64)
    mpi_a = np.full(sub_a, -1, dtype=np.int64)

    q = np.empty(m, dtype=np.float64)
    U = np.empty(m, dtype=np.float64)
    L = np.empty(m, dtype=np.float64)

    for i in range(sub_a):
        if i < dnc_a.size and dnc_a[i]:
            continue
        ok = _znorm_subseq(a, i, m, mu_a, sig_a, q)
        if not ok:
            continue
        envelope_sakoe_chiba(q, warpmax, U, L)

        best = 1e300
        bestj = -1
        for j in range(sub_b):
            if not valid_b[j]:
                continue
            d = lb_keogh_sq(subs_b[j], U, L)
            if d < best:
                best = d
                bestj = j
        if bestj >= 0:
            if best < 0.0:
                best = 0.0
            mp_a[i] = np.sqrt(best)
            mpi_a[i] = bestj

    return mp_a, mpi_a


def lb_keogh_ab_profile_ds(
    a: np.ndarray,
    b: np.ndarray,
    subseqlen: int,
    warpmax: int,
    do_not_compute_a: Optional[np.ndarray] = None,
    do_not_compute_b: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a downsampled AB LB_Keogh profile (A annotated by B)."""
    a = _as_float1d(a)
    b = _as_float1d(b)
    if subseqlen < 4:
        raise ValueError('bad subsequence length')
    if a.size - subseqlen + 1 < 1 or b.size - subseqlen + 1 < 1:
        raise ValueError('time series too short')

    sub_a = a.size - subseqlen + 1
    sub_b = b.size - subseqlen + 1

    if do_not_compute_a is None:
        dnc_a = np.zeros(sub_a, dtype=np.bool_)
    else:
        da = np.asarray(do_not_compute_a).reshape(-1)
        dnc_a = (da[:sub_a] != 0)

    if do_not_compute_b is None:
        dnc_b = np.zeros(sub_b, dtype=np.bool_)
    else:
        db = np.asarray(do_not_compute_b).reshape(-1)
        dnc_b = (db[:sub_b] != 0)

    return _lb_keogh_ab_profile_ds_core(a, b, subseqlen, int(warpmax), dnc_a, dnc_b)

def _stretch_startgrid(x_ds: np.ndarray, target_len: int, fill_value=np.inf) -> np.ndarray:
    """Stretch an array defined over a coarse subseq-start grid onto a finer subseq-start grid."""
    x_ds = np.asarray(x_ds)
    if target_len <= 0:
        return np.empty(0, dtype=x_ds.dtype if x_ds.size else np.float64)
    if x_ds.size == 0:
        return np.full(target_len, fill_value, dtype=np.float64)
    if x_ds.size == 1:
        return np.full(target_len, x_ds[0], dtype=x_ds.dtype)
    # map each fine index -> nearest coarse index
    idx = np.rint(np.linspace(0, x_ds.size - 1, target_len)).astype(np.int64)
    return x_ds[idx]

def _coarsen_prune_mask(mask_fine: np.ndarray, target_len: int) -> np.ndarray:
    """
    Safe coarsening: a coarse position is marked pruned (1) only if *all*
    corresponding fine positions are pruned. This cannot accidentally drop the true motif.
    """
    mask_fine = np.asarray(mask_fine, dtype=np.int8)
    n = mask_fine.size
    if target_len <= 0:
        return np.empty(0, dtype=np.int8)
    if n == 0:
        return np.zeros(target_len, dtype=np.int8)
    if target_len == 1:
        return np.array([1 if np.all(mask_fine == 1) else 0], dtype=np.int8)

    out = np.zeros(target_len, dtype=np.int8)
    # partition [0, n) into target_len bins
    for k in range(target_len):
        lo = int(np.floor(k * n / target_len))
        hi = int(np.floor((k + 1) * n / target_len))
        if hi <= lo:
            hi = min(lo + 1, n)
        out[k] = 1 if np.all(mask_fine[lo:hi] == 1) else 0
    return out

def swamp_ab_top1(
    a: np.ndarray,
    b: np.ndarray,
    subseqlen: int,
    warpmax: Optional[int] = None,
    max_downsample_factor: Optional[int] = None,
    return_diagnostics: bool = True,
) -> ABMotifResult:
    """Find the top-1 DTW motif between A and B (AB-join).

    This is the AB analogue of SWAMP.

    Parameters
    ----------
    a, b:
        1D time series arrays.
    subseqlen:
        Subsequence length L (same for A and B).
    warpmax:
        Sakoe-Chiba DTW radius w.
    max_downsample_factor:
        Coarsest downsample factor D to start Phase I.
        If None, we start at the largest power-of-two D such that L/D >= 4.
    return_diagnostics:
        If True, includes Phase I history arrays.

    Returns
    -------
    ABMotifResult
        distance, start index in A, start index in B.
    """
    a = _as_float1d(a)
    b = _as_float1d(b)
    n_a = a.size
    n_b = b.size
    if subseqlen < 4:
        raise ValueError('bad subsequence length')
    if n_a < subseqlen or n_b < subseqlen:
        raise ValueError('time series too short for given subseqlen')

    sub_a = n_a - subseqlen + 1
    sub_b = n_b - subseqlen + 1

    if warpmax == None:
        warpmax = subseqlen

    # Precompute rolling stats for full-res normalization
    mu_a, sig_a = _rolling_mean_std_ddof0(a, subseqlen)
    mu_b, sig_b = _rolling_mean_std_ddof0(b, subseqlen)

    # Pick a deterministic seed pair (first valid windows) to get a finite best-so-far
    seed_i = -1
    for i in range(sub_a):
        if np.isfinite(sig_a[i]) and sig_a[i] >= 0.0:
            seed_i = i
            break
    seed_j = -1
    for j in range(sub_b):
        if np.isfinite(sig_b[j]) and sig_b[j] >= 0.0:
            seed_j = j
            break
    if seed_i < 0 or seed_j < 0:
        # Degenerate: no valid z-normalized windows
        return ABMotifResult(float('inf'), -1, -1, tuple(), tuple(), tuple())

    qa = np.empty(subseqlen, dtype=np.float64)
    qb = np.empty(subseqlen, dtype=np.float64)
    _znorm_subseq(a, seed_i, subseqlen, mu_a, sig_a, qa)
    _znorm_subseq(b, seed_j, subseqlen, mu_b, sig_b, qb)
    best_sq = dtw_distance_ea_sq(qa, qb, int(warpmax), 1e300)
    best_so_far = float(np.sqrt(best_sq))
    best_i = int(seed_i)
    best_j = int(seed_j)

    # Phase I pruning on A positions
    dnc_a = np.zeros(sub_a, dtype=np.int8)

    hist_bsf = [best_so_far]
    hist_pruned_a = [int(dnc_a.sum())]
    hist_D = []

    # Track the last stretched LB profile for ordering in Phase II
    stretched_last = np.full(sub_a, np.inf, dtype=np.float64)
    j_hint_last = np.full(sub_a, -1, dtype=np.int64)

    # Choose starting downsample factor
    if max_downsample_factor is None:
        # largest power of two with L/D >= 4
        D = 1
        while (subseqlen // (D * 2)) >= 4:
            D *= 2
        max_downsample_factor = D
    if max_downsample_factor < 1:
        max_downsample_factor = 1

    D = int(max_downsample_factor)
    while D >= 1:
        hist_D.append(D)

        if D == 1:
            ds_a = a
            ds_b = b
            warp_d = int(warpmax)
            Ld = int(subseqlen)
            dnc_a_ds = dnc_a.astype(np.float64)
            dnc_b_ds = np.zeros(sub_b, dtype=np.float64)
        else:
            # Downsample by factor D (integer), allow small sizes
            na = max(1, n_a // D)
            nb = max(1, n_b // D)

            # If downsampling doesn't actually reduce length, skip this D level
            if na >= n_a or nb >= n_b:
                D //= 2
                continue

            ds_a = paa_updated(a, na)
            ds_b = paa_updated(b, nb)

            Ld = max(4, subseqlen // D)
            warp_d = max(0, min(warpmax // D, Ld - 1))

            # If series too short at this level, skip
            if ds_a.size < Ld or ds_b.size < Ld:
                D //= 2
                continue

            sub_a_ds = ds_a.size - Ld + 1

            # Only mark coarse cell pruned if all fine cells are pruned
            dnc_a_ds = _coarsen_prune_mask(dnc_a, sub_a_ds).astype(np.float64)
            dnc_b_ds = np.zeros(ds_b.size - Ld + 1, dtype=np.float64)

        if ds_a.size < Ld or ds_b.size < Ld:
            D //= 2
            continue

        sub_a_ds = ds_a.size - Ld + 1
        sub_b_ds = ds_b.size - Ld + 1
        if sub_a_ds < 1 or sub_b_ds < 1:
            D //= 2
            continue

        mp_a_ds, mpi_a_ds = lb_keogh_ab_profile_ds(ds_a, ds_b, Ld, warp_d, dnc_a_ds, dnc_b_ds)

        # Stretch mp_a_ds back to A's original subseq-start grid (start-grid mapping, not sample repeats)
        stretched = _stretch_startgrid(mp_a_ds.astype(np.float64), sub_a, fill_value=np.inf)

        # Stretch the argmin indices the same way, then map B_ds start indices -> B start indices
        j_hint_ds = _stretch_startgrid(mpi_a_ds.astype(np.float64), sub_a, fill_value=-1.0).astype(np.int64)

        j_hint = np.full(sub_a, -1, dtype=np.int64)
        valid = j_hint_ds >= 0
        if np.any(valid):
            if sub_b_ds <= 1:
                # only one possible start in downsampled B
                j_hint[valid] = 0
            else:
                scale_j = (sub_b - 1) / (sub_b_ds - 1)
                j_hint[valid] = np.rint(j_hint_ds[valid] * scale_j).astype(np.int64)
                j_hint[valid] = np.clip(j_hint[valid], 0, sub_b - 1)


        stretched_last = stretched
        j_hint_last = j_hint

        # If everything is Inf, nothing to do at this level
        if not np.isfinite(stretched).any():
            D //= 2
            continue

        # Improve best-so-far using the current best LB candidate (i in A, hinted j in B)
        sam_i = int(np.nanargmin(stretched))
        sam_j = int(j_hint[sam_i])
        if 0 <= sam_i < sub_a and 0 <= sam_j < sub_b:
            ok_a = _znorm_subseq(a, sam_i, subseqlen, mu_a, sig_a, qa)
            ok_b = _znorm_subseq(b, sam_j, subseqlen, mu_b, sig_b, qb)
            if ok_a and ok_b:
                d_sq = dtw_distance_ea_sq(qa, qb, int(warpmax), best_so_far * best_so_far)
                d = float(np.sqrt(d_sq))
                if d < best_so_far:
                    best_so_far = d
                    best_i = sam_i
                    best_j = sam_j

        # Prune A windows whose (scaled) lower bound already exceeds best-so-far
        for i in range(sub_a):
            if dnc_a[i] == 0 and np.isfinite(stretched[i]) and stretched[i] > best_so_far:
                dnc_a[i] = 1

        hist_bsf.append(best_so_far)
        hist_pruned_a.append(int(dnc_a.sum()))

        D //= 2

    # If everything pruned, return current best
    if int(dnc_a.sum()) >= sub_a:
        return ABMotifResult(best_so_far, best_i, best_j, tuple(hist_bsf) if return_diagnostics else tuple(),
                             tuple(hist_pruned_a) if return_diagnostics else tuple(),
                             tuple(hist_D) if return_diagnostics else tuple())

    # ---------- Phase II: ordered brute force with LBKim + LBKeogh + DTW ----------
    # Order A windows by increasing LB (best candidates first)
    order = np.argsort(stretched_last)

    U = np.empty(subseqlen, dtype=np.float64)
    L = np.empty(subseqlen, dtype=np.float64)

    best_sq = best_so_far * best_so_far

    for i in order:
        if i < 0 or i >= sub_a:
            continue
        if dnc_a[i] == 1:
            continue
        lb_i = stretched_last[i]
        if np.isfinite(lb_i) and lb_i >= best_so_far:
            # since order is ascending, no later i can help
            break

        ok_a = _znorm_subseq(a, int(i), subseqlen, mu_a, sig_a, qa)
        if not ok_a:
            dnc_a[i] = 1
            continue
        envelope_sakoe_chiba(qa, int(warpmax), U, L)

        # Try hinted B first
        hinted = int(j_hint_last[i])
        if 0 <= hinted < sub_b:
            ok_b = _znorm_subseq(b, hinted, subseqlen, mu_b, sig_b, qb)
            if ok_b:
                lb_sq = lb_keogh_ea_sq(qb, U, L, best_sq)
                if lb_sq < best_sq:
                    d_sq = dtw_distance_ea_sq(qa, qb, int(warpmax), best_sq)
                    if d_sq < best_sq:
                        best_sq = d_sq
                        best_so_far = float(np.sqrt(d_sq))
                        best_i = int(i)
                        best_j = int(hinted)

        # Full scan of B windows with pruning
        for j in range(sub_b):
            if j == hinted:
                continue
            if not np.isfinite(sig_b[j]) or sig_b[j] < 0.0:
                continue

            ok_b = _znorm_subseq(b, j, subseqlen, mu_b, sig_b, qb)
            if not ok_b:
                continue

            # LB_Kim (very cheap)
            lb_kim = abs(qa[0] - qb[0])
            t = abs(qa[-1] - qb[-1])
            if t > lb_kim:
                lb_kim = t
            if lb_kim >= best_so_far:
                continue

            lb_sq = lb_keogh_ea_sq(qb, U, L, best_sq)
            if lb_sq >= best_sq:
                continue

            d_sq = dtw_distance_ea_sq(qa, qb, int(warpmax), best_sq)
            if d_sq < best_sq:
                best_sq = d_sq
                best_so_far = float(np.sqrt(d_sq))
                best_i = int(i)
                best_j = int(j)

        # Optional: additional pruning in A after improvements
        if best_so_far < 1e300:
            # We can prune any A window whose LB already exceeds current best
            # (safe because stretched_last is a lower bound per i)
            # This helps when the ordering array wasn't computed at full-res.
            if stretched_last[i] > best_so_far:
                dnc_a[i] = 1

    return ABMotifResult(
        best_so_far,
        best_i,
        best_j,
        tuple(hist_bsf) if return_diagnostics else tuple(),
        tuple(hist_pruned_a) if return_diagnostics else tuple(),
        tuple(hist_D) if return_diagnostics else tuple(),
    )

__all__ = [
    'MotifResult',
    'ABMotifResult',
    'paa_updated',
    'mpx_v2',
    'lb_keogh_mp_updated',
    'lb_keogh_ab_profile_ds',
    'swamp_top1',
    'swamp_ab_top1',
]

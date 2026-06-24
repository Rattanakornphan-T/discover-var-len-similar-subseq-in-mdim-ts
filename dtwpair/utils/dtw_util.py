import numpy as np
from numba import njit, float64, int32, prange
import heapq


@njit(float64(float64[:, :], int32[:, :]))
def _sample_dtw_dist(cost_mat: np.ndarray, constraint: np.ndarray) -> float:
    """
    Distance of a sample DTW path from the cost matrix.
    Parameters
    ----------
    cost_mat : np.ndarray
        The cost matrix.
    constraint : np.ndarray
        The constraint matrix.
    x_window_size : int
        The size of the window in the x direction.
    y_window_size : int
        The size of the window in the y direction.

    Returns
    -------
    float
        The sampled DTW distance.
    """
    rows = cost_mat.shape[0]
    cols = cost_mat.shape[1]

    dist = cost_mat[0, 0]
    i = 0
    j = 0

    while i+1 < rows or j+1 < cols:
        i_start = i
        j_start = j

        while i+1 < rows and constraint[0, i+1] <= j+1 < constraint[1, i+1]:
            i += 1
            j += 1
            dist += cost_mat[i, j]

        while i+1 < rows and constraint[0, i+1] <= j < constraint[1, i+1]:
            i += 1
            dist += cost_mat[i, j]
        
        while constraint[0, i] <= j+1 < constraint[1, i] and \
            (i+1 >= rows or \
             not (constraint[0, i+1] <= j+1 < constraint[1, i+1])):

            dist += cost_mat[i, j+1]
            j += 1
        
        if i == i_start and j == j_start:
            raise ValueError("Cannot find a valid path. Please check the constraint matrix.")

    return dist

@njit(float64[:, :](float64[:, :], int32[:, :], int32, int32))
def _slide_sample_dtw_dist(cost_mat: np.ndarray,
                           constraint: np.ndarray,
                           size0: int,
                           size1: int) -> np.ndarray:
    """
    Distance of a sample DTW path from the cost matrix.
    Parameters
    ----------
    cost_mat : np.ndarray
        The cost matrix.
    constraint : np.ndarray
        The constraint matrix.
    x_window_size : int
        The size of the window in the x direction.
    y_window_size : int
        The size of the window in the y direction.

    Returns
    -------
    float
        The sampled DTW distance.
    """
    rows = cost_mat.shape[0]
    cols = cost_mat.shape[1]

    dist = np.empty((rows - size0 + 1, cols - size1 + 1), dtype=np.float64)
    for i in range(rows - size0 + 1):
        for j in range(cols - size1 + 1):
            dist[i, j] = _sample_dtw_dist(cost_mat[i:i+size0, j:j+size1], constraint)

    return dist

def slide_sample_dtw_dist(cost_mat: np.ndarray,
                          size0: int,
                          size1: int,
                          constraint: np.ndarray=None) -> np.ndarray:
    """
    Distance of a sample DTW path from the cost matrix.
    Parameters
    ----------
    cost_mat : np.ndarray
        The cost matrix.
    constraint : np.ndarray
        The constraint matrix.
    x_window_size : int
        The size of the window in the x direction.
    y_window_size : int
        The size of the window in the y direction.

    Returns
    -------
    float
        The sampled DTW distance.
    """
    if constraint is None:
        constraint = np.repeat(
            np.array([[0, size1]]),
            len(cost_mat),
            axis=0
        ).T

    if np.any(constraint[0] < 0) or np.any(constraint[1] > size1):
        raise ValueError("Constraint must be in the range [0, size1]")

    if np.any(constraint[0] > constraint[1]):
        raise ValueError("Constraint must have start < end")
    
    if np.any(constraint[0, :1] > constraint[0, 1:]) or np.any(constraint[1, :1] > constraint[1, 1:]):
        raise ValueError("Constraint must be inclement")

    return _slide_sample_dtw_dist(cost_mat.astype(np.float64),
                                  constraint.astype(np.int32),
                                  size0, size1)

@njit
def lower_bound_distance_with_constraint(cost_mat: np.ndarray,
                                         size0: int,
                                         size1: int,
                                         region: np.ndarray):
    s0, s1 = cost_mat.shape
    out_s0 = s0 - size0 + 1
    out_s1 = s1 - size1 + 1
    result = np.empty((out_s0, out_s1), dtype=np.float64)

    for i in prange(out_s0):
        for j in prange(out_s1):
            s = 0
            for pi in range(size0):
                val = np.inf
                for pj in range(region[0, pi], region[1, pi]):
                    val = min(val, cost_mat[i + pi, j + pj])
                s += val
            result[i, j] = s

    return result


# -----------------------------
# Pair-index constraint region
# -----------------------------

@njit(cache=True)
def compute_sakoechiba_region(n_rows: int, n_cols: int, window_size: int) -> np.ndarray:
    """Return region array (2, n_rows) with start/end (end exclusive) for each row i:
    allowed j satisfy start <= j < end where |i - j| <= window_size.
    """
    region = np.empty((2, n_rows), dtype=np.int32)
    for i in range(n_rows):
        s = i - window_size
        if s < 0:
            s = 0
        e = i + window_size + 1
        if e > n_cols:
            e = n_cols
        region[0, i] = s
        region[1, i] = e
    return region

@njit(cache=True, parallel=True)
def mask_distance_matrix(cost_mat: np.ndarray, region: np.ndarray, val: float = np.inf) -> np.ndarray:
    """Mask 2D matrix in-place according to region (2, n_rows)."""
    n_rows, n_cols = cost_mat.shape
    for i in prange(n_rows):
        start = region[0, i]
        end = region[1, i]
        for j in range(0, start):
            cost_mat[i, j] = val
        for j in range(end, n_cols):
            cost_mat[i, j] = val
    return cost_mat


# ----------------------------------------
# Fast bounds for MaxPath / MinPath (paper)
# ----------------------------------------

@njit(cache=True, fastmath=True, parallel=True)
def _row_sliding_min(arr: np.ndarray, win: int) -> np.ndarray:
    """Sliding min along axis=1 (rows) using a monotonic deque per row."""
    rows, cols = arr.shape
    out_cols = cols - win + 1
    out = np.empty((rows, out_cols), dtype=np.float64)
    for r in prange(rows):
        dq = np.empty(cols, dtype=np.int32)
        head = 0
        tail = 0
        for c in range(cols):
            v = arr[r, c]
            while tail > head and arr[r, dq[tail - 1]] >= v:
                tail -= 1
            dq[tail] = c
            tail += 1
            if dq[head] <= c - win:
                head += 1
            if c >= win - 1:
                out[r, c - win + 1] = arr[r, dq[head]]
    return out

def minpath_lower_bound(cost_mat: np.ndarray, size0: int, size1: int) -> np.ndarray:
    """Compute MinPath lower bound matrix for all window pairs."""
    if size0 >= size1:
        w = size0
        h = size1
        cm = np.asarray(cost_mat, dtype=np.float64)
        transposed = False
    else:
        w = size1
        h = size0
        cm = np.asarray(cost_mat.T, dtype=np.float64)
        transposed = True

    row_min = _row_sliding_min(cm, h)  # (rows, cols-h+1)

    # vertical sliding sum with prefix sums (numpy is very fast here)
    ps = np.cumsum(row_min, axis=0, dtype=np.float64)
    out = ps[w-1:, :] - np.vstack((np.zeros((1, ps.shape[1]), dtype=np.float64), ps[:-w, :]))

    if transposed:
        out = out.T
    return out

@njit(cache=True, fastmath=True)
def _diag_prefix_sum(cm: np.ndarray) -> np.ndarray:
    rows, cols = cm.shape
    ps = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            v = cm[i, j]
            if i > 0 and j > 0:
                v += ps[i - 1, j - 1]
            ps[i, j] = v
    return ps

@njit(cache=True, fastmath=True)
def _col_prefix_sum(cm: np.ndarray) -> np.ndarray:
    rows, cols = cm.shape
    ps = np.empty((rows + 1, cols), dtype=np.float64)
    for j in range(cols):
        ps[0, j] = 0.0
    for i in range(rows):
        for j in range(cols):
            ps[i + 1, j] = ps[i, j] + cm[i, j]
    return ps

@njit(cache=True, fastmath=True)
def _diag_segment_sum(diag_ps: np.ndarray, i0: int, j0: int, L: int) -> float:
    if L <= 0:
        return 0.0
    i1 = i0 + L - 1
    j1 = j0 + L - 1
    s = diag_ps[i1, j1]
    if i0 > 0 and j0 > 0:
        s -= diag_ps[i0 - 1, j0 - 1]
    return s

@njit(cache=True, fastmath=True)
def _col_segment_sum(col_ps: np.ndarray, i0: int, j: int, L: int) -> float:
    return col_ps[i0 + L, j] - col_ps[i0, j]

@njit(cache=True, fastmath=True, parallel=True)
def _maxpath_upper_bound_numba(cm: np.ndarray, w: int, h: int) -> np.ndarray:
    rows, cols = cm.shape
    out_rows = rows - w + 1
    out_cols = cols - h + 1

    diag_ps = _diag_prefix_sum(cm)
    col_ps = _col_prefix_sum(cm)

    Ld = h - 1
    Lv = w - Ld  # = w - h + 1

    out = np.empty((out_rows, out_cols), dtype=np.float64)
    for i in prange(out_rows):
        for j in range(out_cols):
            s = _diag_segment_sum(diag_ps, i, j, Ld)
            s += _col_segment_sum(col_ps, i + Ld, j + Ld, Lv)
            out[i, j] = s
    return out

def maxpath_upper_bound(cost_mat: np.ndarray, size0: int, size1: int) -> np.ndarray:
    """Compute MaxPath upper bound matrix for all window pairs."""
    if size0 >= size1:
        w = size0
        h = size1
        cm = np.asarray(cost_mat, dtype=np.float64)
        transposed = False
    else:
        w = size1
        h = size0
        cm = np.asarray(cost_mat.T, dtype=np.float64)
        transposed = True

    out = _maxpath_upper_bound_numba(cm, int(w), int(h))
    if transposed:
        out = out.T
    return out


# -----------------------------
# Exact DTW for candidate pairs
# -----------------------------

@njit(cache=True, fastmath=True)
def dtw_subseq_cost(cost_mat: np.ndarray,
                    a: int, b: int,
                    size0: int, size1: int,
                    prev: np.ndarray,
                    curr: np.ndarray,
                    cutoff: float) -> float:
    """Exact DTW cost for submatrix starting at (a,b)."""
    inf = 1e308
    prev[0] = 0.0
    for j in range(1, size1 + 1):
        prev[j] = inf

    for i in range(1, size0 + 1):
        curr[0] = inf
        row_min = inf
        ai = a + i - 1
        for j in range(1, size1 + 1):
            bj = b + j - 1
            c = cost_mat[ai, bj]
            v1 = prev[j - 1]
            v2 = prev[j]
            v3 = curr[j - 1]
            m = v1
            if v2 < m:
                m = v2
            if v3 < m:
                m = v3
            val = c + m
            curr[j] = val
            if val < row_min:
                row_min = val

        if cutoff < inf and row_min > cutoff:
            return inf

        for j in range(size1 + 1):
            prev[j] = curr[j]

    return prev[size1]

@njit(cache=True)
def topk_from_sorted_candidates(sorted_flat_idx: np.ndarray,
                               sorted_lb: np.ndarray,
                               n_cols: int,
                               cost_mat: np.ndarray,
                               size0: int,
                               size1: int,
                               top_k: int,
                               tie_eps: float = 0.0):
    """Exact DTW over LB-sorted candidates, returning **top-k with ties**.

    Parameters
    ----------
    sorted_flat_idx : 1D int array
        Candidate window-pair indices in row-major flattened form, sorted by LB.
    sorted_lb : 1D float array
        Corresponding lower bounds (same order as `sorted_flat_idx`).
    n_cols : int
        Number of columns in the 2D (Mx, My) window-pair grid.
    cost_mat : 2D float array
        Full pointwise cost matrix between the two full series.
    size0, size1 : int
        Window sizes in x and y.
    top_k : int
        Requested number of nearest neighbours.
    tie_eps : float
        Treat |d - d_k| <= tie_eps as a tie for k-th distance.
    """

    # candidates is a heap of [-d, i, j] (so heap[0] corresponds to the WORST / largest d)
    candidates = [[np.float64(1.0), np.int32(0), np.int32(0)] for _ in range(0)]

    worst = 1e308

    prev = np.empty(size1 + 1, dtype=np.float64)
    curr = np.empty(size1 + 1, dtype=np.float64)

    for k in range(sorted_flat_idx.shape[0]):
        lb = sorted_lb[k]
        if len(candidates) >= top_k and lb > worst + tie_eps:
            break

        flat = sorted_flat_idx[k]
        i = flat // n_cols
        j = flat - i * n_cols

        d = dtw_subseq_cost(cost_mat, i, j, size0, size1, prev, curr, worst)

        # Keep filling until we have top_k, and keep ties at the k-th distance.
        if len(candidates) < top_k or (-candidates[0][0] - d) <= tie_eps and (d - (-candidates[0][0])) <= tie_eps:
            heapq.heappush(candidates, [-d, np.int32(i), np.int32(j)])
        elif d < -candidates[0][0] - tie_eps:
            # Remove all strictly-worse items while the heap is overfull.
            while len(candidates) >= top_k and d < -candidates[0][0] - tie_eps:
                heapq.heappop(candidates)
            heapq.heappush(candidates, [-d, np.int32(i), np.int32(j)])

        if len(candidates) >= top_k:
            worst = -candidates[0][0]

    m = len(candidates)
    best_i = np.empty(m, dtype=np.int32)
    best_j = np.empty(m, dtype=np.int32)
    best_d = np.empty(m, dtype=np.float64)

    for t in range(m):
        best_d[t] = -candidates[t][0]
        best_i[t] = candidates[t][1]
        best_j[t] = candidates[t][2]

    return best_i, best_j, best_d

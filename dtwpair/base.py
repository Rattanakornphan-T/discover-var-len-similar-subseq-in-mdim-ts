import numpy as np
from numba import njit
from typing import Callable, Tuple
import heapq

from pyts.metrics.dtw import _accumulated_cost_matrix_no_region, _accumulated_cost_matrix_region
from .utils.cost import euclidean_distance, squared_euclidean_cost_matrix
from .utils.normalization import znorm
from .utils.dtw_distance import BaseDtwDistance, SquaredDtwDistance

class BaseDtwPair:
    def __init__(self,
                 cost_function: Callable = None,
                 dtw_distance: BaseDtwDistance = None,
                 verbose: bool = True,
                 dist_mat_batch_size: int = 10 ** 6):
        """Base class for DTW-Pair.

        Parameters
        ----------
        cost_function : callable
            Cost function to calculate cost between point pairs.
            Default is Euclidean distance.
        dtw_distance : BaseDtwDistance
            Wrapper that transforms per-point costs before DTW and inverts the final DTW cost.
            Default is SquaredDtwDistance (i.e., DTW over squared pointwise Euclidean costs).
        verbose : bool
            Print timing info.
        dist_mat_batch_size : int
            Batch size for fallback cost-matrix computation.
        """
        self.cost_function = cost_function or euclidean_distance
        self.dtw_distance = dtw_distance or SquaredDtwDistance()
        self.verbose = verbose
        self.dist_mat_batch_size = dist_mat_batch_size

    def get_index_pairs(self, size1, size2):
        _groups_idx_1 = np.repeat([np.arange(size1)], size2, axis=0).swapaxes(0, 1)
        _groups_idx_2 = np.repeat([np.arange(size2)], size1, axis=0)
        groups_idx = np.concatenate([
            _groups_idx_1[:, :, np.newaxis],
            _groups_idx_2[:, :, np.newaxis]
        ], axis=-1) #.reshape(size1 * size2, 2)

        return groups_idx

    def validate_input(self,
                       x: np.ndarray,
                       y: np.ndarray,
                       x_window_size: int,
                       y_window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(x, (list, np.ndarray)):
            raise TypeError("x must be a numpy array")
        if not isinstance(y, (list, np.ndarray)):
            raise TypeError("y must be a numpy array")

        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim > 2:
            raise ValueError("x must be a 1D or 2D array")
        if y.ndim > 2:
            raise ValueError("y must be a 1D or 2D array")

        if not (1 <= x_window_size <= x.shape[0]):
            raise ValueError("x_window_size must be between 1 and len(x)")
        if not (1 <= y_window_size <= y.shape[0]):
            raise ValueError("y_window_size must be between 1 and len(y)")

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # ensure float64 contiguous for fast math
        x = np.asarray(x, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64, order="C")
        return x, y

    def z_normalize(self, x: np.ndarray, eps: float = 1e-12, engine: str = "numba") -> np.ndarray:
        return znorm(x, eps=eps, engine=engine)

    def get_cost_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute per-point cost matrix between two time series.

        Fast path:
        - if cost_function is euclidean_distance AND dtw_distance is SquaredDtwDistance,
          compute squared Euclidean distances directly (avoids sqrt + square).

        Fallback:
        - compute cost_function(x_i, y_j) then dtw_distance.calc elementwise.
        """
        M, N = x.shape[0], y.shape[0]

        # Fast path: squared Euclidean
        if (self.cost_function is euclidean_distance) and isinstance(self.dtw_distance, SquaredDtwDistance):
            # x and y are (M, D), (N, D)
            if x.shape[1] == 1 and y.shape[1] == 1:
                dx = x[:, 0][:, None] - y[:, 0][None, :]
                return dx * dx
            # Use BLAS for dot product when D > 1
            x2 = np.sum(x * x, axis=1)[:, None]
            y2 = np.sum(y * y, axis=1)[None, :]
            dot = x @ y.T
            dist2 = x2 + y2 - 2.0 * dot
            # numerical cleanup
            np.maximum(dist2, 0.0, out=dist2)
            return dist2

        # Alternative fast path: if dtw_distance is SquaredDtwDistance but cost_function is not euclidean_distance
        # we still can apply dtw_distance.calc on the result; compute in chunks to limit peak memory for broadcasting.
        total = M * N
        batch_size = min(self.dist_mat_batch_size or total, total)
        flat = np.empty(total, dtype=np.float64)

        for i0 in range(0, total, batch_size):
            idx = np.arange(i0, min(total, i0 + batch_size), dtype=np.int64)
            xr = idx // N
            yr = idx - xr * N
            flat[i0:i0 + idx.shape[0]] = self.cost_function(x[xr], y[yr])

        cost = flat.reshape(M, N)
        return self.dtw_distance.calc(cost)

@njit
def dtw_pairs_loop_no_bound(pairs: np.ndarray,
                            x_window_size: int,
                            y_window_size: int,
                            cost_mat: np.ndarray,
                            top_k: int):

    candidates = [[np.float64(1.0), 1, 1] for x in range(0)]

    for i in range(len(pairs)):
        pair = pairs[i]

        acc_cost_mat = _accumulated_cost_matrix_no_region(
            cost_mat[pair[0]: pair[0] + x_window_size, pair[1]: pair[1] + y_window_size]
        )
        dtw_d = acc_cost_mat[-1, -1]

        if len(candidates) < top_k or -candidates[0][0] == dtw_d:
            heapq.heappush(candidates, [-dtw_d, pair[0], pair[1]])
        elif dtw_d < -candidates[0][0]:
            while len(candidates) >= top_k and dtw_d < -candidates[0][0]:
                heapq.heappop(candidates)
            heapq.heappush(candidates, [-dtw_d, pair[0], pair[1]])

    for i in range(len(candidates)):
        candidates[i][0] = -candidates[i][0]

    return candidates

@njit
def dtw_pairs_loop(pairs: np.ndarray,
                   min_dtw_pairs: np.ndarray,
                   x_window_size: int,
                   y_window_size: int,
                   cost_mat: np.ndarray,
                   top_k: int):

    candidates = [[np.float64(1.0), 1, 1] for x in range(0)]

    for i in range(len(pairs)):
        pair = pairs[i]

        min_d = min_dtw_pairs[i]

        if len(candidates) >= top_k and min_d > -candidates[0][0]:
            break

        acc_cost_mat = _accumulated_cost_matrix_no_region(
            cost_mat[pair[0]: pair[0] + x_window_size, pair[1]: pair[1] + y_window_size]
        )
        dtw_d = acc_cost_mat[-1, -1]

        if len(candidates) < top_k or -candidates[0][0] == dtw_d:
            heapq.heappush(candidates, [-dtw_d, pair[0], pair[1]])
        elif dtw_d < -candidates[0][0]:
            while len(candidates) >= top_k and dtw_d < -candidates[0][0]:
                heapq.heappop(candidates)
            heapq.heappush(candidates, [-dtw_d, pair[0], pair[1]])

    for i in range(len(candidates)):
        candidates[i][0] = -candidates[i][0]

    return candidates

@njit
def dtw_pairs_loop_no_bound_with_region(pairs: np.ndarray,
                                        x_window_size: int,
                                        y_window_size: int,
                                        cost_mat: np.ndarray,
                                        top_k: int,
                                        region: np.ndarray):

    candidates = [[np.float64(1.0), 1, 1] for x in range(0)]

    for i in range(len(pairs)):
        pair = pairs[i]

        acc_cost_mat = _accumulated_cost_matrix_region(
            cost_mat[pair[0]: pair[0] + x_window_size, pair[1]: pair[1] + y_window_size],
            region
        )
        dtw_d = acc_cost_mat[-1, -1]

        if len(candidates) < top_k or -candidates[0][0] == dtw_d:
            heapq.heappush(candidates, [-dtw_d, pair[0], pair[1]])
        elif dtw_d < -candidates[0][0]:
            while len(candidates) >= top_k and dtw_d < -candidates[0][0]:
                heapq.heappop(candidates)
            heapq.heappush(candidates, [-dtw_d, pair[0], pair[1]])

    for i in range(len(candidates)):
        candidates[i][0] = -candidates[i][0]

    return candidates

@njit
def dtw_pairs_loop_with_region(pairs: np.ndarray,
                               min_dtw_pairs: np.ndarray,
                               x_window_size: int,
                               y_window_size: int,
                               cost_mat: np.ndarray,
                               top_k: int,
                               region: np.ndarray):

    candidates = [[np.float64(1.0), 1, 1] for x in range(0)]

    for i in range(len(pairs)):
        pair = pairs[i]

        min_d = min_dtw_pairs[i]

        if len(candidates) >= top_k and min_d > -candidates[0][0]:
            break

        acc_cost_mat = _accumulated_cost_matrix_region(
            cost_mat[pair[0]: pair[0] + x_window_size, pair[1]: pair[1] + y_window_size],
            region
        )
        dtw_d = acc_cost_mat[-1, -1]

        if len(candidates) < top_k or -candidates[0][0] == dtw_d:
            heapq.heappush(candidates, [-dtw_d, pair[0], pair[1]])
        elif dtw_d < -candidates[0][0]:
            while len(candidates) >= top_k and dtw_d < -candidates[0][0]:
                heapq.heappop(candidates)
            heapq.heappush(candidates, [-dtw_d, pair[0], pair[1]])

    for i in range(len(candidates)):
        candidates[i][0] = -candidates[i][0]

    return candidates

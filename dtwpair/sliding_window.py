import numpy as np
from typing import Tuple

from .base import BaseDtwPair
from .utils.timer import Timer
from .utils.dtw_util import (
    minpath_lower_bound,
    maxpath_upper_bound,
    compute_sakoechiba_region,
    mask_distance_matrix,
    topk_from_sorted_candidates,
)

class SlidingWindowDtwPair(BaseDtwPair):
    def approximate_lower_bound(self, cost_mat: np.ndarray, x_window_size: int, y_window_size: int) -> np.ndarray:
        return minpath_lower_bound(cost_mat, x_window_size, y_window_size)

    def approximate_upper_bound(self, cost_mat: np.ndarray, x_window_size: int, y_window_size: int) -> np.ndarray:
        return maxpath_upper_bound(cost_mat, x_window_size, y_window_size)

    def approximate_dtw_pair(self, cost_mat: np.ndarray, x_window_size: int, y_window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.verbose:
            print("Computing MaxPath (upper bound)")
        with Timer(self.verbose):
            max_dtw = self.approximate_upper_bound(cost_mat, x_window_size, y_window_size)

        if self.verbose:
            print("Computing MinPath (lower bound)")
        with Timer(self.verbose):
            min_dtw = self.approximate_lower_bound(cost_mat, x_window_size, y_window_size)

        return max_dtw, min_dtw

    def subsequence_nearest_neighbour(self,
                                      x: np.ndarray,
                                      y: np.ndarray,
                                      x_window_size: int,
                                      y_window_size: int,
                                      top_k: int = 1,
                                      stride: int = 1,
                                      allow_temporal_shift: int = -1,
                                      normalization: bool = True):
        """
        Compute the nearest neighbour distance between two sequences.
        Parameters
        ----------
        x : np.ndarray
            First timeseries of scalars (shape (M,)) or vector with D dimension (shape (M, D)).
        y : np.ndarray
            Second timeseries of scalars (shape (N,)) or vector with D dimension (shape (N, D)).
        x_window_size : int
            Window size of subsequences of the first timeseries.
        y_window_size : int
            Window size of subsequences  the second timeseries.
        top_k : int
            Number of nearest neighbours to return.
        stride : int
            Stride of the sliding window.
        allow_temporal_shift : int
            How far subsequences in a pair can be further apart.
        normalization : bool
            Whether to normalize both time series first.
        """
        x, y = self.validate_input(x, y, x_window_size, y_window_size)  # (M,D), (N,D)

        if normalization:
            x = self.z_normalize(x)
            y = self.z_normalize(y)

        if self.verbose:
            print("Computing full cost matrix")
        with Timer(self.verbose):
            cost_mat = self.get_cost_matrix(x, y)  # (M,N)

        # bounds for all window pairs
        max_dtw_pairs, min_dtw_pairs = self.approximate_dtw_pair(cost_mat, x_window_size, y_window_size)
        # both are shape (Mx, My) where Mx=M-x_window_size+1, My=N-y_window_size+1

        # mask for striding / temporal-shift constraints (on window start indices)
        mask = np.zeros(min_dtw_pairs.shape, dtype=bool)

        if stride > 1:
            keep = np.zeros_like(mask)
            keep[::stride, ::stride] = True
            mask = ~keep

        if allow_temporal_shift != -1:
            region = compute_sakoechiba_region(min_dtw_pairs.shape[0], min_dtw_pairs.shape[1], int(allow_temporal_shift))
            # apply region to mask: positions outside region get masked
            tmp = mask.astype(np.float64)
            tmp = mask_distance_matrix(tmp, region.astype(np.int32), 1.0)
            mask = tmp.astype(np.bool_)

        max_dtw_pairs = max_dtw_pairs.copy()
        min_dtw_pairs = min_dtw_pairs.copy()
        max_dtw_pairs[mask] = np.inf
        min_dtw_pairs[mask] = np.inf

        if self.verbose:
            print("Pruning candidates")
        with Timer(self.verbose):
            flat_ub = max_dtw_pairs.ravel()
            if top_k > 1:
                # kth smallest upper bound threshold
                ub_thresh = np.partition(flat_ub, top_k - 1)[top_k - 1]
            else:
                ub_thresh = np.min(flat_ub)

            flat_lb = min_dtw_pairs.ravel()
            cand_flat = np.flatnonzero(flat_lb <= ub_thresh)

            if cand_flat.size == 0:
                return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float64)

            cand_lb = flat_lb[cand_flat]
            order = np.argsort(cand_lb)
            cand_flat = cand_flat[order].astype(np.int64)
            cand_lb = cand_lb[order].astype(np.float64)

            if self.verbose:
                total_pairs = flat_lb.size
                print(f"Pruned {total_pairs - cand_flat.size} out of {total_pairs} pairs")

        # exact DTW on candidates (in LB order)
        if self.verbose:
            print("Computing exact DTW on candidates")
        with Timer(self.verbose):
            n_cols = min_dtw_pairs.shape[1]
            best_i, best_j, best_d = topk_from_sorted_candidates(
                cand_flat,
                cand_lb,
                n_cols,
                cost_mat,
                x_window_size,
                y_window_size,
                int(top_k),
            )

            # Convert to final distance scale
            best_d = self.dtw_distance.invert(best_d)

            # Sort ascending
            sort_idx = np.argsort(best_d)
            best_i = best_i[sort_idx]
            best_j = best_j[sort_idx]
            best_d = best_d[sort_idx]

            pairs = np.empty((best_i.shape[0], 2), dtype=np.int32)
            pairs[:, 0] = best_i
            pairs[:, 1] = best_j
            return pairs, best_d

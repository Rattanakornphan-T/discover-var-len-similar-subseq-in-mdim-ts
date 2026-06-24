import numpy as np
from typing import Callable, Tuple
from .utils.pooling import max_pooling, sum_pooling
from .utils.dtw_util import slide_sample_dtw_dist, lower_bound_distance_with_constraint, mask_distance_matrix
from .base import BaseDtwPair, dtw_pairs_loop_with_region, dtw_pairs_loop_no_bound_with_region
from .utils.timer import Timer
from .utils.dtw_distance import BaseDtwDistance
from pyts.metrics.dtw import _compute_region







class SchibaSlidingWindowDtwPair(BaseDtwPair):
    def __init__(self,
                 cost_function: Callable=None,
                 dtw_distance: BaseDtwDistance=None,
                 window_size: int=3,
                 verbose: bool=True):
        """
        SakoeChiba class for DTW-Pair.
        Parameters
        ----------
        cost_function : callable
            Cost function to calculate cost matrix (cost between each data point pairs).
        dtw_distance : BaseDtwDistance
            Distance function to calculate DTW distance. Default is SquaredDtwDistance.
        window_size : int
            Sakoe-Chiba window size. Default is 0.1 (10% of the longest window size).
        """
        super().__init__(cost_function=cost_function,
                         dtw_distance=dtw_distance,
                         verbose=verbose)
        self.window_size = window_size
    

    def approximate_lower_bound(self,
                               cost_mat: np.ndarray, # (M, N)
                               x_window_size: int, # < M
                               y_window_size: int, # < N
                               region: np.ndarray,
                               region_t: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
        if x_window_size > y_window_size:
            width = x_window_size
            height = y_window_size
        else:
            width = y_window_size
            height = x_window_size
            cost_mat = cost_mat.T
            region = region_t
        
        h_sum_min_pool = lower_bound_distance_with_constraint(
            cost_mat,
            width,
            height,
            region
        )
        
        if not (x_window_size > y_window_size):
            h_sum_min_pool = h_sum_min_pool.T

        return h_sum_min_pool
    
    def approximate_upper_bound(self,
                               cost_mat: np.ndarray, # (M, N)
                               x_window_size: int, # < M
                               y_window_size: int, # < N
                               region: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return slide_sample_dtw_dist(
            cost_mat,
            x_window_size,
            y_window_size,
            region
        ) # (M - x_window_size + 1, N - y_window_size + 1)

    def approximate_dtw_pair(self,
                            cost_mat: np.ndarray, # (M, N)
                            x_window_size: int,
                            y_window_size: int,
                            region: np.ndarray,
                            region_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.verbose:
            print("Computing max_dtw")
        with Timer(self.verbose):
            max_dtw = self.approximate_upper_bound(cost_mat, x_window_size, y_window_size, region)
        
        if self.verbose:
            print("Computing min_dtw")
        with Timer(self.verbose):
            min_dtw = self.approximate_lower_bound(cost_mat, x_window_size, y_window_size, region, region_t)
        return max_dtw, min_dtw
    
    def subsequence_nearest_neighbour(self,
                                      x: np.ndarray,
                                      y: np.ndarray,
                                      x_window_size: int,
                                      y_window_size: int,
                                      top_k: int=1,
                                      normalization: bool=True) -> float:
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
        Returns
        -------
        float
            Nearest neighbour distance.
        """
        
        x, y = self.validate_input(x, y, x_window_size, y_window_size) # (M, D), (N, D)

        if normalization:
            x = self.z_normalize(x)
            y = self.z_normalize(y)

        # Compute the distance matrix
        cost_mat = self.get_cost_matrix(x, y) # (M, N)

        region = _compute_region(x_window_size, y_window_size, "sakoechiba",
                                 "precomputed", window_size=self.window_size)
        region_t = _compute_region(y_window_size, x_window_size, "sakoechiba",
                                   "precomputed", window_size=self.window_size)

        # Approximate DTW of each subsequences pair
        max_dtw_pairs, min_dtw_pairs = self.approximate_dtw_pair(cost_mat, x_window_size, y_window_size, region, region_t)
        # ( (M - x_window_size + 1, N - y_window_size + 1), ) * 2
        
        if self.verbose:
            print("Computing dtw")
        with Timer(self.verbose):
            min_dtw = np.min(max_dtw_pairs)

            # Flatten the DTW aproximation matrix
            pairs = self.get_index_pairs(
                len(x) - x_window_size + 1,
                len(y) - y_window_size + 1
            ) # ((M - x_window_size + 1), (N - y_window_size + 1), 2)
            assert pairs.shape[0] == max_dtw_pairs.shape[0], f"Pairs shape: {pairs.shape}, max_dtw_pairs shape: {max_dtw_pairs.shape}"
            assert pairs.shape[0] == min_dtw_pairs.shape[0], f"Pairs shape: {pairs.shape}, min_dtw_pairs shape: {min_dtw_pairs.shape}"
            pairs = pairs.reshape(-1, 2) # ((M - x_window_size + 1) * (N - y_window_size + 1), 2)

            max_dtw_pairs = max_dtw_pairs.reshape(-1) # ((M - x_window_size + 1) * (N - y_window_size + 1))
            min_dtw_pairs = min_dtw_pairs.reshape(-1) # ((M - x_window_size + 1) * (N - y_window_size + 1))

            n_pairs = pairs.shape[0]

            # Filter only the candidate pairs
            filter = min_dtw_pairs <= min_dtw
            if sum(filter) >= top_k:
                pairs = pairs[filter]
                max_dtw_pairs = max_dtw_pairs[filter]
                min_dtw_pairs = min_dtw_pairs[filter]
            
            if self.verbose:
                print(f"Pruned {n_pairs - pairs.shape[0]} out of {n_pairs} pairs")

            # Sort the pairs by lower bound
            sorted_pairs_idx = np.argsort(min_dtw_pairs)
            pairs = pairs[sorted_pairs_idx]
            min_dtw_pairs = min_dtw_pairs[sorted_pairs_idx]

            # Find the best pairs
            res = dtw_pairs_loop_no_bound_with_region(
                pairs,
                # min_dtw_pairs,
                x_window_size,
                y_window_size,
                cost_mat,
                top_k,
                region
            )

            res = np.array(res, dtype=np.float64)
            candidates = res[:, 1:].astype(np.int32)
            min_dtw = res[:, 0]
            min_dtw = self.dtw_distance.invert(min_dtw)
            min_dtw_idx = np.argsort(min_dtw)

            return candidates[min_dtw_idx], min_dtw[min_dtw_idx]


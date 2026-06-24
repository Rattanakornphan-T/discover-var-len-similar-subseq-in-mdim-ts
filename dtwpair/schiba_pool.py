import numpy as np
from typing import Tuple, Callable
from .utils.pooling import region_max_pooling
from .base import BaseDtwPair, dtw_pairs_loop_with_region
from pyts.metrics.dtw import _compute_region
from .utils.dtw_distance import BaseDtwDistance







class SchibaLbkPoolDtwPair(BaseDtwPair):
    def __init__(self,
                 cost_function: Callable=None,
                 dtw_distance: BaseDtwDistance=None,
                 window_size: int=0.1):
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
                         dtw_distance=dtw_distance)
        self.window_size = window_size

    def approximate_dtw_pair(self,
                            cost_mat: np.ndarray,
                            x_window_size: int,
                            y_window_size: int,
                            region: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        max_pooled = region_max_pooling(cost_mat, x_window_size, y_window_size, region)   # (M - x_window_size + 1, N - y_window_size + 1)
        min_pooled = -region_max_pooling(-cost_mat, x_window_size, y_window_size, region) # (M - x_window_size + 1, N - y_window_size + 1)

        max_dtw = max_pooled * (x_window_size + y_window_size - 1) # Walk at the edge
        min_dtw = min_pooled * max(x_window_size, y_window_size) # Walk diagonally

        return max_dtw, min_dtw # [ (M - x_window_size + 1, N - y_window_size + 1) ] * 2

    
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

        # Approximate DTW of each subsequences pair
        max_dtw_pairs, min_dtw_pairs = self.approximate_dtw_pair(cost_mat, x_window_size, y_window_size, region)
        # ( (M - x_window_size + 1, N - y_window_size + 1), ) * 2


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


        # Filter only the candidate pairs
        filter = min_dtw_pairs <= min_dtw
        if sum(filter) >= top_k:
            pairs = pairs[filter]
            max_dtw_pairs = max_dtw_pairs[filter]
            min_dtw_pairs = min_dtw_pairs[filter]

        # Sort the pairs by lower bound
        sorted_pairs_idx = np.argsort(min_dtw_pairs)
        pairs = pairs[sorted_pairs_idx]
        min_dtw_pairs = min_dtw_pairs[sorted_pairs_idx]

        
        # Find the best pairs
        res = dtw_pairs_loop_with_region(
            pairs,
            min_dtw_pairs,
            x_window_size,
            y_window_size,
            cost_mat,
            top_k,
            region
        )

        res = np.array(res, dtype=np.float64)
        candidates = res[:, 1:].astype(np.int32)
        min_dtw_idx = np.argsort(min_dtw)

        return candidates[min_dtw_idx], min_dtw[min_dtw_idx]






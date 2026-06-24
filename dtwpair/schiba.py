import numpy as np
from .base import BaseDtwPair, dtw_pairs_loop_no_bound_with_region
from pyts.metrics import dtw
from .utils.dtw_distance import BaseDtwDistance
from typing import Callable, Tuple
from pyts.metrics.dtw import _compute_region





class SakoeChibaDtwPair(BaseDtwPair):
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
    

    def compute_dtw(self,
                    x: np.ndarray,
                    y: np.ndarray,
                    cost_mat: np.ndarray,
                    x_slice: slice,
                    y_slice: slice) -> float:
        _x = x[x_slice]
        _y = y[y_slice]
        return self.dtw_distance.invert(
            dtw(_x, _y,
                method='sakoechiba',
                dist='precomputed',
                precomputed_cost=cost_mat[x_slice, y_slice],
                options={
                    'window_size': self.window_size,
                })
        )

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

        # Flatten the DTW aproximation matrix
        pairs = self.get_index_pairs(
            len(x) - x_window_size + 1,
            len(y) - y_window_size + 1
        ) # ((M - x_window_size + 1), (N - y_window_size + 1), 2)
        pairs = pairs.reshape(-1, 2) # ((M - x_window_size + 1) * (N - y_window_size + 1), 2)
        
        region = _compute_region(x_window_size, y_window_size, "sakoechiba",
                                 "precomputed", window_size=self.window_size)

        res = dtw_pairs_loop_no_bound_with_region(
            pairs,
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

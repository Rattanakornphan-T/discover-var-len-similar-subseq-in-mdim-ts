import numpy as np
from typing import Tuple
from .utils.pooling import max_pooling
from .base import BaseDtwPair, dtw_pairs_loop
from .utils.timer import Timer







class PoolDtwPair(BaseDtwPair):

    def approximate_dtw_pair(self,
                            cost_mat: np.ndarray, # (M, N)
                            x_window_size: int,
                            y_window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.verbose:
            print("Computing max_dtw")
        with Timer(self.verbose):
            max_pooled = max_pooling(cost_mat, x_window_size, y_window_size)   # (M - x_window_size + 1, N - y_window_size + 1)
            max_dtw = max_pooled * (x_window_size + y_window_size - 1) # Walk at the edge

        if self.verbose:
            print("Computing min_dtw")
        with Timer(self.verbose):
            min_pooled = -max_pooling(-cost_mat, x_window_size, y_window_size) # (M - x_window_size + 1, N - y_window_size + 1)
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

        # Approximate DTW of each subsequences pair
        max_dtw_pairs, min_dtw_pairs = self.approximate_dtw_pair(cost_mat, x_window_size, y_window_size)
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
            res = dtw_pairs_loop(
                pairs,
                min_dtw_pairs,
                x_window_size,
                y_window_size,
                cost_mat,
                top_k
            )

            res = np.array(res, dtype=np.float64)
            candidates = res[:, 1:].astype(np.int32)
            min_dtw = res[:, 0]
            min_dtw = self.dtw_distance.invert(min_dtw)
            min_dtw_idx = np.argsort(min_dtw)

            return candidates[min_dtw_idx], min_dtw[min_dtw_idx]




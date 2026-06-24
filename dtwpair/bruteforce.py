import numpy as np
from .base import BaseDtwPair, dtw_pairs_loop_no_bound





class BruteforceDtwPair(BaseDtwPair):
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
        
        x, y = self.validate_input(x, y, x_window_size, y_window_size) # (N, D), (M, D)

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

        # Find the best pairs
        res = dtw_pairs_loop_no_bound(
            pairs,
            x_window_size,
            y_window_size,
            cost_mat,
            top_k
        )

        res = np.array(res, dtype=np.float64)
        candidates = res[:, 1:].astype(np.int32)
        min_dtw = res[:, 0]

        return candidates, self.dtw_distance.invert(min_dtw)

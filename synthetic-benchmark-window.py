from dtwpair import BaseDtwPair
from dtwpair import BruteforceDtwPair
from dtwpair import SlidingWindowDtwPair
from dtwpair import SakoeChibaDtwPair
from dtwpair import SchibaSlidingWindowDtwPair
import numpy as np
import matplotlib.pyplot as plt
from dtwpair.utils.dtw_distance import AbsoluteDtwDistance, SquaredDtwDistance
from dtwpair.utils.timer import Timer
import pickle

from tqdm import tqdm
import os




def generate_time_series(length,
                         moving_average=20,
                         seed=0,
                         mean=0,
                         std=2,
                         num_motifs=8,
                         motif_mean=0,
                         motif_length=(70, 70),
                         motif_amplitude=(1, 1),
                         motif_frequency=(1/70, 1/70),
                         motif_weight=0.9,
                         return_motifs=False):
    """
    Generate two time series with a known DTW distance.
    """
    # motif_length = int(round(motif_length[0] * length)), int(round(motif_length[1] * length))
    # motif_frequency = motif_frequency[0] * length, motif_frequency[1] * length

    np.random.seed(seed)

    x = np.random.normal(0, std, length)
    x = np.convolve(x, np.ones(moving_average)/moving_average, mode='same') + mean

    # Insert the motif at random positions
    positions = np.random.choice(np.arange(length - motif_length[1] + 1), num_motifs, replace=False)
    
    mtx = []

    for pos in positions:
        length = np.random.randint(motif_length[0], motif_length[1]+1)
        amplitude = np.random.uniform(motif_amplitude[0], motif_amplitude[1])
        frequency = np.random.uniform(motif_frequency[0], motif_frequency[1])
        # phase = np.random.uniform(0, length - 1)
        phase = 0
        motif = amplitude * np.sin((np.arange(length) + phase) * 2 * np.pi * frequency) + motif_mean

        x[pos:pos + length] = (1- motif_weight) * x[pos:pos + length] + motif_weight * motif

        mtx.append((pos, pos+length, x[pos:pos + length], motif))
    
    if return_motifs:
        return x, mtx
    else:
        return x




def benchmark_dtw_pair(dtwp: BaseDtwPair, x, y, w):
    """
    Benchmark the DTW pair algorithm.
    """
    with Timer() as t:
        res = dtwp.subsequence_nearest_neighbour(x, y, w, w)
        elapsed, elapsed_process = t.stop()
    return {
        'res': res,
        'elapsed': elapsed,
        'elapsed_process': elapsed_process,
    }


length = 1000

for trial in range(10):
    result = []
    for win_size in range(10, 91, 10):
        if os.path.exists(f'benchmark3w/results_{trial}.pkl'):
            with open(f'benchmark3w/results_{trial}.pkl', 'rb') as f:
                result = pickle.load(f)
            if win_size in [r[0] for r in result]:
                print(f'Already run win_size {win_size} for trial {trial}')
                continue

        win_size_steps = int(length * win_size / 100)
        print(f'Running trial {trial} for win_size {win_size}')
        data = (
            generate_time_series(length, num_motifs=1, seed=0,
                                 motif_length=(win_size_steps, win_size_steps),
                                 motif_frequency=(1/win_size_steps, 1/win_size_steps)),
            generate_time_series(length, num_motifs=1, seed=1,
                                 motif_length=(win_size_steps, win_size_steps),
                                 motif_frequency=(1/win_size_steps, 1/win_size_steps))
        )
        res = {}
        for dtwp in tqdm([
            BruteforceDtwPair(dtw_distance=SquaredDtwDistance(), verbose=False),
            SlidingWindowDtwPair(dtw_distance=SquaredDtwDistance(), verbose=False),
            SakoeChibaDtwPair(dtw_distance=SquaredDtwDistance(), window_size=0.1, verbose=False),
            SchibaSlidingWindowDtwPair(dtw_distance=SquaredDtwDistance(), window_size=0.1, verbose=False),
        ]):
            res[dtwp.__class__.__name__] = benchmark_dtw_pair(dtwp, data[0], data[1],
                                                              win_size_steps)
        result.append((win_size, res, data))

        with open(f'benchmark3w/results_{trial}.pkl', 'wb') as f:
            pickle.dump(result, f)



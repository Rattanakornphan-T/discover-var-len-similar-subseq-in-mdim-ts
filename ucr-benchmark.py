import argparse
import json
import time
import math
import statistics
import random
from typing import Any, Dict, List, Tuple
from pathlib import Path
import pandas as pd
from baselines import ucr_dtw_two_sub, modified_swamp
from dtwpair.utils.dtw_distance import SquaredDtwDistance
import dtwpair

def return_ts_names(path):
    ts_names = []
    root = Path(path)

    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = subdir.name
        train = subdir / f"{name}_TRAIN.tsv"
        test  = subdir / f"{name}_TEST.tsv"

        if train.exists():
            ts_names.append((name, train.as_posix()))
        if test.exists():
            ts_names.append((name, test.as_posix()))

    return ts_names

def read_in_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None)
    return df.iloc[:, 1:] # Remove the target/class column (the first column)

def benchmarking(fn, num_repeat=5):  
    fn() # warmup

    repeat = []
    for _ in range(num_repeat):
        t0 = time.perf_counter()
        fn()
        repeat.append(time.perf_counter() - t0)

    return statistics.median(repeat)

def run(UCRArchive_filepath, output_filepath, random_seed=1):
    ts_names = return_ts_names(UCRArchive_filepath)
    dtwp = dtwpair.SlidingWindowDtwPair(dtw_distance=SquaredDtwDistance(), verbose=False)

    results = {'ucr': [], 'swamp': [], 'ours': []}

    random.seed(random_seed)
    for _ in range(1000):
        ts = random.choices(ts_names, k=2)
        ts1 = read_in_data(ts[0][1]).sample(n=1, random_state=2).iloc[0].dropna().to_numpy() # Fix random_state to fix random seed
        ts2 = read_in_data(ts[1][1]).sample(n=1, random_state=3).iloc[0].dropna().to_numpy()

        subseq_len = random.randint(4, min(ts1.size, ts2.size))

        ucr = benchmarking(lambda: ucr_dtw_two_sub.find_closest_subsequences(
            ts1, ts2,
            subseq_len=subseq_len,            
            engine="numba"
        ))
        swamp = benchmarking(lambda: modified_swamp.swamp_ab_top1(ts1, ts2, subseqlen=subseq_len))
        ours = benchmarking(lambda: dtwp.subsequence_nearest_neighbour(ts1, ts2, subseq_len, subseq_len))

        results['ucr'].append(ucr)
        results['swamp'].append(swamp)
        results['ours'].append(ours)

    with open(output_filepath, 'w') as f:
        json.dump(results, f)

    return results

# ----------- Process Results -----------

def geometric_mean(values: List[float]) -> float:
    """Geometric mean via logs (stable). Requires all values > 0."""
    if not values:
        raise ValueError("geometric_mean() got an empty list")
    log_sum = 0.0
    for v in values:
        if v <= 0:
            raise ValueError(f"Geometric mean requires positive values, got {v}")
        log_sum += math.log(v)
    return math.exp(log_sum / len(values))

def median_and_iqr(values: List[float]) -> Tuple[float, float]:
    """Median and Interquartile range (IQR)"""
    if not values:
        raise ValueError("median_and_iqr() got an empty list")
    q1, q2, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return (q2, q3 - q1)

def relative_times_and_statistics(
    combined: Dict[str, List[Any]],
    *,
    skip_if_min_nonpositive: bool = True,
) -> Dict[str, Any]:
    ucr = combined["ucr"]
    swamp = combined["swamp"]
    ours = combined["ours"]

    if not (len(ucr) == len(swamp) == len(ours)):
        raise ValueError(f"Lengths differ: ucr={len(ucr)}, swamp={len(swamp)}, ours={len(ours)}")

    t_ucr: List[float] = []
    t_swamp: List[float] = []
    t_ours: List[float] = []

    skipped = 0
    for i, (a, b, c) in enumerate(zip(ucr, swamp, ours)):
        m = min(a, b, c)
        if m <= 0:
            if skip_if_min_nonpositive:
                skipped += 1
                continue
            raise ValueError(f"Index {i}: min(ucr, swamp, ours) = {m} (must be > 0)")

        t_ucr.append(a / m)
        t_swamp.append(b / m)
        t_ours.append(c / m)
    
    median_ucr, iqr_ucr = median_and_iqr(t_ucr)
    median_swamp, iqr_swamp = median_and_iqr(t_swamp)
    median_ours, iqr_ours = median_and_iqr(t_ours)

    return {
        "t_ucr": t_ucr,
        "t_swamp": t_swamp,
        "t_ours": t_ours,
        "gmean_t_ucr": geometric_mean(t_ucr),
        "median_t_ucr": median_ucr,
        "iqr_t_ucr": iqr_ucr,
        "gmean_t_swamp": geometric_mean(t_swamp),
        "median_t_swamp": median_swamp,
        "iqr_t_swamp": iqr_swamp,
        "gmean_t_ours": geometric_mean(t_ours),
        "median_t_ours": median_ours,
        "iqr_t_ours": iqr_ours,
        "n_total": len(ucr),
        "n_used": len(t_ucr),
        "n_skipped": skipped,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("UCRArchive_filepath", type=str)
    parser.add_argument("output_filepath", type=str, default="./results.json", nargs="?")
    parser.add_argument("random_seed", type=int, default=1, nargs="?")

    args = parser.parse_args()
    print("UCRArchive_filepath:", args.UCRArchive_filepath)
    print("output_filepath:", args.output_filepath)
    print("random_seed:", args.random_seed)

    results = run(args.UCRArchive_filepath, args.output_filepath, args.random_seed)

    stats = relative_times_and_statistics(results)
    print("Adapted UCR Suite, Adapted SWAMP, Our method")
    print("Geometric mean:", stats["gmean_t_ucr"], stats["gmean_t_swamp"], stats["gmean_t_ours"])
    print("Median:", stats["median_t_ucr"], stats["median_t_swamp"], stats["median_t_ours"])
    print("IQR:", stats["iqr_t_ucr"], stats["iqr_t_swamp"], stats["iqr_t_ours"])

if __name__ == "__main__":
    main()
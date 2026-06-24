# Inferring the Most Similar Variable-length Subsequences between Multidimensional Time Series

This project implements a framework for identifying the most similar variable-length subsequences between two multidimensional time series. Traditional similarity measures typically rely on fixed-length subsequences and may not perform well when the optimal matching segments differ in length or exhibit complex alignments. To address this, the proposed method leverages a two-step strategy combining (1) coarse candidate discovery using dimensionality reduction and correlation-based pre-filtering, and (2) fine-grained similarity estimation through alignment and adaptive windowing.

The framework supports high robustness to noise, scalability across time series dimensions and lengths, and flexible matching that allows discovering semantically meaningful patterns. It is evaluated using both synthetic and real-world datasets, demonstrating superior performance over baseline methods in terms of both accuracy and computational efficiency.

This implementation enables researchers and practitioners to compare time series data at a granular level and uncover relevant patterns without requiring prior knowledge of their exact location or duration.


## Environment Setup
1. **Install Python**
    Make sure you have Python 3.11 or higher installed. You can download it from [python.org](https://www.python.org/downloads/).
    
    Alternatively, you can use Anaconda, which comes with Python and many useful packages pre-installed. Download it from [anaconda.com](https://www.anaconda.com/products/distribution).

2. **Create a Python environment (Optional but)**  
    You can use `venv`, `conda`, or any other environment manager of your choice.
    For example, using `venv`:
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```
    Or using `conda`:
    ```bash
    conda create -n myenv python=3.11
    conda activate myenv
    ```

3. **Install dependencies**  
    ```
    pip install -U -r requirements.txt
    ```

## Usage

Example: Find the best subsequences match between S&P500 and NASDAQ index.

```python
from dtwpair import SlidingWindowDtwPair
from dtwpair import SchibaSlidingWindowDtwPair
import numpy as np
import matplotlib.pyplot as plt
from dtwpair.utils.dtw_distance import AbsoluteDtwDistance, SquaredDtwDistance
from dtwpair.utils.timer import Timer

import yfinance as yf


# Download stock price data
stock_price = yf.download(['^GSPC', '^IXIC'], start='2000-01-01', end='2025-01-01')['Close']
stock_price = np.log(stock_price)
stock_price = (stock_price - stock_price.rolling(window=365).mean()) / stock_price.rolling(window=365).std()
stock_price = stock_price.dropna()


# Usage of SlidingWindowDtwPair
dtwp = SlidingWindowDtwPair(dtw_distance=SquaredDtwDistance())
pairs, min_dtw = dtwp.subsequence_nearest_neighbour(x, y, 30, 40)

```

Parameters:
- **cost_function**
    
    Cost function to calculate cost matrix which is `euclidean_distance` or `manhattan_distance` function from `utils.cost`.

- **x, y**

    numpy.ndarray time series with shape `(Timesteps,)` or `(Timesteps, Dimensions)`.

- **x_window_size, y_window_size**

    Lengths of the subsequences of time series `x` and `y` to compare.

- top_k (default = 1):
    
    Number of most similar subsequence pairs to return (return starting indices of subsequences).

- stride (default = 1):

    Number of subsequences to be skipped calculating. stride = `n` means skipping `n-1` subsequences for every subsequence (of both time series) we calculate.

- allow_temporal_shift (default = -1):
    
    How far two subsequences in a pair can be further apart. `-1` means no constraint is applied. 


## Running Synthetic Benchmarks

- **Time Series Length vs Runtime**  
  Run:
  ```
  mkdir -p benchmark3
  python synthetic-benchmark.py
  ```

- **Window Size vs Runtime**  
  Run:
  ```
  mkdir -p benchmark3w
  python synthetic-benchmark-window.py
  ```

- **Noise Level vs Runtime and F1 Score**  
  Run:
  ```
  mkdir -p benchmark3r
  python synthetic-benchmark-robust.py
  ```

## Visualizing Results

To plot and analyze the synthetic benchmark results:

1. Open `visualize-synthetic-benchmark-all.ipynb` in Jupyter Notebook.
2. Select your Python kernel.
3. Run all cells to generate the plots.

## Running UCR Archive Benchmark

To reproduce the UCR Time Series Archive benchmark results, run:
  ```
  python ucr-benchmark.py <UCRArchive_filepath> <output_filepath>
  ```
<UCRArchive_filepath>: Path to the extracted UCR Time Series Archive directory.
Download from the UCR site: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

<output_filepath>: Path where the benchmark results will be written (e.g., "./results.json").

## Running Baboon Result

To make heatmaps of baboons' lead difference:

1. Open `baboon-results.ipynb` in Jupyter Notebook.
2. Select your Python kernel.
3. Run all cells to generate the heatmaps.

(Baboon dataset is in `dataset/baboon`)

## Running Stock Result

To make plots of stock prices:

1. Open `stock-results.ipynb` in Jupyter Notebook.
2. Select your Python kernel.
3. Run all cells to generate the plots.

(Stock price dataset will be automatically downloaded from Yahoo Finance.)

---

For more details, please refer to the original paper and the code documentation.

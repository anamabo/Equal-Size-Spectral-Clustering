![Logo](https://raw.githubusercontent.com/anamabo/Equal-Size-Spectral-Clustering/main/images/logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# ELSARA: EquaL-size SpectrAl clusteRing Algorithm
This is a modification of the [spectral clustering algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html) that builds clusters balanced 
in the number of points. A detailed explanation of the model can be found 
[in this Medium blog post](https://medium.com/data-science/equal-size-spectral-clustering-cce65c6f9ba3).

## Installation

Installation occurs through pip:

`pip install elsara`

## Toy datasets
In the folder `datasets` we have provided you with a toy dataset
so you can run the clustering code right away. 

* *restaurants_in_amsterdam.csv:* A table with locations of restaurants in the city of Amsterdam
*  *symmetric_distr_tr.npy:* A file with the travel distance between the restaurants

You can find more specification on how to use these datasets in the project's [blog post](https://medium.com/data-science/equal-size-spectral-clustering-cce65c6f9ba3). 

## Examples
* *example1.py:* From a set of hyperparameters, you obtain clusters with sizes roughly equal to N / `nclusters`  
* *example2.py:* From a range of cluster sizes, you obtain the clusters hyperparameters to run the clustering code. 

## Usage

### Example 1: fixed hyperparameters

Provide `nclusters`, `nneighbors`, and `equity_fraction` directly. Each cluster will contain roughly `N / nclusters` points.

```python
import pandas as pd
import numpy as np
from elsara import SpectralEqualSizeClustering, visualise_clusters

# coords is used only for visualization
coords = pd.read_csv("datasets/restaurants_in_amsterdam.csv")
dist_tr = np.load("datasets/symmetric_dist_tr.npy")

clustering = SpectralEqualSizeClustering(
    nclusters=6, nneighbors=int(dist_tr.shape[0] * 0.1), equity_fraction=1, seed=1234
)

labels = clustering.fit(dist_tr)

coords["cluster"] = labels
clusters_figure = visualise_clusters(
    coords,
    longitude_colname="longitude",
    latitude_colname="latitude",
    label_col="cluster",
    zoom=11,
)
clusters_figure.show()
```

### Example 2: derive hyperparameters from a target size range

Specify the desired min/max cluster size and let the algorithm derive the hyperparameters automatically.

```python
import pandas as pd
import numpy as np
from elsara import SpectralEqualSizeClustering, visualise_clusters

coords = pd.read_csv("datasets/restaurants_in_amsterdam.csv")
dist_tr = np.load("datasets/symmetric_dist_tr.npy")

min_range, max_range = 50, 70  # desired number of points per cluster

npoints = coords.shape[0]
avg_range = (max_range + min_range) / 2.0
nclusters = int(npoints / avg_range)
equity_fraction = 1 - ((avg_range - min_range) / avg_range)
nneighbors = int(npoints * (avg_range / npoints))

clustering = SpectralEqualSizeClustering(
    nclusters=nclusters, nneighbors=nneighbors, equity_fraction=equity_fraction, seed=1234
)

labels = clustering.fit(dist_tr)

coords["cluster"] = labels
clusters_figure = visualise_clusters(
    coords,
    longitude_colname="longitude",
    latitude_colname="latitude",
    label_col="cluster",
    zoom=11,
)
clusters_figure.show()
```

# How to contribute

## Prerequisities
* Python 3.13 
* Poetry (in MAC: `brew install poetry`)

## Setup

Install dependencies and register the git hooks:

```bash
poetry install
make install-hooks
```

## Code formatting

This project uses [ruff](https://docs.astral.sh/ruff/) and [black](https://black.readthedocs.io/) for code formatting. To format all files manually, run:

```bash
make format
```

Formatting also runs automatically on every commit via [pre-commit](https://pre-commit.com/).

## PR creation

1. Create a branch using one of the following prefixes and open a PR to `main`:
   - `fix/` — bug fix, bumps the patch version (e.g. 0.2.0 → 0.2.1)
   - `feature/` — new feature, bumps the minor version (e.g. 0.2.0 → 0.3.0)
   - `breaking/` — breaking change, bumps the major version (e.g. 0.2.0 → 1.0.0)
   
**Important:** If the branch name doesn't follow this name convention, won't be accepted, as it won't update the package in PyPi.
   
2. Create a PR to be reviewed by anamabo. She will suggest changes or approve your PR. In the later case, a new version of ELSARA is published to PyPI automatically.

## License
This project is licensed under the [MIT License](LICENSE).

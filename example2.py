"""
This script shows how to get a set of hyperparameters based on
a range of desired cluster sizes. This is contrary to what is in example1.py
"""
import pandas as pd
import numpy as np
import logging
from source_code.spectral_equal_size_clustering import SpectralEqualSizeClustering
from source_code.visualisation import visualise_clusters

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# read the file with coordinates. This file is used only for visualization purposes
coords = pd.read_csv("datasets/restaurants_in_amsterdam.csv")

# read the file of the symmetric distance matrix associated to the coords data frame
dist_tr = np.load("datasets/symmetric_dist_tr.npy")

min_range, max_range = 50, 70  # desired number of points per cluster

# Get the cluster hyperparameters
npoints = coords.shape[0]
avg_range = (max_range + min_range)/2.
nclusters = int(npoints / avg_range)
eq_fr = 1 - ((avg_range - min_range)/avg_range)
nn_fr = avg_range / npoints
nneighbors = int(npoints*nn_fr)

logging.info(f"The hyperparameters are: nclusters={nclusters}, nneighbors={nneighbors}, equity_fraction={eq_fr}")

clustering = SpectralEqualSizeClustering(nclusters=nclusters,
                                         nneighbors=nneighbors,
                                         equity_fraction=eq_fr,
                                         seed=1234)

labels = clustering.fit(dist_tr)

coords["cluster"] = labels
logging.info(f"Points per cluster: \n {coords.cluster.value_counts()}")
clusters_figure = visualise_clusters(coords,
                                     longitude_colname="longitude",
                                     latitude_colname="latitude",
                                     label_col="cluster",
                                     zoom=11)
clusters_figure.show()

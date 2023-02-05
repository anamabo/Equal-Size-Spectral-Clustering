import pandas as pd
import numpy as np
from source_code.spectral_esc import SpectralEqualSizeClustering
from source_code.visualisation import visualise_clusters

# read the file with coordinates. This file is used only for visualization purposes
coords = pd.read_csv("datasets/restaurants_in_amsterdam.csv")

# read the file of the symmetric distance matrix associated to the coords data frame
dist_tr = np.load("datasets/symmetric_dist_tr.npy")

clustering = SpectralEqualSizeClustering(nclusters=6,
                                         nneighbors=int(dist_tr.shape[0] * 0.1),
                                         equity_fraction=1,
                                         seed=1234)

labels = clustering.fit(dist_tr)

coords["cluster"] = labels
print(coords.cluster.value_counts())
clusters_figure = visualise_clusters(coords,
                                     longitude_colname="longitude",
                                     latitude_colname="latitude",
                                     label_col="cluster",
                                     zoom=11)
clusters_figure.show()

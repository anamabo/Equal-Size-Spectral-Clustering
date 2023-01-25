import pandas as pd
import numpy as np
from source_code.spectral_esc import SpectralEqualSizeClustering
from source_code.visualisation import visualise_clusters

# read file with coordinates
coords = pd.read_csv("datasets/restaurants_in_amsterdam.csv")

# read the distance matrix file associated to the coordinates
dist_tr = np.load("datasets/symmetric_dist_tr.npy")  # symmetrize TM. Ojo, poner esto en el blog!

clustering = SpectralEqualSizeClustering(nclusters=6,
                                         nneighbors=int(dist_tr.shape[0] * 0.1),
                                         equity_fr=0.9,
                                         seed=1234
                                         )

labels = clustering.fit(dist_tr)
coords["cluster"] = labels

print(coords.cluster.value_counts())
clusters_figure = visualise_clusters(
    coords,
    longitude_colname="longitude",
    latitude_colname="latitude",
    label_col="cluster",
    zoom=11,
)
clusters_figure.show()

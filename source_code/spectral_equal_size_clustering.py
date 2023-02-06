"""
Module containing the Spectral Equal Size Clustering method
"""
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
import logging
import math

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class SpectralEqualSizeClustering:
    """
    Uses spectral clustering to obtain an initial configuration of clusters.
    This configuration is compact but NOT equal-sized. To make clusters equal-sized (in number of points),
    we use the method cluster_equalization().
    Input parameters:
        nclusters (int): number of clusters
        nneighbors (int): number of neighbors. Used by the spectral clustering to
                          construct the affinity matrix. Good values are between 7% and 15%
                          of the dataset points.
        equity_fraction (float): Equity fraction. Value in range (0,1] which decides how equal the clusters
                           could be. The higher the fraction, the more equal the clusters BUT the less
                           compact.
        seed (int): Random seed generator.

    Attributes:
        first_clustering (data frame): Table containing the cluster labels of each point in the initialisation.
        first_cluster_dispersion (data frame): A table with indexes corresponding to each cluster and a column
                                containing the dispersion in distance of each cluster.
        first_total_cluster_dispersion (float): sum of first_cluster_dispersion
        final_clustering  (data frame): Table containing the cluster labels of each point after the balancing
                                        of the clusters in size.
        final_cluster_dispersion (data frame): A table with indexes corresponding to each cluster and a column
                                containing the dispersion in distance of each cluster (after the balancing in size).
        total_cluster_dispersion (float): sum of final_cluster_dispersion.
                                 This attribute can be used as a metric to optimise the cluster hyperparameters.


    How to use this class:
    cl = SpectralEqualSizeClustering(nclusters=2, nneighbors=100, equity_fraction=0.5, seed=11362)
    cluster_labels = cl.fit(dm)
    """

    def __init__(self, nclusters: int = None, nneighbors: int = None, equity_fraction=0.3, seed=None):
        self.nclusters = nclusters
        self.equity_fr = equity_fraction
        self.nneighbors = nneighbors
        self.seed = seed

        self.first_clustering = None
        self.first_cluster_dispersion = None  # A table with each cluster dispersion (in distance)
        self.first_total_cluster_dispersion = None  # total cluster dispersion.

        self.range_points = None
        self.nn_df = None  # table of number of neighbors per point
        self.cneighbors = None  # Dictionary of cluster neighbors

        # Final results after equalization of clusters
        self.final_clustering = None
        self.final_cluster_dispersion = None
        self.total_cluster_dispersion = None

    @staticmethod
    def _cluster_dispersion(dist_matrix, clusters):
        """
        Function that computes the cluster dispersion. The cluster dispersion is defined
        as the standard deviation in distance of all the elements within a cluster. The sum of the cluster dispersion
        of all the clusters in a dataset is called the total cluster dispersion. The lower the cluster dispersion,
        the more compact the clusters are.
        Inputs:
        dist_matrix: numpy array of the distance matrix
        clusters: table with cluster labels of each event. columns: 'label', index: points
        """

        if "label" not in clusters.columns:
            raise ValueError("Table of clusters does not have 'label' column.")

        def std_distances(points, dm):
            distances = np.tril(dm[np.ix_(points, points)])
            distances[distances == 0] = np.nan
            cdispersion = np.nanstd(distances)
            return cdispersion

        nclusters = clusters["label"].nunique()
        points_per_cluster = [list(clusters[clusters.label == cluster].index) for cluster in range(nclusters)]
        wcsdist = [std_distances(points_per_cluster[cluster], dist_matrix) for cluster in range(nclusters)]
        cluster_dispersion_df = pd.DataFrame(wcsdist, index=np.arange(nclusters), columns=["cdispersion"])
        return cluster_dispersion_df

    @staticmethod
    def _optimal_cluster_sizes(nclusters, npoints):
        """
        Gives the optimal number of points in each cluster.
        For instance,  if we have 11 points, and we want 3 clusters,
        2 clusters will have 4 points and one cluster, 3.
        """
        min_points, max_points = math.floor(npoints / float(nclusters)), math.floor(npoints / float(nclusters)) + 1
        number_clusters_with_max_points = npoints % nclusters
        number_clusters_with_min_points = nclusters - number_clusters_with_max_points

        list1 = list(max_points * np.ones(number_clusters_with_max_points).astype(int))
        list2 = list(min_points * np.ones(number_clusters_with_min_points).astype(int))
        return list1 + list2

    @staticmethod
    def get_nneighbours_per_point(dm, nneighbors):
        """
        Computes the number of neighbours of each point.
        IMPORTANT:  I do not consider the point it self as neighbour.
                    This assumption is important so don't change it!
        """
        npoints = dm.shape[0]
        nn_data = [[p, list(pd.Series(dm[:, p]).sort_values().index[1: nneighbors])] for p in range(0, npoints)]
        nn_data = pd.DataFrame(nn_data, columns=["index", "nn"]).set_index("index")
        return nn_data

    def _get_cluster_neighbors(self, df):
        """
        Function to find the cluster neighbors of each cluster.
        The cluster neighbors are selected based on a smaller number of neighbours
        because I don't want to get no neighboring clusters.
        The minimun number of nn to get cluster neighbors is 30. This choice is arbitrary.
        Imputs:
            df: a table with points as index and a "label" column
        Returns:
            A dictionary of shape: {i: [neighbor clusters]}, i= 0,..,nclusters
        """
        if self.nn_df is None:
            raise Exception(
                "Nearest neighbour table not found. Use self.get_nneighbours_per_point(distmat, nneighbors)")

        def cluster_neighbor_for_point(nn_list, nneighbours):
            nn_labels = df1.loc[nn_list[0:nneighbours], 'label']
            return np.unique(nn_labels)

        df1 = df.copy()
        df1 = pd.merge(df1, self.nn_df, left_index=True, right_index=True)
        nn = min(30, self.nneighbors)  # nearest neighbours to compute border points: this def is arbitrary
        df1["unique_clusters"] = df1.apply(lambda row: cluster_neighbor_for_point(row["nn"], nn), axis=1)

        temp = df1[["unique_clusters"]]
        # get neighbor clusters (remove own cluster)
        neighbors = {}
        for c in range(self.nclusters):
            points_in_cluster = df1.label == c
            neighbors_in_cluster = temp.loc[points_in_cluster, "unique_clusters"].to_list()
            neighbors[c] = {i for l in neighbors_in_cluster for i in l if i != c}
        return neighbors

    @staticmethod
    def _get_clusters_outside_range(clustering, minr, maxr):
        """
        Function to get clusters outside the min_range, max_range
        Input: clustering: table with idx as points, and a "label" column
        """
        csizes = clustering.label.value_counts().reset_index()
        csizes.columns = ["cluster", "npoints"]

        large_c = list(csizes[csizes.npoints > maxr]["cluster"].values)
        small_c = list(csizes[csizes.npoints < minr]["cluster"].values)

        return large_c, small_c

    @staticmethod
    def _get_no_large_clusters(clustering, maxr):
        """
        Function to get clusters smaller than max_range
        Input: clustering: table with idx as points, and a "label" column
        """
        csizes = clustering.label.value_counts().reset_index()
        csizes.columns = ["cluster", "npoints"]

        return list(csizes[(csizes.npoints < maxr)]["cluster"].values)

    @staticmethod
    def _get_points_to_switch(dmatrix, cl_elements, clusters_to_modify, idxc):
        """
        Function to obtain the closest distance of points in cl_elements with respect to the clusters in
        clusters_to_modify
        Inputs:
            dmatrix: distance matrix
            cl_elements: list of points of the cluster(s) that give points
            cluster_to_modify: a list of labels of clusters that receive points.
            idxc: dictionary with keys clusters_to_modify and values the points of these clusters, ex:
                  {'0': [idx1, idx2,...,idxn]}
        Returns:
            A table with the closest distance of points in clabel to clusters in
            clusters_to_modify
        """
        neighbor_cluster = []
        distances = []
        for point in cl_elements:
            dist = {c: dmatrix[idxc[c], point].mean() for c in clusters_to_modify}  # Instead of min. Worth future inv.
            new_label = min(dist, key=dist.get)  # closest cluster
            neighbor_cluster.append(new_label)
            distances.append(dist[new_label])

        cdistances = pd.DataFrame({"points": cl_elements, "neighbor_c": neighbor_cluster, "distance": distances})
        cdistances = cdistances.sort_values(by="distance", ascending=True).set_index("points")
        return cdistances

    def cluster_initialization(self, dist_matrix):
        """
        Uses Spectral clustering to get initial cluster configurations. These clusters
        are imbalanced.
        """
        # discretize is less sensitive to random initialization.
        initial_clustering = SpectralClustering(n_clusters=self.nclusters,
                                                assign_labels="discretize",
                                                n_neighbors=self.nneighbors,
                                                affinity="precomputed_nearest_neighbors",
                                                random_state=self.seed)
        initial_clustering.fit(dist_matrix)
        initial_labels = initial_clustering.labels_
        self.first_clustering = pd.DataFrame(initial_labels, columns=["label"])
        self.first_cluster_dispersion = self._cluster_dispersion(dist_matrix, self.first_clustering)
        self.first_total_cluster_dispersion = self.first_cluster_dispersion["cdispersion"].sum()

    def cluster_equalization(self, dmatrix):
        """
        Function to equalize the clusters obtained during the initialization.
        clusters larger than max_range will give points while clusters smaller than min_range
        will steal points.
        The results are stored in the attributes: final_clustering; final_wcsd and final_wcss

        Inputs:
            dmatrix: distance matrix associated with the events
        Returns:
            None
        """
        npoints = dmatrix.shape[0]
        elements_per_cluster = self._optimal_cluster_sizes(self.nclusters, npoints)
        min_range = np.array(elements_per_cluster).min() * self.equity_fr
        max_range = np.array(elements_per_cluster).max() * (2 - self.equity_fr)
        self.range_points = (min_range, max_range)
        logging.info(f"ideal elements per cluster: {elements_per_cluster}")
        logging.info(f"min-max range of elements: {min_range}-{max_range}")

        all_clusters = list(np.arange(0, self.nclusters))
        clustering = self.first_clustering.copy()

        large_clusters, small_clusters = self._get_clusters_outside_range(clustering, min_range, max_range)

        if (len(large_clusters) == 0) & (len(small_clusters) == 0):
            self.final_clustering = self.first_clustering.copy()
            self.final_cluster_dispersion = self._cluster_dispersion(dmatrix, self.final_clustering)
            self.total_cluster_dispersion = self.final_cluster_dispersion.sum(axis=0).wcsd

        other_clusters = list(set(all_clusters) - set(large_clusters))  # clusters that receive points
        inx = {c: list(clustering[clustering.label == c].index) for c in other_clusters}

        for clarge in large_clusters:  # make smaller the big clusters
            cl_elements = list(clustering[clustering.label == clarge].index)
            closest_distance = self._get_points_to_switch(dmatrix, cl_elements, other_clusters, inx)

            leftovers = len(cl_elements) - elements_per_cluster[clarge]

            for point in list(closest_distance.index):
                if leftovers <= 0:
                    break

                new_label = closest_distance.loc[point, "neighbor_c"]
                points_new_label = clustering[clustering.label == new_label].shape[0]

                if points_new_label >= max_range:
                    continue

                if new_label in self.cneighbors[clarge]:
                    # "Possible TO DO: recalculate the points_to_switch in case best switches changed."
                    clustering.loc[point, "label"] = new_label
                    leftovers -= 1

            other_clusters = self._get_no_large_clusters(clustering, max_range)
            inx = {c: list(clustering[clustering.label == c].index) for c in other_clusters}

        # update clusters
        large_clusters, small_clusters = self._get_clusters_outside_range(clustering, min_range, max_range)
        clusters_to_steal = list(set(all_clusters) - set(small_clusters))

        if len(small_clusters) == 0:
            self.final_clustering = clustering
            self.final_cluster_dispersion = self._cluster_dispersion(dmatrix, self.final_clustering)
            self.total_cluster_dispersion = self.final_cluster_dispersion.sum(axis=0).wcsd

        else:  # get bigger the small clusters
            cl_elements = list(clustering[clustering.label.isin(clusters_to_steal)].index)
            inx = {c: list(clustering[clustering.label == c].index) for c in small_clusters}
            closest_distance = self._get_points_to_switch(dmatrix, cl_elements, small_clusters, inx)

            needed_points = {c: min_range - clustering[clustering.label == c].shape[0] for c in small_clusters}

            for point in list(closest_distance.index):
                new_label = closest_distance.loc[point, "neighbor_c"]  # cluster that might receive the point
                current_label = clustering.loc[point, "label"]
                points_current_label = clustering[clustering.label == current_label].shape[0]

                if needed_points[new_label] <= 0:
                    break

                if points_current_label <= min_range:
                    continue

                if new_label in self.cneighbors[current_label]:
                    # "Possible TO DO: recalculate the points_to_switch in case best switches changed."
                    clustering.loc[point, "label"] = new_label
                    needed_points[new_label] -= 1

            self.final_clustering = clustering
            self.final_cluster_dispersion = self._cluster_dispersion(dmatrix, self.final_clustering)
            self.total_cluster_dispersion = self.final_cluster_dispersion["cdispersion"].sum(axis=0)

        return None

    def fit(self, dmatrix):
        """
        Main function to carry out the equal size clustering.
        """

        logging.info(
            f"parameters of the cluster: nclusters: {self.nclusters,} "
            f"equity_fr: {self.equity_fr} nneighbours: {self.nneighbors}")

        # number of neighbors and cluster neighbors. They do not change
        self.nn_df = self.get_nneighbours_per_point(dmatrix, self.nneighbors)

        if self.nclusters == np.shape(dmatrix)[0]:
            raise Exception("Number of clusters equal to number of events.")

        if self.nclusters <= 1:
            raise ValueError("Incorrect number of clusters. It should be higher or equal than 2.")

        else:
            self.cluster_initialization(dmatrix)
            self.cneighbors = self._get_cluster_neighbors(self.first_clustering)
            self.cluster_equalization(dmatrix)

        return list(self.final_clustering.label.values)

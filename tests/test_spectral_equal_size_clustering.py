import numpy as np
import pandas as pd
import pytest

from elsara.spectral_equal_size_clustering import SpectralEqualSizeClustering


# ---------------------------------------------------------------------------
# _optimal_cluster_sizes
# ---------------------------------------------------------------------------


class TestOptimalClusterSizes:
    def test_uneven_division(self):
        # 11 points, 3 clusters → 2 clusters of 4, 1 of 3
        result = SpectralEqualSizeClustering._optimal_cluster_sizes(3, 11)
        assert sorted(result, reverse=True) == [4, 4, 3]
        assert len(result) == 3

    def test_even_division(self):
        # 12 points, 3 clusters → all clusters of 4
        result = SpectralEqualSizeClustering._optimal_cluster_sizes(3, 12)
        assert result == [4, 4, 4]

    def test_one_remainder(self):
        # 10 points, 3 clusters → 1 cluster of 4, 2 of 3
        result = SpectralEqualSizeClustering._optimal_cluster_sizes(3, 10)
        assert sorted(result, reverse=True) == [4, 3, 3]

    def test_total_equals_npoints(self):
        for nclusters, npoints in [(5, 23), (7, 100), (4, 16)]:
            result = SpectralEqualSizeClustering._optimal_cluster_sizes(
                nclusters, npoints
            )
            assert sum(result) == npoints

    def test_lengths_match_nclusters(self):
        result = SpectralEqualSizeClustering._optimal_cluster_sizes(6, 20)
        assert len(result) == 6

    def test_sizes_differ_by_at_most_one(self):
        result = SpectralEqualSizeClustering._optimal_cluster_sizes(5, 23)
        assert max(result) - min(result) <= 1

    def test_two_clusters(self):
        result = SpectralEqualSizeClustering._optimal_cluster_sizes(2, 5)
        assert sorted(result, reverse=True) == [3, 2]


# ---------------------------------------------------------------------------
# _cluster_dispersion
# ---------------------------------------------------------------------------


@pytest.fixture
def two_cluster_matrix():
    """
    6-point distance matrix: two well-separated groups of 3.
    Cluster 0 (points 0,1,2): intra-distances 1, 2, 3 → dispersion = std([1,2,3])
    Cluster 1 (points 3,4,5): intra-distances all 1  → dispersion = 0.0
    """
    dm = np.array(
        [
            [0, 1, 2, 99, 99, 99],
            [1, 0, 3, 99, 99, 99],
            [2, 3, 0, 99, 99, 99],
            [99, 99, 99, 0, 1, 1],
            [99, 99, 99, 1, 0, 1],
            [99, 99, 99, 1, 1, 0],
        ],
        dtype=float,
    )
    clusters = pd.DataFrame({"label": [0, 0, 0, 1, 1, 1]})
    return dm, clusters


class TestClusterDispersion:
    def test_output_shape(self, two_cluster_matrix):
        dm, clusters = two_cluster_matrix
        result = SpectralEqualSizeClustering._cluster_dispersion(dm, clusters)
        assert result.shape == (2, 1)
        assert "cdispersion" in result.columns

    def test_uniform_cluster_has_zero_dispersion(self, two_cluster_matrix):
        dm, clusters = two_cluster_matrix
        result = SpectralEqualSizeClustering._cluster_dispersion(dm, clusters)
        assert result.loc[1, "cdispersion"] == pytest.approx(0.0)

    def test_varied_cluster_dispersion(self, two_cluster_matrix):
        dm, clusters = two_cluster_matrix
        result = SpectralEqualSizeClustering._cluster_dispersion(dm, clusters)
        expected = np.std([1.0, 2.0, 3.0])
        assert result.loc[0, "cdispersion"] == pytest.approx(expected)

    def test_missing_label_column_raises(self, two_cluster_matrix):
        dm, _ = two_cluster_matrix
        bad_clusters = pd.DataFrame({"cluster": [0, 0, 0, 1, 1, 1]})
        with pytest.raises(ValueError, match="label"):
            SpectralEqualSizeClustering._cluster_dispersion(dm, bad_clusters)

    def test_returns_non_negative_values(self, two_cluster_matrix):
        dm, clusters = two_cluster_matrix
        result = SpectralEqualSizeClustering._cluster_dispersion(dm, clusters)
        assert (result["cdispersion"] >= 0).all()


# ---------------------------------------------------------------------------
# get_nneighbours_per_point
# ---------------------------------------------------------------------------


class TestGetNNeighboursPerPoint:
    @pytest.fixture
    def simple_dm(self):
        # Point 0 is closest to 1, then 2, then 3
        return np.array(
            [
                [0, 1, 2, 3],
                [1, 0, 4, 5],
                [2, 4, 0, 6],
                [3, 5, 6, 0],
            ],
            dtype=float,
        )

    def test_output_shape(self, simple_dm):
        result = SpectralEqualSizeClustering.get_nneighbours_per_point(simple_dm, 3)
        assert result.shape == (4, 1)
        assert "nn" in result.columns

    def test_self_excluded(self, simple_dm):
        result = SpectralEqualSizeClustering.get_nneighbours_per_point(simple_dm, 3)
        for point in range(4):
            assert point not in result.loc[point, "nn"]

    def test_nearest_neighbor_ordering(self, simple_dm):
        # Point 0: nearest neighbors by distance are 1 (d=1), 2 (d=2)
        result = SpectralEqualSizeClustering.get_nneighbours_per_point(simple_dm, 3)
        assert result.loc[0, "nn"] == [1, 2]

    def test_all_points_have_nn_list(self, simple_dm):
        result = SpectralEqualSizeClustering.get_nneighbours_per_point(simple_dm, 3)
        for point in range(4):
            assert isinstance(result.loc[point, "nn"], list)


# ---------------------------------------------------------------------------
# _get_clusters_outside_range
# ---------------------------------------------------------------------------


class TestGetClustersOutsideRange:
    @pytest.fixture
    def clustering(self):
        # Cluster 0: 3 pts, Cluster 1: 2 pts, Cluster 2: 5 pts
        return pd.DataFrame({"label": [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]})

    def test_large_clusters(self, clustering):
        large, _ = SpectralEqualSizeClustering._get_clusters_outside_range(
            clustering, 3, 4
        )
        assert 2 in large
        assert 0 not in large
        assert 1 not in large

    def test_small_clusters(self, clustering):
        _, small = SpectralEqualSizeClustering._get_clusters_outside_range(
            clustering, 3, 4
        )
        assert 1 in small
        assert 0 not in small
        assert 2 not in small

    def test_all_in_range(self):
        clustering = pd.DataFrame({"label": [0, 0, 1, 1, 2, 2]})
        large, small = SpectralEqualSizeClustering._get_clusters_outside_range(
            clustering, 2, 2
        )
        assert large == []
        assert small == []


# ---------------------------------------------------------------------------
# _get_points_to_switch
# ---------------------------------------------------------------------------


class TestGetPointsToSwitch:
    @pytest.fixture
    def setup(self):
        # Points 0 and 1 are to be switched.
        # Point 0 is clearly closer to cluster 1 (points 2,3).
        # Point 1 is clearly closer to cluster 2 (points 4,5).
        dm = np.array(
            [
                [0, 1, 1, 2, 8, 9],
                [1, 0, 8, 9, 1, 2],
                [1, 8, 0, 1, 8, 9],
                [2, 9, 1, 0, 9, 8],
                [8, 1, 8, 9, 0, 1],
                [9, 2, 9, 8, 1, 0],
            ],
            dtype=float,
        )
        cl_elements = [0, 1]
        clusters_to_modify = [1, 2]
        idxc = {1: [2, 3], 2: [4, 5]}
        return dm, cl_elements, clusters_to_modify, idxc

    def test_output_columns(self, setup):
        dm, cl_elements, clusters_to_modify, idxc = setup
        result = SpectralEqualSizeClustering._get_points_to_switch(
            dm, cl_elements, clusters_to_modify, idxc
        )
        assert "neighbor_c" in result.columns
        assert "distance" in result.columns

    def test_correct_neighbor_assignment(self, setup):
        dm, cl_elements, clusters_to_modify, idxc = setup
        result = SpectralEqualSizeClustering._get_points_to_switch(
            dm, cl_elements, clusters_to_modify, idxc
        )
        assert result.loc[0, "neighbor_c"] == 1
        assert result.loc[1, "neighbor_c"] == 2

    def test_sorted_by_distance_ascending(self, setup):
        dm, cl_elements, clusters_to_modify, idxc = setup
        result = SpectralEqualSizeClustering._get_points_to_switch(
            dm, cl_elements, clusters_to_modify, idxc
        )
        distances = list(result["distance"])
        assert distances == sorted(distances)

    def test_output_indexed_by_points(self, setup):
        dm, cl_elements, clusters_to_modify, idxc = setup
        result = SpectralEqualSizeClustering._get_points_to_switch(
            dm, cl_elements, clusters_to_modify, idxc
        )
        assert set(result.index) == set(cl_elements)

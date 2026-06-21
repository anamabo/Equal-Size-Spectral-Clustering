# Tests

Unit tests for the pure functions in `elsara/spectral_equal_size_clustering.py`.

## How to run

```bash
poetry run pytest tests/ -v
```

## Coverage

| Class | Tests | What's covered |
|---|---|---|
| `TestOptimalClusterSizes` | 7 | Uneven division, even division, total = npoints, sizes differ by ≤1 |
| `TestClusterDispersion` | 5 | Output shape, zero dispersion for uniform cluster, computed value against `np.std`, missing column raises `ValueError`, non-negative values |
| `TestGetNNeighboursPerPoint` | 4 | Output shape, self-exclusion invariant, correct ordering, all points have a list |
| `TestGetClustersOutsideRange` | 3 | Large cluster detection, small cluster detection, all-in-range case |
| `TestGetPointsToSwitch` | 4 | Output columns, correct neighbor assignment, ascending sort by distance, index = input points |

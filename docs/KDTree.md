# KD-Tree

The KD-Tree module is a custom implementation used for nearest-neighbor search in ICP.

## What It Is

A KD-Tree is a binary spatial partitioning tree. In 3D, each node splits points along one coordinate axis: `x`, `y`, or `z`.

## Node Structure

Each node stores:

```python
point
index
axis
left
right
```

The `index` is the original row index in the input point array, so query results can be mapped back to the source data.

## Build Flow

1. Select the split axis from recursion depth: `axis = depth % 3`.
2. Sort points by that axis.
3. Use the median point as the current node.
4. Recursively build the left subtree from lower-axis points.
5. Recursively build the right subtree from higher-axis points.

Median split keeps the tree reasonably balanced for demo and benchmark data.

## Nearest Neighbor Query

The nearest-neighbor search:

1. Descends into the subtree that contains the query point.
2. Updates the best current candidate.
3. Backtracks when the hypersphere around the query intersects the splitting plane.

Backtracking is required because the nearest point may lie on the other side of the split plane.

## KNN Query

KNN search keeps a bounded max-heap of the best `k` candidates. The heap stores the current worst accepted neighbor at the root, so a better point can replace it quickly.

The implementation returns sorted `(index, distance)` pairs after traversal.

## Radius Search

Radius search visits a node when it may contain points within the query radius. The opposite subtree is pruned when:

```text
(query_axis - split_axis_value)^2 > radius^2
```

This avoids searching regions that cannot contain valid neighbors.

## Complexity

Average balanced-tree behavior:

```text
Build: O(N log N)
Nearest query: O(log N) average
Worst-case query: O(N)
```

Brute-force nearest-neighbor search is:

```text
O(N)
```

per query. KD-Tree search is therefore useful when many queries are performed against the same target cloud, as in ICP.

## Role in ICP

ICP repeatedly searches target nearest neighbors for every source point. Building a KD-Tree once per target cloud reduces the correspondence-search cost across all iterations.

## Queries

- `nearest_neighbor(query_point)` returns `(index, distance)`.
- `knn_search(query_point, k)` returns sorted `(index, distance)` pairs.
- `radius_search(query_point, radius)` returns all neighbors inside the radius.

The tests compare all query types against brute-force NumPy search.


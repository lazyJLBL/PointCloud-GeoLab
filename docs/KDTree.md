# KD-Tree

The KD-Tree module is a custom implementation used for nearest-neighbor search in ICP.

## Node Structure

Each node stores:

```python
point
index
axis
left
right
```

The tree is built with a balanced median split. The split axis alternates across `x`, `y`, and `z`.

## Queries

- `nearest_neighbor(query_point)` returns `(index, distance)`.
- `knn_search(query_point, k)` returns sorted `(index, distance)` pairs.
- `radius_search(query_point, radius)` returns all neighbors inside the radius.

The tests compare all query types against brute-force NumPy search.


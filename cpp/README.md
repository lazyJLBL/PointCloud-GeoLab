# C++17 Optional Demo

This folder is intentionally independent from the Python package. It gives a
small C++17 KDTree nearest-neighbor example for interviews where systems
implementation matters, without adding complexity to the Python test suite.

Build:

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build --config Release
```

Run:

```bash
cpp/build/kdtree_demo
```

On Windows with a multi-config generator, the executable may be under
`cpp/build/Release/kdtree_demo.exe`.

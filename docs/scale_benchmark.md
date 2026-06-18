# Scale Benchmark

The scale benchmark records local timing and memory reference points for
synthetic point counts. It is useful for reviewer sanity checks, not a portable
performance claim.

## Quick Gate

```bash
python scripts/run_scale_benchmark.py \
  --quick \
  --repeat 2 \
  --output-dir outputs/scale_benchmark
python scripts/verify_benchmarks.py \
  --output-dir outputs/scale_benchmark \
  --suite scale
```

Quick mode uses 1k and 10k synthetic points. The manual full mode adds 100k and
1M points:

```bash
python scripts/run_scale_benchmark.py \
  --no-quick \
  --repeat 3 \
  --output-dir outputs/scale_benchmark_full
```

Full mode belongs in a manual gate because runtime and memory depend strongly
on the local machine.

## Outputs

- `scale_benchmark.json`: rows, repeat stats, machine info, and memory metadata.
- `scale_benchmark.csv`: tabular rows for quick inspection.
- `scale_benchmark.md`: reviewer-readable summary.
- `scale_benchmark.png`: simple timing trend chart.

The JSON uses `tracemalloc` peak-memory metadata as a local reference. It should
not be described as a hardware-independent benchmark.

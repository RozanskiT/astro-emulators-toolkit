"""Benchmark Cannon design-matrix implementations and record timings to JSON.

This maintainer-focused script compares the current library implementation
against a vectorized candidate without changing package behavior.

Typical usage:

    uv run python examples/development/17_benchmark_cannon_design_matrix.py --label baseline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from astro_emulators_toolkit.models.cannon import (
    cannon_design_matrix,
    cannon_feature_dim,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "examples" / "runs" / "development_benchmarks"

DEFAULT_DIMS = (3, 8, 16, 32, 64, 128)
DEFAULT_BATCH_SIZE = 512
DEFAULT_WARMUP = 3
DEFAULT_REPEAT = 10
DEFAULT_COMPILE_REPEAT = 2
INCLUDE_BIAS = True


@dataclass(frozen=True)
class BenchmarkStats:
    mean_ms: float
    median_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    num_samples: int


def _block_tree(tree: Any) -> Any:
    def _block_leaf(x: Any) -> Any:
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        return x

    return jax.tree_util.tree_map(_block_leaf, tree)


def _stats(samples_ms: list[float]) -> BenchmarkStats:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return BenchmarkStats(
        mean_ms=float(arr.mean()),
        median_ms=float(np.median(arr)),
        p90_ms=float(np.percentile(arr, 90.0)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        std_ms=float(arr.std()),
        num_samples=int(arr.size),
    )


def _time_fn(fn: Callable[[], Any], *, warmup: int, repeat: int) -> BenchmarkStats:
    for _ in range(int(warmup)):
        fn()
    samples_ms: list[float] = []
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return _stats(samples_ms)


def _time_first_jit_call(
    build_fn: Callable[[], Callable[[jnp.ndarray], jnp.ndarray]],
    x: jnp.ndarray,
    *,
    repeat: int,
) -> BenchmarkStats:
    samples_ms: list[float] = []
    for _ in range(int(repeat)):
        fn = jax.jit(build_fn())
        t0 = time.perf_counter()
        y = fn(x)
        _block_tree(y)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return _stats(samples_ms)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _build_current_fn(*, include_bias: bool) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def current(x: jnp.ndarray) -> jnp.ndarray:
        return cannon_design_matrix(x, include_bias=include_bias)

    return current


def _build_vectorized_fn(
    *, in_dim: int, include_bias: bool
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    tri_i, tri_j = np.triu_indices(int(in_dim))

    def vectorized(x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 2:
            raise ValueError(f"Cannon expects x.ndim == 2, got shape={x.shape}.")
        if int(x.shape[1]) != int(in_dim):
            raise ValueError(f"Expected x.shape[1] == {in_dim}, got {x.shape[1]}.")

        quad = jnp.einsum("bi,bj->bij", x, x)[:, tri_i, tri_j]
        columns = [x]
        if include_bias:
            columns = [jnp.ones((x.shape[0], 1), dtype=x.dtype), *columns]
        columns.append(quad)
        return jnp.concatenate(columns, axis=1)

    return vectorized


def _flatten_metrics(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_metrics(next_prefix, child, out)
        return
    if prefix.endswith("median_ms") and isinstance(value, (int, float)):
        out[prefix] = float(value)


def _compare(current: dict[str, Any], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text())
    flat_current: dict[str, float] = {}
    flat_baseline: dict[str, float] = {}
    _flatten_metrics("", current.get("benchmarks", {}), flat_current)
    _flatten_metrics("", baseline.get("benchmarks", {}), flat_baseline)

    print(f"Comparison vs {baseline_path}:")
    for key in sorted(set(flat_current) & set(flat_baseline)):
        cur = flat_current[key]
        base = flat_baseline[key]
        if base == 0.0:
            continue
        delta_pct = 100.0 * (cur - base) / base
        print(f"  {key}: {cur:.3f} ms vs {base:.3f} ms ({delta_pct:+.1f}%)")


def _benchmark_impl(
    *,
    build_fn: Callable[[], Callable[[jnp.ndarray], jnp.ndarray]],
    x: jnp.ndarray,
    warmup: int,
    repeat: int,
    compile_repeat: int,
) -> dict[str, Any]:
    eager_fn = build_fn()
    eager = _time_fn(lambda: _block_tree(eager_fn(x)), warmup=warmup, repeat=repeat)

    steady_jit_fn = jax.jit(build_fn())
    _block_tree(steady_jit_fn(x))
    jit_steady = _time_fn(
        lambda: _block_tree(steady_jit_fn(x)), warmup=warmup, repeat=repeat
    )
    jit_first_call = _time_first_jit_call(build_fn, x, repeat=compile_repeat)

    return {
        "eager": asdict(eager),
        "jit_first_call": asdict(jit_first_call),
        "jit_steady": asdict(jit_steady),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", required=True, help="Output label used for the JSON filename."
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="Optional earlier JSON file to compare against.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--compile-repeat", type=int, default=DEFAULT_COMPILE_REPEAT)
    parser.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    dims = tuple(int(dim) for dim in args.dims)

    payload: dict[str, Any] = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "device": [str(device) for device in jax.devices()],
        "jax_version": getattr(jax, "__version__", "unknown"),
        "numpy_version": getattr(np, "__version__", "unknown"),
        "settings": {
            "batch_size": int(args.batch_size),
            "warmup": int(args.warmup),
            "repeat": int(args.repeat),
            "compile_repeat": int(args.compile_repeat),
            "dims": list(dims),
        },
        "benchmarks": {},
    }

    print("Cannon design-matrix benchmark:")
    for in_dim in dims:
        x = jax.device_put(
            rng.normal(size=(int(args.batch_size), int(in_dim))).astype(np.float32)
        )

        def current_fn_builder() -> Callable[[jax.Array], jax.Array]:
            return _build_current_fn(include_bias=INCLUDE_BIAS)

        def candidate_fn_builder(
            in_dim: int = in_dim,
        ) -> Callable[[jax.Array], jax.Array]:
            return _build_vectorized_fn(in_dim=in_dim, include_bias=INCLUDE_BIAS)

        current = _benchmark_impl(
            build_fn=current_fn_builder,
            x=x,
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            compile_repeat=int(args.compile_repeat),
        )
        candidate = _benchmark_impl(
            build_fn=candidate_fn_builder,
            x=x,
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            compile_repeat=int(args.compile_repeat),
        )

        y_current = _build_current_fn(include_bias=INCLUDE_BIAS)(x)
        y_candidate = _build_vectorized_fn(in_dim=in_dim, include_bias=INCLUDE_BIAS)(x)
        max_abs_diff = float(jnp.max(jnp.abs(y_current - y_candidate)))
        feature_dim = int(cannon_feature_dim(in_dim, include_bias=INCLUDE_BIAS))

        eager_speedup = current["eager"]["median_ms"] / max(
            candidate["eager"]["median_ms"], 1e-12
        )
        first_call_speedup = current["jit_first_call"]["median_ms"] / max(
            candidate["jit_first_call"]["median_ms"], 1e-12
        )
        steady_speedup = current["jit_steady"]["median_ms"] / max(
            candidate["jit_steady"]["median_ms"], 1e-12
        )

        payload["benchmarks"][f"dim_{in_dim}"] = {
            "feature_dim": feature_dim,
            "max_abs_diff": max_abs_diff,
            "current": current,
            "candidate": candidate,
            "speedup_ratio": {
                "eager_median": eager_speedup,
                "jit_first_call_median": first_call_speedup,
                "jit_steady_median": steady_speedup,
            },
        }

        print(
            f"  dim={in_dim:>3} feat={feature_dim:>5} "
            f"eager {current['eager']['median_ms']:.3f}->{candidate['eager']['median_ms']:.3f} ms "
            f"({eager_speedup:.2f}x), "
            f"jit first {current['jit_first_call']['median_ms']:.3f}->{candidate['jit_first_call']['median_ms']:.3f} ms "
            f"({first_call_speedup:.2f}x), "
            f"jit steady {current['jit_steady']['median_ms']:.3f}->{candidate['jit_steady']['median_ms']:.3f} ms "
            f"({steady_speedup:.2f}x), "
            f"diff={max_abs_diff:.3e}"
        )

    out_path = RUN_DIR / f"{args.label}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote benchmark results to {out_path}")

    if args.compare_to is not None:
        _compare(payload, args.compare_to)


if __name__ == "__main__":
    main()

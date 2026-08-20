import numpy as np
import pandas as pd

_CHUNK = 256


def _as_matrix(differences) -> np.ndarray:
    if isinstance(differences, pd.DataFrame):
        matrix = differences.to_numpy(dtype=float)
    else:
        matrix = np.asarray(differences, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"differences must be 1- or 2-dimensional, got {matrix.ndim}")
    if matrix.shape[0] < 2:
        raise ValueError(f"need at least 2 observations, got {matrix.shape[0]}")
    if matrix.shape[1] < 1:
        raise ValueError("need at least 1 strategy column")
    if not np.isfinite(matrix).all():
        raise ValueError("differences contain NaN or infinite values")
    return matrix


def default_mean_block(n: int) -> float:
    return max(1.0, float(n) ** (1.0 / 3.0))


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if mean_block <= 0:
        raise ValueError(f"mean_block must be positive, got {mean_block}")

    restart_prob = min(1.0, 1.0 / mean_block)
    idx = np.empty((n_boot, n), dtype=np.int32)
    idx[:, 0] = rng.integers(0, n, size=n_boot)

    if n > 1:
        restart = rng.random((n_boot, n - 1)) < restart_prob
        fresh = rng.integers(0, n, size=(n_boot, n - 1), dtype=np.int32)
        for t in range(1, n):
            carry_on = idx[:, t - 1] + 1
            carry_on[carry_on == n] = 0
            idx[:, t] = np.where(restart[:, t - 1], fresh[:, t - 1], carry_on)
    return idx


def _bootstrap_means(
    matrix: np.ndarray, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    n = matrix.shape[0]
    out = np.empty((n_boot, matrix.shape[1]))
    for start in range(0, n_boot, _CHUNK):
        size = min(_CHUNK, n_boot - start)
        idx = stationary_bootstrap_indices(n, size, mean_block, rng)
        counts = np.zeros((size, n))
        for row in range(size):
            counts[row] = np.bincount(idx[row], minlength=n)
        out[start : start + size] = (counts @ matrix) / n
    return out


def reality_check(
    differences,
    n_boot: int = 1000,
    mean_block: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> dict:
    matrix = _as_matrix(differences)
    n = matrix.shape[0]
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if mean_block is None:
        mean_block = default_mean_block(n)
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    scale = n**0.5
    observed_means = matrix.mean(axis=0)
    boot_means = _bootstrap_means(matrix, n_boot, mean_block, rng)

    centred = boot_means - observed_means

    statistic = float((scale * observed_means).max())
    null = (scale * centred).max(axis=1)

    exceedances = int((null >= statistic).sum())
    return {
        "p_value": (1 + exceedances) / (n_boot + 1),
        "statistic": statistic,
        "null": null,
        "n_boot": n_boot,
        "mean_block": float(mean_block),
    }

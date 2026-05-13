from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .toy_utils import (
    build_line_model_state,
    evaluate_line_model_batch,
    generate_linelist,
    generate_rff_dataset,
)


@dataclass
class ToyRFFDataset:
    """Tiny Random Fourier Features toy dataset for quick tests."""

    n_samples: int = 256
    x_dim: int = 4
    y_dim: int = 2
    n_features: int = 16
    freq_scale: float = 2.0
    noise_std: float = 0.0
    x_dist: Literal["uniform", "normal"] = "uniform"
    seed: int = 0

    def __post_init__(self) -> None:
        self.x, self.y = generate_rff_dataset(
            n_samples=self.n_samples,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            n_features=self.n_features,
            freq_scale=self.freq_scale,
            noise_std=self.noise_std,
            x_dist=self.x_dist,
            seed=self.seed,
        )

    def __len__(self) -> int:
        return self.n_samples

    def get_batch(self, idx: np.ndarray) -> dict[str, Any]:
        idx = np.asarray(idx)
        return {"x": self.x[idx], "y": self.y[idx]}


@dataclass
class ToyNormalizedFluxDataset:
    """Toy normalized-flux dataset on a fixed wavelength grid."""

    n_samples: int = 256
    x_dim: int = 5
    y_dim: int = 128
    lines_per_atom: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.x_dim < 2:
            raise ValueError(
                "x_dim must be >= 2 (temperature + at least one abundance control)."
            )

        rng = np.random.default_rng(self.seed)
        self.wavelength_grid = np.linspace(0.0, 1.0, self.y_dim, dtype=np.float32)
        n_atoms = self.x_dim - 1
        linelist = generate_linelist(
            rng, n_atoms=n_atoms, lines_per_atom=self.lines_per_atom
        )
        self.line_state = build_line_model_state(self.wavelength_grid, linelist)

        self.x = rng.uniform(0.0, 1.0, size=(self.n_samples, self.x_dim)).astype(
            np.float32
        )
        i_bck, i_out, _ = evaluate_line_model_batch(
            self.x, mu=1.0, state=self.line_state
        )
        self.y = (i_out / (i_bck + 1e-12)).astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def get_batch(self, idx: np.ndarray) -> dict[str, Any]:
        idx = np.asarray(idx)
        return {"x": self.x[idx], "y": self.y[idx]}


@dataclass
class ToyIntensityDataset:
    """Toy intensity dataset with two channels on a fixed wavelength grid."""

    n_samples: int = 256
    x_dim: int = 6
    y_dim: int = 128
    lines_per_atom: int = 10
    mu_min: float = 0.1
    mu_max: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.x_dim < 3:
            raise ValueError(
                "x_dim must be >= 3 (mu + temperature + abundance controls)."
            )

        rng = np.random.default_rng(self.seed)
        self.wavelength_grid = np.linspace(0.0, 1.0, self.y_dim, dtype=np.float32)

        n_atoms = self.x_dim - 2
        linelist = generate_linelist(
            rng, n_atoms=n_atoms, lines_per_atom=self.lines_per_atom
        )
        self.line_state = build_line_model_state(self.wavelength_grid, linelist)

        self.x = rng.uniform(0.0, 1.0, size=(self.n_samples, self.x_dim)).astype(
            np.float32
        )
        mu = self.mu_min + (self.mu_max - self.mu_min) * np.clip(self.x[:, 0], 0.0, 1.0)
        p_batch = self.x[:, 1:]

        i_bck, i_out, _ = evaluate_line_model_batch(
            p_batch, mu=mu, state=self.line_state
        )
        i_bck_batch = np.repeat(i_bck, repeats=self.n_samples, axis=0)
        self.y = np.stack([i_bck_batch, i_out], axis=-1).astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def get_batch(self, idx: np.ndarray) -> dict[str, Any]:
        idx = np.asarray(idx)
        return {"x": self.x[idx], "y": self.y[idx]}


@dataclass
class ToyBinaryClassificationDataset:
    """Toy binary dataset from RFF logits -> sigmoid -> Bernoulli sample."""

    n_samples: int = 256
    x_dim: int = 4
    n_features: int = 16
    freq_scale: float = 2.0
    amplitude: float = 1.0
    x_dist: Literal["uniform", "normal"] = "uniform"
    seed: int = 0

    def __post_init__(self) -> None:
        base = ToyRFFDataset(
            n_samples=self.n_samples,
            x_dim=self.x_dim,
            y_dim=1,
            n_features=self.n_features,
            freq_scale=self.freq_scale,
            x_dist=self.x_dist,
            seed=self.seed,
        )
        base.y *= self.amplitude
        rng = np.random.default_rng(self.seed)
        probs = 1.0 / (1.0 + np.exp(-base.y[:, 0]))
        self.x = base.x
        self.y_prob = probs.astype(np.float32)
        self.y = rng.binomial(1, self.y_prob).astype(np.float32)[:, None]

    def __len__(self) -> int:
        return self.n_samples

    def get_batch(self, idx: np.ndarray) -> dict[str, Any]:
        idx = np.asarray(idx)
        return {"x": self.x[idx], "y": self.y[idx]}

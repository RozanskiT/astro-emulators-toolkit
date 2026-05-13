from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToyLineList:
    center: np.ndarray
    atom: np.ndarray
    strength: np.ndarray
    width: np.ndarray
    ion: np.ndarray
    t_sens: np.ndarray


@dataclass
class ToyLineModelState:
    wavelength_grid: np.ndarray
    linelist: ToyLineList
    profile: np.ndarray
    flip: np.ndarray
    i_background: np.ndarray


def expit(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(1.0 / (1.0 + np.exp(-x)))


def generate_rff_dataset(
    *,
    n_samples: int,
    x_dim: int,
    y_dim: int,
    n_features: int,
    freq_scale: float,
    noise_std: float,
    x_dist: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if x_dist == "uniform":
        x = rng.uniform(-1.0, 1.0, size=(n_samples, x_dim))
    elif x_dist == "normal":
        x = rng.normal(size=(n_samples, x_dim))
    else:
        raise ValueError(f"Unsupported x_dist='{x_dist}'.")

    w = rng.normal(scale=freq_scale, size=(n_features, x_dim))
    b = rng.uniform(0.0, 2 * np.pi, size=(n_features,))
    a = rng.normal(scale=1.0 / np.sqrt(2 * n_features), size=(y_dim, 2 * n_features))

    z = x @ w.T + b
    phi = np.concatenate([np.sin(z), np.cos(z)], axis=1)
    y = phi @ a.T
    if noise_std > 0:
        y = y + rng.normal(scale=noise_std, size=y.shape)

    return x.astype(np.float32), y.astype(np.float32)


def planck_proxy_lambda01(
    lam: np.ndarray,
    temperature: np.ndarray | float,
    *,
    shift: float = 0.1,
) -> np.ndarray:
    lam_eff = np.asarray(lam) + shift
    temperature = np.clip(np.asarray(temperature), 1e-6, None)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x = 1.0 / (lam_eff * temperature)
        return (1.0 / lam_eff**5) / (np.expm1(x) + 1e-12)


def generate_linelist(
    rng: np.random.Generator,
    n_atoms: int,
    lines_per_atom: int,
    *,
    min_strength: float = 0.5,
    max_strength: float = 2.0,
    min_width: float = 0.001,
    max_width: float = 0.005,
    sens_t_min: float = 0.0,
    sens_t_max: float = 1.0,
) -> ToyLineList:
    n_lines = n_atoms * lines_per_atom
    atom = np.repeat(np.arange(n_atoms, dtype=np.int32), lines_per_atom)
    rng.shuffle(atom)
    return ToyLineList(
        center=rng.uniform(0.0, 1.0, size=n_lines),
        atom=atom,
        strength=rng.uniform(min_strength, max_strength, size=n_lines),
        width=rng.uniform(min_width, max_width, size=n_lines),
        ion=rng.integers(0, 2, size=n_lines, dtype=np.int32),
        t_sens=rng.uniform(sens_t_min, sens_t_max, size=n_lines),
    )


def build_line_model_state(
    wavelength_grid: np.ndarray,
    linelist: ToyLineList,
    *,
    background_temperature: float = 1.0,
) -> ToyLineModelState:
    profile = np.exp(
        -((linelist.center[:, None] - wavelength_grid[None, :]) ** 2)
        / (linelist.width[:, None] ** 2)
    )
    flip = (2.0 * linelist.ion - 1.0).astype(np.float32)
    i_background = planck_proxy_lambda01(wavelength_grid, background_temperature)
    return ToyLineModelState(
        wavelength_grid=wavelength_grid,
        linelist=linelist,
        profile=profile,
        flip=flip,
        i_background=i_background,
    )


def evaluate_line_model_batch(
    p_batch: np.ndarray,
    mu: np.ndarray | float,
    state: ToyLineModelState,
    *,
    temp_fg_min: float = 0.1,
    temp_fg_max: float = 0.9,
    abundance_min: float = 0.0,
    abundance_max: float = 2.0,
    sigmoid_k: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(p_batch, dtype=np.float32)
    if p.ndim == 1:
        p = p[None, :]

    n_atoms = int(np.max(state.linelist.atom)) + 1
    if p.shape[1] < 1 + n_atoms:
        raise ValueError(
            f"Expected at least {1 + n_atoms} parameters, got {p.shape[1]}."
        )

    t_fg = temp_fg_min + (temp_fg_max - temp_fg_min) * np.clip(p[:, 0], 0.0, 1.0)
    abund = abundance_min + (abundance_max - abundance_min) * np.clip(
        p[:, 1 : 1 + n_atoms], 0.0, 1.0
    )

    b_fg = planck_proxy_lambda01(state.wavelength_grid[None, :], t_fg[:, None])
    abund_line = abund[:, state.linelist.atom]
    sens = expit(
        (state.flip[None, :] * (t_fg[:, None] - state.linelist.t_sens[None, :]))
        / sigmoid_k
    )
    amp = (state.linelist.strength[None, :] * abund_line) * sens
    tau = amp @ state.profile

    mu = np.asarray(mu, dtype=np.float32)
    if mu.ndim == 0:
        mu = np.full((p.shape[0],), float(mu), dtype=np.float32)
    trans = np.exp(-tau / np.clip(mu[:, None], 1e-3, None))
    i_out = state.i_background[None, :] * trans + (1.0 - trans) * b_fg
    return state.i_background[None, :], i_out, tau

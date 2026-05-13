from __future__ import annotations

from ..config.schema import (
    IOTreeSpec,
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)


def _validate_profile(profile: str) -> str:
    normalized = str(profile)
    if normalized not in {"smoke", "cpu_recommended"}:
        raise ValueError(
            f"Unsupported preset profile '{profile}'. Expected 'smoke' or 'cpu_recommended'."
        )
    return normalized


def payne_flux_mlp(
    *,
    workdir: str = "./runs/preset_payne_flux_mlp",
    profile: str = "cpu_recommended",
    init_hints: dict[str, int] | None = None,
) -> RootConfig:
    """Return a stable MLP preset for Payne-style flux regression."""
    profile = _validate_profile(profile)
    if profile == "smoke":
        model_params = {
            "hidden_sizes": (64, 64),
            "activation": "gelu",
            "dtype": "float32",
        }
        optim = OptimConfig(
            name="soap", lr=1e-3, schedule="cosine", warmup_steps=1, weight_decay=1e-5
        )
        training = TrainConfig(
            workdir=workdir,
            batch_size=64,
            num_steps=6,
            logging_interval_steps=1,
            evaluation_interval_steps=2,
            checkpoint_interval_steps=2,
            max_saved_checkpoints=2,
        )
    else:
        model_params = {
            "hidden_sizes": (128, 128),
            "activation": "gelu",
            "dtype": "float32",
        }
        optim = OptimConfig(
            name="soap",
            lr=1e-3,
            schedule="cosine",
            warmup_steps=1_000,
            weight_decay=1e-5,
        )
        training = TrainConfig(
            workdir=workdir,
            batch_size=128,
            num_steps=10_000,
            logging_interval_steps=50,
            evaluation_interval_steps=500,
            checkpoint_interval_steps=500,
            max_saved_checkpoints=5,
        )
    return RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None}),
            outputs=IOTreeSpec(structure_tree={"flux": None}),
        ),
        model=ModelSpec(
            name="mlp",
            params=model_params,
            init_hints={} if init_hints is None else dict(init_hints),
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=optim,
        training=training,
    )


def isochrone_mlp(
    *,
    workdir: str = "./runs/preset_isochrone_mlp",
    profile: str = "cpu_recommended",
    init_hints: dict[str, int] | None = None,
) -> RootConfig:
    """Return a stable MLP preset for tabular isochrone regression."""
    profile = _validate_profile(profile)
    if profile == "smoke":
        model_params = {
            "hidden_sizes": (64, 64),
            "activation": "gelu",
            "dtype": "float32",
        }
        optim = OptimConfig(
            name="soap", lr=3e-3, schedule="cosine", warmup_steps=1, weight_decay=1e-5
        )
        training = TrainConfig(
            workdir=workdir,
            batch_size=64,
            num_steps=6,
            logging_interval_steps=1,
            evaluation_interval_steps=2,
            checkpoint_interval_steps=0,
            max_saved_checkpoints=0,
        )
    else:
        model_params = {
            "hidden_sizes": (128, 128),
            "activation": "gelu",
            "dtype": "float32",
        }
        optim = OptimConfig(
            name="soap",
            lr=3e-3,
            schedule="cosine",
            warmup_steps=1_000,
            weight_decay=1e-5,
        )
        training = TrainConfig(
            workdir=workdir,
            batch_size=128,
            num_steps=10_000,
            logging_interval_steps=50,
            evaluation_interval_steps=500,
            checkpoint_interval_steps=500,
            max_saved_checkpoints=5,
        )
    return RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None}),
            outputs=IOTreeSpec(structure_tree={"targets": None}),
        ),
        model=ModelSpec(
            name="mlp",
            params=model_params,
            init_hints={} if init_hints is None else dict(init_hints),
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=optim,
        training=training,
    )

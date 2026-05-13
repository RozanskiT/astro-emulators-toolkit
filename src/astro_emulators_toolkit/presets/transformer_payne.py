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


def _intensity_channel_names(channels: int) -> tuple[str, ...]:
    if int(channels) == 2:
        return ("normalized_intensity", "log10_continuum_minmax")
    return tuple(f"channel_{i}" for i in range(int(channels)))


def transformer_payne_flux(
    *,
    channels: int = 1,
    workdir: str = "./runs/preset_transformer_payne_flux",
    profile: str = "cpu_recommended",
    init_hints: dict[str, int] | None = None,
) -> RootConfig:
    """Return a stable transformer-payne preset for flux prediction."""
    profile = _validate_profile(profile)
    if profile == "smoke":
        model_params = {
            "channels": channels,
            "dim": 16,
            "dim_head": 8,
            "no_layers": 1,
            "no_tokens": 2,
            "dim_ff_multiplier": 2,
            "min_period": 3e-2,
            "max_period": 30.0,
            "dtype": "float32",
        }
        optim = OptimConfig(
            name="soap", lr=1e-3, schedule="cosine", warmup_steps=1, weight_decay=1e-5
        )
        training = TrainConfig(
            workdir=workdir,
            batch_size=32,
            num_steps=4,
            logging_interval_steps=1,
            evaluation_interval_steps=1,
            checkpoint_interval_steps=0,
            max_saved_checkpoints=0,
        )
    else:
        model_params = {
            "channels": channels,
            "dim": 32,
            "dim_head": 32,
            "no_layers": 2,
            "no_tokens": 4,
            "dim_ff_multiplier": 2,
            "min_period": 3e-2,
            "max_period": 30.0,
            "dtype": "float32",
        }
        optim = OptimConfig(
            name="soap", lr=3e-3, schedule="cosine", warmup_steps=150, weight_decay=1e-5
        )
        training = TrainConfig(
            workdir=workdir,
            batch_size=64,
            num_steps=1_500,
            logging_interval_steps=50,
            evaluation_interval_steps=200,
            checkpoint_interval_steps=500,
            max_saved_checkpoints=5,
        )
    return RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None, "wavelengths": None}),
            outputs=IOTreeSpec(structure_tree={"flux": None}),
        ),
        model=ModelSpec(
            name="transformer_payne",
            params=model_params,
            init_hints={} if init_hints is None else dict(init_hints),
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=optim,
        training=training,
    )


def transformer_payne_intensity(
    *,
    channels: int = 2,
    workdir: str = "./runs/preset_transformer_payne_intensity",
    profile: str = "cpu_recommended",
    init_hints: dict[str, int] | None = None,
) -> RootConfig:
    """Return a transformer-payne preset with intensity-style output channels."""
    cfg = transformer_payne_flux(
        channels=channels, workdir=workdir, profile=profile, init_hints=init_hints
    )
    return RootConfig(
        schema_version=cfg.schema_version,
        seed=cfg.seed,
        model=cfg.model,
        task=cfg.task,
        solver=cfg.solver,
        optim=cfg.optim,
        training=cfg.training,
        bundle=cfg.bundle,
        hub=cfg.hub,
        io=IOSpec(
            inputs=cfg.io.inputs,
            outputs=IOTreeSpec(
                structure_tree={"flux": None},
                channel_names_tree={"flux": _intensity_channel_names(channels)},
            ),
            reference_scaling_inputs=cfg.io.reference_scaling_inputs,
            reference_scaling_outputs=cfg.io.reference_scaling_outputs,
            input_domain=cfg.io.input_domain,
        ),
    )

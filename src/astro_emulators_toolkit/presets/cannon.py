from __future__ import annotations

from ..config.schema import (
    IOTreeSpec,
    IOSpec,
    ModelSpec,
    RootConfig,
    SolverConfig,
    TaskSpec,
    TrainConfig,
)


def cannon_flux(
    *,
    workdir: str = "./runs/preset_cannon_flux",
    profile: str = "cpu_recommended",
    init_hints: dict[str, int] | None = None,
) -> RootConfig:
    """Return a stable closed-form Cannon preset for flux regression."""
    if profile not in {"smoke", "cpu_recommended"}:
        raise ValueError(
            f"Unsupported preset profile '{profile}'. Expected 'smoke' or 'cpu_recommended'."
        )
    training = TrainConfig(
        workdir=workdir,
        batch_size=256,
        num_steps=1,
        logging_interval_steps=1,
        evaluation_interval_steps=1,
        checkpoint_interval_steps=0,
        max_saved_checkpoints=0,
    )
    return RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None}),
            outputs=IOTreeSpec(structure_tree={"flux": None}),
        ),
        model=ModelSpec(
            name="cannon",
            params={"include_bias": True},
            init_hints={} if init_hints is None else dict(init_hints),
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        solver=SolverConfig(
            name="closed_form_linear",
            params={"ridge": 1e-4, "regularize_intercept": False},
        ),
        training=training,
    )

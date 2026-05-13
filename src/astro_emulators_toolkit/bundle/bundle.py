from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config.io import load_config, save_config
from ..config.schema import (
    CONFIG_SCHEMA_VERSION,
    RootConfig,
    canonicalize_config_names,
)
from .safetensors_io import load_weights, save_weights


CONFIG_FILENAME = "config.json"
METADATA_FILENAME = "metadata.json"
BUNDLE_INTEGRITY_FILENAME = "bundle_integrity.json"
README_FILENAME = "README.txt"
WEIGHTS_SUBDIR = "weights"
WEIGHTS_FILENAME = "weights.safetensors"
PORTABLE_BUNDLE_WORKDIR = "./runs/from_bundle"


def make_portable_bundle_config(cfg: RootConfig) -> RootConfig:
    return canonicalize_config_names(
        cfg.with_updates(
            schema_version=CONFIG_SCHEMA_VERSION,
            training=replace(cfg.training, workdir=PORTABLE_BUNDLE_WORKDIR),
        )
    )


@dataclass
class Bundle:
    cfg: RootConfig
    params_pure: dict[str, Any]
    metadata: dict[str, Any]
    readme_text: str | None = None

    def save(self, dirpath: str | Path) -> None:
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        save_config(make_portable_bundle_config(self.cfg), d / CONFIG_FILENAME)
        weights_dir = d / WEIGHTS_SUBDIR
        weights_dir.mkdir(parents=True, exist_ok=True)
        save_weights(weights_dir / WEIGHTS_FILENAME, params=self.params_pure)
        (d / METADATA_FILENAME).write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True)
        )
        (d / README_FILENAME).write_text(
            self.readme_text or "Astro Emulators Toolkit Bundle\n"
        )

    @classmethod
    def load(cls, dirpath: str | Path) -> "Bundle":
        d = Path(dirpath)
        for filename in (CONFIG_FILENAME, METADATA_FILENAME):
            if not (d / filename).exists():
                raise FileNotFoundError(
                    f"Bundle is missing required file: {d / filename}"
                )

        weights_path = d / WEIGHTS_SUBDIR / WEIGHTS_FILENAME
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Bundle is missing required weights file: {weights_path}"
            )

        readme_path = d / README_FILENAME
        if not readme_path.exists():
            raise FileNotFoundError(f"Bundle is missing required file: {readme_path}")

        cfg = load_config(d / CONFIG_FILENAME)
        params_pure = load_weights(weights_path)
        metadata = json.loads((d / METADATA_FILENAME).read_text())
        return cls(
            cfg=cfg,
            params_pure=params_pure,
            metadata=metadata,
            readme_text=readme_path.read_text(),
        )

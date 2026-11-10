"""Model cards (``README.md``) for bundles published on the Hugging Face Hub.

A card has three parts:

1. YAML front matter; ``library_name: astro-emulators-toolkit`` makes the
   bundle findable as an astro-emulators-toolkit model on the Hub.
2. The author's text: introduction, intended use, training data, validation,
   known limitations, citation. The default card starts each section with an
   ``<!-- aemu:todo ... -->`` prompt, which the author replaces with real text;
   ``aemu check`` refuses to publish while any prompt is left.
3. A generated block between ``<!-- aemu:generated:begin -->`` and
   ``<!-- aemu:generated:end -->``: interface table, usage snippet,
   requirements, release and provenance, contract files. It is rebuilt from
   the bundle metadata by ``write_model_card`` (``aemu card``) and before
   every ``aemu publish``; author text outside the block is never touched.

The card is a documentation file: it is not part of the bundle contract and
is not covered by the integrity manifest.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

CARD_FILENAME = "README.md"
LIBRARY_NAME = "astro-emulators-toolkit"
GENERATED_BEGIN = "<!-- aemu:generated:begin -->"
GENERATED_END = "<!-- aemu:generated:end -->"
TODO_MARKER = "<!-- aemu:todo"

# Families whose wavelength handling needs JAX float64.
_X64_FAMILY_PREFIXES = ("transformer_payne",)

_AUTHOR_SECTIONS = (
    (
        None,
        "A short introduction: what the emulator predicts, for which objects, "
        "and which expensive model or grid it emulates.",
    ),
    (
        "Intended use and domain",
        "Where predictions can be trusted and where not (gaps in the training "
        "grid, extrapolation regions).",
    ),
    (
        "Training data",
        "Source of the training set (link it in the front matter `datasets:`), "
        "its size, the train/validation split, and any preprocessing beyond "
        "the recorded scaling.",
    ),
    (
        "Model and training",
        "Architecture and training choices worth knowing, including any "
        "fine-tuning history.",
    ),
    (
        "Validation",
        "Held-out errors, residual structure against physical parameters, "
        "behaviour near the domain edges, comparison with the expensive "
        "model. The fingerprint listed below is a reproducibility check, not "
        "scientific validation.",
    ),
    (
        "Known limitations",
        "Physical or numerical limitations users must know about.",
    ),
    (
        "Citation and contact",
        "How to cite this emulator, and whom to contact.",
    ),
)


def _leaf_paths(tree: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            yield from _leaf_paths(value, prefix + (str(key),))
    else:
        yield prefix


def _get(tree: Any, path: tuple[str, ...]) -> Any:
    node = tree
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _item(values: Any, index: int) -> Any:
    if isinstance(values, (list, tuple)) and index < len(values):
        return values[index]
    return None


def _fmt(value: Any) -> str:
    number = float(value)
    return f"{number:.6g}" if math.isfinite(number) else str(number)


def _fmt_float(value: Any) -> str:
    """Like _fmt but always a float literal, so generated arrays are float."""
    text = _fmt(value)
    return text if any(c in text for c in ".eEn") else text + ".0"


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _range(lo: Any, hi: Any) -> str:
    if lo is None or hi is None:
        return "—"
    return f"{_fmt(lo)} – {_fmt(hi)}"


def _interface_rows(
    spec_side: Mapping[str, Any],
    domain: Mapping[str, Any] | None,
    *,
    with_domain: bool,
) -> tuple[list[list[str]], list[str]]:
    rows: list[list[str]] = []
    notes: list[str] = []
    min_tree = (domain or {}).get("min_tree")
    max_tree = (domain or {}).get("max_tree")
    for path in _leaf_paths(spec_side.get("structure_tree") or {}):
        leaf = "/".join(path)
        names = _get(spec_side.get("channel_names_tree"), path)
        meanings = _get(spec_side.get("channel_meanings_tree"), path)
        units = _get(spec_side.get("channel_units_tree"), path)
        leaf_meaning = _get(spec_side.get("leaf_meanings_tree"), path)
        leaf_unit = _get(spec_side.get("leaf_units_tree"), path)
        lo = _get(min_tree, path)
        hi = _get(max_tree, path)
        lo_arr = None if lo is None else np.asarray(lo, dtype=np.float64)
        hi_arr = None if hi is None else np.asarray(hi, dtype=np.float64)
        if leaf_meaning:
            notes.append(f"`{leaf}`: {_cell(leaf_meaning)}")
        if isinstance(names, (list, tuple)) and names:
            for index, name in enumerate(names):
                row = [
                    f"`{leaf}`",
                    f"`{name}`",
                    _cell(_item(meanings, index)),
                    _cell(_item(units, index) or leaf_unit),
                ]
                if with_domain:
                    row.append(
                        _range(
                            _item(
                                None if lo_arr is None else lo_arr.ravel().tolist(),
                                index,
                            ),
                            _item(
                                None if hi_arr is None else hi_arr.ravel().tolist(),
                                index,
                            ),
                        )
                    )
                rows.append(row)
        else:
            row = [f"`{leaf}`", "—", "—", _cell(leaf_unit)]
            if with_domain:
                if lo_arr is not None and hi_arr is not None and lo_arr.size == 1:
                    row.append(_range(lo_arr.item(), hi_arr.item()))
                elif lo_arr is not None and hi_arr is not None:
                    row.append(_range(lo_arr.min(), hi_arr.max()))
                else:
                    row.append("—")
            rows.append(row)
    return rows, notes


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _example_inputs(domain: Mapping[str, Any] | None) -> str | None:
    """A runnable physical-input literal at the domain midpoint, if possible."""
    if not domain:
        return None
    min_tree, max_tree = domain.get("min_tree"), domain.get("max_tree")
    if not isinstance(min_tree, Mapping) or not isinstance(max_tree, Mapping):
        return None
    parts: list[str] = []
    for path in _leaf_paths(min_tree):
        if len(path) != 1:
            return None  # nested trees: leave the literal to the author
        lo = np.asarray(_get(min_tree, path), dtype=np.float64)
        hi = np.asarray(_get(max_tree, path), dtype=np.float64)
        if lo.ndim == 0:
            # Full precision: rounded endpoints can fall outside the domain,
            # and wavelength-explicit models are precision sensitive.
            values = f"np.linspace({float(lo)!r}, {float(hi)!r}, 5)[None, :]"
        else:
            mid = ", ".join(_fmt_float(v) for v in ((lo + hi) / 2.0).ravel())
            values = f"np.array([[{mid}]])"
        parts.append(f'    "{path[0]}": {values},')
    return "x_physical = {  # domain midpoint\n" + "\n".join(parts) + "\n}"


def _usage(
    *,
    repo_id: str | None,
    revision: str | None,
    has_input_scaling: bool,
    has_output_scaling: bool,
    domain: Mapping[str, Any] | None,
    needs_x64: bool,
) -> list[str]:
    repo = repo_id or "<owner>/<repo>"
    pin = f', revision="{revision}"' if revision else ""
    imports = ["normalize_tree"] if has_input_scaling else []
    if has_output_scaling:
        imports.append("denormalize_tree")
    code: list[str] = []
    if needs_x64:
        code += [
            "import os",
            'os.environ.setdefault("JAX_ENABLE_X64", "1")  # before JAX is imported',
            "",
        ]
    code += [
        "import numpy as np",
        "from astro_emulators_toolkit import " + ", ".join(["Emulator", *imports]),
        "",
        f'emu = Emulator.from_pretrained("{repo}"{pin})',
    ]
    example = _example_inputs(domain)
    code += [example] if example else ["x_physical = ...  # see the Interface table"]
    if has_input_scaling:
        code += [
            "ref = emu.reference_scaling_inputs",
            'y = emu.predict(normalize_tree(x_physical, ref["min_tree"], ref["max_tree"]))',
        ]
    else:
        code.append("y = emu.predict(x_physical)")
    if has_output_scaling:
        code += [
            "out = emu.reference_scaling_outputs",
            'y_physical = denormalize_tree(y, out["min_tree"], out["max_tree"])',
        ]
    lines = ["## Usage", ""]
    if revision:
        lines += [
            f'Pin the release you validated against (`revision="{revision}"`); '
            "without it you get the latest version on `main`.",
            "",
        ]
    lines += ["```python", *code, "```"]
    if has_input_scaling:
        lines += [
            "",
            "The toolkit never rescales silently: map physical inputs to the "
            "model representation with the recorded `reference_scaling_inputs`, "
            "as above.",
        ]
    return lines


def render_generated_section(
    bundle_dir: str | Path,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
) -> str:
    """The generated part of the card, from the bundle's own metadata."""
    from ..emulator import Emulator
    from .integrity import _load_bundle_integrity_manifest
    from .versions import minimum_loading_toolkit_version

    root = Path(bundle_dir)
    emu = Emulator.from_bundle(root)
    metadata = emu.bundle_metadata or {}
    family = str(metadata.get("model_family_id") or "unknown")
    needs_x64 = family.startswith(_X64_FAMILY_PREFIXES)
    domain = emu.input_domain
    spec = metadata.get("spec") or {}

    lines = [
        GENERATED_BEGIN,
        "<!-- Generated by `aemu card` from the bundle metadata. Edits inside "
        "this block are overwritten; write your own text above it. -->",
        "",
        "## Interface",
        "",
        "Inputs and outputs are canonical dict trees with a leading batch axis.",
        "",
    ]
    input_rows, input_notes = _interface_rows(
        spec.get("inputs") or {}, domain, with_domain=True
    )
    lines += _table(
        ["Input", "Channel", "Meaning", "Units", "Intended domain (physical)"],
        input_rows,
    )
    output_rows, output_notes = _interface_rows(
        spec.get("outputs") or {}, None, with_domain=False
    )
    lines += [""]
    lines += _table(["Output", "Channel", "Meaning", "Units"], output_rows)
    notes = input_notes + output_notes
    if notes:
        lines += ["", *[f"- {note}" for note in notes]]
    if domain is None:
        lines += [
            "",
            "This bundle declares no `input_domain`, so its intended-use "
            "bounds are not recorded.",
        ]
    lines += [""]
    lines += _usage(
        repo_id=repo_id,
        revision=revision,
        has_input_scaling=emu.reference_scaling_inputs is not None,
        has_output_scaling=emu.reference_scaling_outputs is not None,
        domain=domain,
        needs_x64=needs_x64,
    )

    minimum = minimum_loading_toolkit_version(dict(metadata))
    lines += ["", "## Requirements", ""]
    lines.append(
        f'- `pip install "astro-emulators-toolkit>={minimum}"`'
        if minimum
        else "- astro-emulators-toolkit (minimum version unknown for this bundle)"
    )
    if needs_x64:
        lines.append(
            "- `JAX_ENABLE_X64=1` set before JAX is imported (this model family "
            "handles wavelengths in float64)."
        )

    release = metadata.get("release") or {}
    provenance = metadata.get("provenance") or {}
    deps = provenance.get("dependencies") or {}
    saved_with = f"astro-emulators-toolkit {provenance.get('toolkit_version', '?')}"
    dep_text = ", ".join(f"{k} {v}" for k, v in deps.items() if v)
    if dep_text:
        saved_with += f"; {dep_text}"
    if provenance.get("python_version"):
        saved_with += f"; Python {provenance['python_version']}"
    fingerprint = metadata.get("fingerprint_evaluation")
    if isinstance(fingerprint, Mapping):
        fingerprint_text = (
            f"stored (rtol={fingerprint.get('rtol')}, atol={fingerprint.get('atol')}); "
            "a reproducibility check, best reproduced on a similar platform"
        )
    else:
        fingerprint_text = "none"
    stamps = (
        f"bundle format {metadata.get('bundle_format_version')}, config schema "
        f"{metadata.get('config_schema_version')}, spec {spec.get('spec_version')}, "
        f"weights layout `{metadata.get('weights_layout')}`"
    )
    lines += ["", "## Release and provenance", ""]
    lines += _table(
        ["", ""],
        [
            [
                "Release",
                _cell(
                    f"{release.get('name')} {release.get('version')} ({release.get('status')})"
                    if release
                    else None
                ),
            ],
            ["Bundle id", f"`{metadata.get('bundle_id')}`"],
            ["Model family", f"`{family}`"],
            ["Saved with", _cell(saved_with)],
            ["Saved on", _cell(provenance.get("platform"))],
            ["Saved at", _cell(provenance.get("created_at"))],
            ["Fingerprint", _cell(fingerprint_text)],
            ["Format", _cell(stamps)],
        ],
    )

    manifest = _load_bundle_integrity_manifest(root)
    contract = sorted(
        [entry["path"] for entry in manifest["tree"]] + ["bundle_integrity.json"]
    )
    lines += [
        "",
        "## Files",
        "",
        "Bundle contract, verified on every load (do not edit these files): "
        + ", ".join(f"`{path}`" for path in contract)
        + ". Every other file in this repository is documentation.",
        "",
        GENERATED_END,
    ]
    return "\n".join(lines) + "\n"


def _split_front_matter(text: str) -> tuple[dict[str, Any] | None, str]:
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text
    import yaml  # type: ignore[import-untyped]

    data = yaml.safe_load(text[4:end]) or {}
    if not isinstance(data, dict):
        raise ValueError("Model card front matter must be a YAML mapping.")
    return data, text[end + len("\n---\n") :]


def _join_front_matter(data: Mapping[str, Any], body: str) -> str:
    import yaml  # type: ignore[import-untyped]

    dumped = str(yaml.safe_dump(dict(data), sort_keys=False))
    return "---\n" + dumped + "---\n" + body


def card_front_matter(text: str) -> dict[str, Any] | None:
    """The card's YAML front matter, or None when it has none."""
    return _split_front_matter(text)[0]


def update_model_card(text: str, generated_section: str) -> str:
    """Replace the generated block and make sure ``library_name`` is set.

    The author's text is kept verbatim. A card without the generated
    markers gets the block appended at the end.
    """
    front, body = _split_front_matter(text)
    front = dict(front or {})
    front.setdefault("library_name", LIBRARY_NAME)
    begin = body.find(GENERATED_BEGIN)
    end = body.find(GENERATED_END)
    if begin >= 0 and end > begin:
        body = (
            body[:begin]
            + generated_section.rstrip("\n")
            + body[end + len(GENERATED_END) :]
        )
    else:
        body = body.rstrip("\n") + "\n\n" + generated_section
    return _join_front_matter(front, body)


def default_model_card(
    bundle_dir: str | Path,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
) -> str:
    """A new card: front matter, author prompts, and the generated block."""
    from ..emulator import Emulator

    metadata = Emulator.from_bundle(Path(bundle_dir)).bundle_metadata or {}
    family = str(metadata.get("model_family_id") or "emulator")
    family_tag = family.rsplit("_v", 1)[0].replace("_", "-")
    release = metadata.get("release") or {}
    front = {
        "library_name": LIBRARY_NAME,
        "tags": ["astronomy", "emulator", LIBRARY_NAME, family_tag],
    }
    body = [f"# {release.get('name') or Path(bundle_dir).name}", ""]
    for heading, prompt in _AUTHOR_SECTIONS:
        if heading:
            body += [f"## {heading}", ""]
        body += [f"{TODO_MARKER}: {prompt} -->", ""]
    text = _join_front_matter(front, "\n" + "\n".join(body))
    return update_model_card(
        text, render_generated_section(bundle_dir, repo_id=repo_id, revision=revision)
    )


def write_model_card(
    bundle_dir: str | Path,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
) -> Path:
    """Create ``README.md`` in a bundle directory, or refresh its generated block.

    ``repo_id`` and ``revision`` only fill in the usage snippet; ``aemu
    publish`` passes the real values.
    """
    path = Path(bundle_dir) / CARD_FILENAME
    if path.exists():
        text = update_model_card(
            path.read_text(),
            render_generated_section(bundle_dir, repo_id=repo_id, revision=revision),
        )
    else:
        text = default_model_card(bundle_dir, repo_id=repo_id, revision=revision)
    path.write_text(text)
    return path


__all__ = [
    "CARD_FILENAME",
    "GENERATED_BEGIN",
    "GENERATED_END",
    "LIBRARY_NAME",
    "TODO_MARKER",
    "card_front_matter",
    "default_model_card",
    "render_generated_section",
    "update_model_card",
    "write_model_card",
]

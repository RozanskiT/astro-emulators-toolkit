"""Publish a released bundle directory to the Hugging Face Hub.

What gets uploaded
------------------

A bundle directory holds three kinds of files:

- **Contract**: the files listed in ``bundle_integrity.json`` plus the
  manifest itself (``config.json``, ``metadata.json``, ``README.txt``,
  ``weights/``, and the sidecars the metadata refers to). They are verified
  on every load and must not be edited after ``prepare_bundle_release``.
- **Documentation**: every other file the author adds, such as the
  ``README.md`` model card, ``LICENSE``, ``CITATION.cff`` or a ``docs/``
  folder. Loading ignores them. They must not sit inside a contract folder
  (``weights/``, ``extras/``, ``fingerprint_evaluation/``).
- **Skipped**: hidden files and folders, ``__pycache__``, ``*.pyc``, and
  training state (``checkpoints/``, ``checkpoints_best/``, ``runs/``), which
  is not part of the sharing contract.

Versions on the Hub
-------------------

Each release is one commit on ``main`` plus the tag ``v<release_version>``
(the version ``prepare_bundle_release`` recorded). Tags are not moved unless
``overwrite_tag=True``. A release must not be older than the latest tagged
one, and a change of the interface (input/output structure, channel names,
units) since the latest release needs a new major version. ``tag=False``
publishes a documentation-only update to ``main``; it is refused when any
contract file differs from what ``main`` already has.

Hub support in ``Emulator`` itself stays download-only
(``Emulator.from_pretrained``); publishing lives here and in the ``aemu``
command.
"""

from __future__ import annotations

import json
import re
import tempfile
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bundle import BUNDLE_INTEGRITY_FILENAME, METADATA_FILENAME
from .card import (
    CARD_FILENAME,
    GENERATED_BEGIN,
    LIBRARY_NAME,
    TODO_MARKER,
    card_front_matter,
    render_generated_section,
    update_model_card,
)

_SKIPPED_DIR_NAMES = frozenset(
    {"__pycache__", "checkpoints", "checkpoints_best", "runs"}
)
_SKIPPED_SUFFIXES = (".pyc",)
# Files the Hub manages itself; never deleted by mirroring.
_HUB_MANAGED = frozenset({".gitattributes"})
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_INTERFACE_KEYS = (
    "structure_tree",
    "channel_names_tree",
    "channel_units_tree",
    "leaf_units_tree",
)


@dataclass(frozen=True)
class Finding:
    """One result of a pre-publish check: ``level`` is "error" or "warning"."""

    level: str
    message: str


@dataclass(frozen=True)
class UploadPlan:
    """How the files in a bundle directory are treated by ``publish_bundle``."""

    bundle_dir: Path
    contract: tuple[str, ...]
    documentation: tuple[str, ...]
    skipped: tuple[str, ...]
    misplaced: tuple[str, ...]

    @property
    def uploads(self) -> tuple[str, ...]:
        return self.contract + self.documentation


@dataclass
class PublishResult:
    repo_id: str
    tag: str | None
    commit: str | None
    dry_run: bool
    plan: UploadPlan
    findings: list[Finding] = field(default_factory=list)
    deleted: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def url(self) -> str:
        suffix = f"/tree/{self.tag}" if self.tag else ""
        return f"https://huggingface.co/{self.repo_id}{suffix}"


class PublishError(RuntimeError):
    """Raised when publishing is refused; ``findings`` holds the reasons."""

    def __init__(self, findings: list[Finding]):
        self.findings = findings
        errors = "\n".join(f"  - {f.message}" for f in findings if f.level == "error")
        super().__init__(f"Publishing refused:\n{errors}")


def _parse_version(text: Any) -> tuple[int, int, int] | None:
    match = _VERSION_RE.match(str(text).strip())
    if match is None:
        return None
    major, minor, patch = (int(group) for group in match.groups())
    return major, minor, patch


def release_tag(version: str) -> str:
    """The Hub tag for a release version: ``"1.2.0"`` becomes ``"v1.2.0"``."""
    return f"v{version}"


def _tag_version(tag: str) -> tuple[int, int, int] | None:
    return _parse_version(tag[1:]) if tag.startswith("v") else None


def plan_bundle_upload(bundle_dir: str | Path) -> UploadPlan:
    """Sort the files of a bundle directory into contract/documentation/skipped."""
    from .integrity import _load_bundle_integrity_manifest

    root = Path(bundle_dir)
    manifest = _load_bundle_integrity_manifest(root)
    contract = {str(entry["path"]) for entry in manifest["tree"]}
    contract.add(BUNDLE_INTEGRITY_FILENAME)
    contract_dirs = {path.split("/")[0] for path in contract if "/" in path}
    documentation: list[str] = []
    skipped: list[str] = []
    misplaced: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in contract:
            continue
        parts = rel.split("/")
        if (
            any(part.startswith(".") for part in parts)
            or any(part in _SKIPPED_DIR_NAMES for part in parts[:-1])
            or rel.endswith(_SKIPPED_SUFFIXES)
        ):
            skipped.append(rel)
        elif len(parts) > 1 and parts[0] in contract_dirs:
            misplaced.append(rel)
        else:
            documentation.append(rel)
    return UploadPlan(
        bundle_dir=root,
        contract=tuple(sorted(contract)),
        documentation=tuple(documentation),
        skipped=tuple(skipped),
        misplaced=tuple(misplaced),
    )


def _leaf_paths(tree: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            yield from _leaf_paths(value, prefix + (str(key),))
    else:
        yield prefix


def _get(tree: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        if not isinstance(tree, Mapping) or part not in tree:
            return None
        tree = tree[part]
    return tree


def _metadata_findings(metadata: Mapping[str, Any], has_domain: bool) -> list[Finding]:
    findings: list[Finding] = []
    spec = metadata.get("spec") or {}
    for side in ("inputs", "outputs"):
        block = spec.get(side) or {}
        for path in _leaf_paths(block.get("structure_tree") or {}):
            leaf = "/".join(path)
            named = _get(block.get("channel_names_tree"), path) or _get(
                block.get("leaf_meanings_tree"), path
            )
            if not named:
                findings.append(
                    Finding(
                        "warning",
                        f"{side} leaf `{leaf}` records no channel names or meaning; "
                        "users will only see its key.",
                    )
                )
            if not (
                _get(block.get("channel_units_tree"), path)
                or _get(block.get("leaf_units_tree"), path)
            ):
                findings.append(
                    Finding("warning", f"{side} leaf `{leaf}` records no units.")
                )
    if not has_domain:
        findings.append(
            Finding(
                "warning",
                "No input_domain: the intended-use bounds are not recorded, so "
                "the card cannot show them.",
            )
        )
    return findings


def _card_findings(text: str | None) -> list[Finding]:
    if text is None:
        return [
            Finding(
                "error",
                f"No {CARD_FILENAME} model card: run `aemu card DIR`, then "
                "replace its prompts with your own text.",
            )
        ]
    findings: list[Finding] = []
    try:
        front = card_front_matter(text)
    except ValueError as exc:
        return [Finding("error", f"{CARD_FILENAME}: {exc}")]
    if not front or front.get("library_name") != LIBRARY_NAME:
        findings.append(
            Finding(
                "error",
                f"{CARD_FILENAME} front matter must set `library_name: "
                f"{LIBRARY_NAME}` (`aemu card` adds it).",
            )
        )
    todos = text.count(TODO_MARKER)
    if todos:
        findings.append(
            Finding(
                "error",
                f"{CARD_FILENAME} still has {todos} unfilled `aemu:todo` "
                "prompt(s); replace each with your own text (or delete it).",
            )
        )
    if GENERATED_BEGIN not in text:
        findings.append(
            Finding(
                "warning",
                f"{CARD_FILENAME} has no generated block; `aemu card` or "
                "`aemu publish` appends it.",
            )
        )
    if front and not front.get("license"):
        findings.append(
            Finding(
                "warning",
                f"No `license:` in the {CARD_FILENAME} front matter "
                "(for example `license: cc-by-4.0`).",
            )
        )
    return findings


def check_bundle_for_publishing(
    bundle_dir: str | Path,
    *,
    allow_fingerprint_mismatch: bool = False,
    card_text: str | None = None,
) -> list[Finding]:
    """Run the pre-publish checklist on a bundle directory.

    Errors block ``publish_bundle``; warnings do not. ``card_text`` checks
    a card that is not written to disk yet (``publish_bundle`` passes the
    refreshed card); by default the directory's ``README.md`` is read.
    """
    from ..config.migrations import LegacyConfigWarning
    from ..emulator import Emulator
    from .release import verify_bundle_fingerprint_evaluation

    root = Path(bundle_dir)
    findings: list[Finding] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            emu = Emulator.from_bundle(root)
        except Exception as exc:  # any load failure blocks publishing
            return [Finding("error", f"The bundle does not load: {exc}")]
    for warning in caught:
        if issubclass(warning.category, LegacyConfigWarning):
            findings.append(
                Finding(
                    "warning",
                    "The bundle uses an older config schema and loads only via "
                    "an in-memory upgrade; consider re-saving it with the "
                    "current toolkit before publishing.",
                )
            )
        else:
            warnings.warn(warning.message, warning.category, stacklevel=2)
    metadata = emu.bundle_metadata or {}

    release = metadata.get("release")
    version = release.get("version") if isinstance(release, Mapping) else None
    if not version:
        findings.append(
            Finding(
                "error",
                "The bundle has no release identity: run prepare_bundle_release("
                "..., release_name=..., release_version=...) first. The release "
                "version becomes the Hub tag.",
            )
        )
    elif _parse_version(version) is None:
        findings.append(
            Finding(
                "error",
                f"release_version {version!r} is not MAJOR.MINOR.PATCH, which Hub "
                "tags and version ordering need.",
            )
        )

    if not metadata.get("fingerprint_evaluation"):
        findings.append(
            Finding(
                "error",
                "The bundle has no release fingerprint; prepare_bundle_release adds one.",
            )
        )
    else:
        try:
            verify_bundle_fingerprint_evaluation(root)
        except ValueError as exc:
            findings.append(
                Finding(
                    "warning" if allow_fingerprint_mismatch else "error",
                    f"The release fingerprint does not reproduce here ({exc}). If "
                    "it was recorded on another platform this can be float32 "
                    "numerics; allow_fingerprint_mismatch (--allow-fingerprint-"
                    "mismatch) publishes anyway.",
                )
            )

    plan = plan_bundle_upload(root)
    if plan.misplaced:
        findings.append(
            Finding(
                "error",
                "Documentation files inside contract folders: "
                + ", ".join(plan.misplaced)
                + ". Move them out (for example into docs/).",
            )
        )
    if plan.skipped:
        findings.append(
            Finding("warning", "Not uploaded: " + ", ".join(plan.skipped) + ".")
        )

    if card_text is None and (root / CARD_FILENAME).exists():
        card_text = (root / CARD_FILENAME).read_text()
    findings += _card_findings(card_text)
    findings += _metadata_findings(metadata, has_domain=emu.input_domain is not None)
    return findings


def interface_signature(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The parts of a spec whose change breaks users' code."""
    signature = {
        side: {key: (spec.get(side) or {}).get(key) for key in _INTERFACE_KEYS}
        for side in ("inputs", "outputs")
    }
    normalized: dict[str, Any] = json.loads(
        json.dumps(signature, sort_keys=True, default=str)
    )
    return normalized


def _interface_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    return [
        f"{side}.{key}"
        for side in ("inputs", "outputs")
        for key in _INTERFACE_KEYS
        if (old.get(side) or {}).get(key) != (new.get(side) or {}).get(key)
    ]


def _download_json(api: Any, repo_id: str, filename: str, revision: str) -> Any:
    with tempfile.TemporaryDirectory() as tmp:
        path = api.hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision, cache_dir=tmp
        )
        return json.loads(Path(path).read_text())


def _versioning_findings(
    api: Any,
    repo_id: str,
    metadata: Mapping[str, Any],
    tag: str,
    remote_tags: Mapping[str, str],
    *,
    overwrite_tag: bool,
    allow_interface_change: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    if tag in remote_tags and not overwrite_tag:
        findings.append(
            Finding(
                "error",
                f"Tag {tag} already exists on {repo_id}. Bump release_version "
                "(prepare_bundle_release), or pass overwrite_tag "
                "(--overwrite-tag) to replace it deliberately.",
            )
        )
    new_version = _parse_version(str(tag)[1:])
    released: list[tuple[tuple[int, int, int], str]] = []
    for name in remote_tags:
        parsed = _tag_version(name)
        if parsed is not None and name != tag:
            released.append((parsed, name))
    if not released or new_version is None:
        return findings
    latest_version, latest_tag = max(released)
    if new_version < latest_version:
        findings.append(
            Finding(
                "error",
                f"{tag} is older than the latest release {latest_tag}; publishing "
                "it would move main backwards. Use a higher release_version.",
            )
        )
        return findings
    previous = _download_json(api, repo_id, METADATA_FILENAME, revision=latest_tag)
    changes = _interface_changes(
        interface_signature(previous.get("spec") or {}),
        interface_signature(metadata.get("spec") or {}),
    )
    if changes and new_version[0] <= latest_version[0]:
        findings.append(
            Finding(
                "warning" if allow_interface_change else "error",
                f"The interface changed since {latest_tag} ({', '.join(changes)}). "
                f"That needs a new major version ({latest_version[0] + 1}.0.0); "
                "allow_interface_change (--allow-interface-change) publishes "
                f"it as {tag} anyway.",
            )
        )
    return findings


def _docs_only_findings(
    api: Any, repo_id: str, root: Path, exists: bool
) -> list[Finding]:
    if not exists:
        return [
            Finding(
                "error",
                f"{repo_id} does not exist yet; publish the first release with a tag.",
            )
        ]
    local_id = json.loads((root / BUNDLE_INTEGRITY_FILENAME).read_text()).get(
        "bundle_id"
    )
    try:
        remote = _download_json(api, repo_id, BUNDLE_INTEGRITY_FILENAME, "main")
    except Exception as exc:  # missing file or network error
        return [Finding("error", f"Could not read the bundle on main: {exc}")]
    if remote.get("bundle_id") != local_id:
        return [
            Finding(
                "error",
                "A documentation-only update (no tag) must not change the bundle, "
                f"but its bundle_id differs from main ({local_id} vs "
                f"{remote.get('bundle_id')}). Publish a new release instead.",
            )
        ]
    return []


def _verify_published(
    api: Any,
    repo_id: str,
    revision: str,
    root: Path,
    *,
    allow_fingerprint_mismatch: bool,
) -> list[Finding]:
    from ..emulator import Emulator
    from .release import verify_bundle_fingerprint_evaluation

    local_id = json.loads((root / BUNDLE_INTEGRITY_FILENAME).read_text()).get(
        "bundle_id"
    )
    with tempfile.TemporaryDirectory() as tmp:
        downloaded = Path(
            api.snapshot_download(repo_id=repo_id, revision=revision, cache_dir=tmp)
        )
        try:
            Emulator.from_bundle(downloaded)
        except Exception as exc:
            return [Finding("error", f"The published copy does not load: {exc}")]
        remote_id = json.loads(
            (downloaded / BUNDLE_INTEGRITY_FILENAME).read_text()
        ).get("bundle_id")
        if remote_id != local_id:
            return [
                Finding(
                    "error",
                    f"The published bundle_id {remote_id} differs from the local "
                    f"{local_id}.",
                )
            ]
        try:
            verify_bundle_fingerprint_evaluation(downloaded)
        except ValueError as exc:
            return [
                Finding(
                    "warning" if allow_fingerprint_mismatch else "error",
                    f"The published copy does not reproduce its fingerprint ({exc}).",
                )
            ]
    return []


_TOKEN_HELP = (
    "Create a token with write access at https://huggingface.co/settings/tokens "
    "(type 'Write', or a fine-grained token with 'Write access to contents/"
    "settings of all repos under your personal namespace' or of the target "
    "organization), then run `hf auth login` or set HF_TOKEN."
)


def _auth_findings(api: Any, repo_id: str, exists: bool) -> list[Finding]:
    """Can the current Hugging Face login write to ``repo_id``? Best effort."""
    try:
        who = api.whoami()
    except Exception:
        return [
            Finding(
                "error",
                "Not logged in to the Hugging Face Hub (no valid token found). "
                + _TOKEN_HELP,
            )
        ]
    token = (who.get("auth") or {}).get("accessToken") or {}
    role = token.get("role")
    name = (
        f"token {token.get('displayName')!r}" if token.get("displayName") else "token"
    )
    if role == "read":
        return [
            Finding(
                "error",
                f"Your Hugging Face {name} is read-only, so it cannot publish. "
                + _TOKEN_HELP,
            )
        ]
    if role != "fineGrained":
        return []
    fine = token.get("fineGrained") or {}
    namespace = repo_id.split("/")[0]
    wanted = repo_id if exists else namespace
    granted = set(fine.get("global") or [])
    for scope in fine.get("scoped") or []:
        if (scope.get("entity") or {}).get("name") in (wanted, namespace):
            granted.update(scope.get("permissions") or [])
    if "repo.write" in granted:
        return []
    action = (
        f"write to {repo_id}" if exists else f"create repositories under {namespace!r}"
    )
    return [
        Finding(
            "error",
            f"Your fine-grained Hugging Face {name} is not allowed to {action}. "
            + _TOKEN_HELP,
        )
    ]


def publish_bundle(
    bundle_dir: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    tag: bool = True,
    overwrite_tag: bool = False,
    mirror: bool = False,
    allow_interface_change: bool = False,
    allow_fingerprint_mismatch: bool = False,
    dry_run: bool = False,
    verify: bool = True,
    commit_message: str | None = None,
    api: Any = None,
    log: Callable[[str], None] = print,
) -> PublishResult:
    """Publish a released bundle directory to ``repo_id`` on the Hub.

    Uploads the contract and documentation files in one commit, tags it
    ``v<release_version>`` (unless ``tag=False``), then downloads the result
    and checks that it loads, has the same bundle_id, and reproduces its
    fingerprint. Before uploading, the generated block of ``README.md`` is
    refreshed for ``repo_id`` and the tag, and written back to the directory.

    With ``dry_run=True`` nothing is written; the returned result lists the
    plan and every finding. Otherwise errors raise ``PublishError``.
    ``mirror=True`` also deletes files on the Hub that the directory no
    longer has. Authentication uses your Hugging Face login or ``HF_TOKEN``.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    root = Path(bundle_dir)
    metadata = json.loads((root / METADATA_FILENAME).read_text())
    release = metadata.get("release") or {}
    version = release.get("version")
    tag_name = release_tag(str(version)) if tag and version else None

    exists = bool(api.repo_exists(repo_id))
    remote_tags: dict[str, str] = {}
    remote_files: list[str] = []
    if exists:
        refs = api.list_repo_refs(repo_id)
        remote_tags = {ref.name: ref.target_commit for ref in refs.tags}
        remote_files = list(api.list_repo_files(repo_id))

    findings: list[Finding] = []
    if tag_name is not None:
        findings += _versioning_findings(
            api,
            repo_id,
            metadata,
            tag_name,
            remote_tags,
            overwrite_tag=overwrite_tag,
            allow_interface_change=allow_interface_change,
        )
        card_revision: str | None = tag_name
    elif not tag:
        findings += _docs_only_findings(api, repo_id, root, exists)
        tagged = [(v, name) for name in remote_tags if (v := _tag_version(name))]
        card_revision = max(tagged)[1] if tagged else None
    else:
        card_revision = None  # no release version; the check below reports it

    card_path = root / CARD_FILENAME
    new_card: str | None = None
    if card_path.exists():
        try:
            new_card = update_model_card(
                card_path.read_text(),
                render_generated_section(root, repo_id=repo_id, revision=card_revision),
            )
        except ValueError as exc:
            findings.append(Finding("error", f"{CARD_FILENAME}: {exc}"))
    findings = (
        check_bundle_for_publishing(
            root,
            allow_fingerprint_mismatch=allow_fingerprint_mismatch,
            card_text=new_card,
        )
        + findings
    )

    plan = plan_bundle_upload(root)
    keep = set(plan.uploads) | _HUB_MANAGED
    stale = tuple(sorted(path for path in remote_files if path not in keep))
    deleted = stale if mirror else ()
    if stale and not mirror:
        findings.append(
            Finding(
                "warning",
                "Left on the Hub although not in this directory: "
                + ", ".join(stale)
                + " (mirror / --mirror deletes them).",
            )
        )
    result = PublishResult(
        repo_id=repo_id,
        tag=tag_name,
        commit=None,
        dry_run=dry_run,
        plan=plan,
        findings=findings,
        deleted=deleted,
        stale=stale,
    )
    findings += _auth_findings(api, repo_id, exists)
    if dry_run:
        return result
    if result.errors:
        raise PublishError(findings)

    if new_card is not None and new_card != card_path.read_text():
        card_path.write_text(new_card)
        log(f"Refreshed the generated block of {card_path}.")

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    operations: list[Any] = [
        CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(root / rel))
        for rel in plan.uploads
    ]
    operations += [CommitOperationDelete(path_in_repo=rel) for rel in deleted]
    message = commit_message or (
        f"Release {release.get('name')} {version}"
        if tag_name
        else "Update documentation"
    )
    from huggingface_hub.errors import HfHubHTTPError

    try:
        api.create_repo(repo_id, private=private, exist_ok=True)
        info = api.create_commit(repo_id, operations=operations, commit_message=message)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        reason = (getattr(exc, "server_message", None) or str(exc)).strip()
        hint = " " + _TOKEN_HELP if status in (401, 403) else ""
        raise PublishError(
            [Finding("error", f"The Hub refused the upload ({status}): {reason}{hint}")]
        ) from exc
    result.commit = str(info.oid)
    log(f"Committed {len(operations)} file operation(s) to {repo_id}: {result.commit}")
    if tag_name is not None:
        if tag_name in remote_tags:
            api.delete_tag(repo_id, tag=tag_name)
        api.create_tag(
            repo_id, tag=tag_name, revision=result.commit, tag_message=message
        )
        log(f"Tagged {result.commit[:12]} as {tag_name}.")
    if verify:
        result.findings += _verify_published(
            api,
            repo_id,
            tag_name or result.commit,
            root,
            allow_fingerprint_mismatch=allow_fingerprint_mismatch,
        )
    return result


__all__ = [
    "Finding",
    "PublishError",
    "PublishResult",
    "UploadPlan",
    "check_bundle_for_publishing",
    "interface_signature",
    "plan_bundle_upload",
    "publish_bundle",
    "release_tag",
]

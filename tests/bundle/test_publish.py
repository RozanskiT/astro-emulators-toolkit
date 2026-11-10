"""Model cards, pre-publish checks and Hub publishing, against an in-memory Hub."""

from __future__ import annotations

import json
import re
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.bundle import (
    PublishError,
    check_bundle_for_publishing,
    plan_bundle_upload,
    prepare_bundle_release,
    publish_bundle,
    write_model_card,
)
from astro_emulators_toolkit.bundle.card import (
    GENERATED_BEGIN,
    TODO_MARKER,
    render_generated_section,
)
from astro_emulators_toolkit.bundle.publish import (
    _versioning_findings,
    interface_signature,
)
from astro_emulators_toolkit.bundle.versions import minimum_loading_toolkit_version
from astro_emulators_toolkit.cli import main as aemu
from astro_emulators_toolkit.config import HubConfig

ASSETS = Path(__file__).resolve().parents[2] / "examples" / "assets"
DEV_FIXTURE = Path(__file__).parent / "fixtures" / "v0_2_0_dev0" / "mlp_bfloat16"
REPO = "someone/test-emulator"


class FakeHub:
    """The subset of huggingface_hub.HfApi that publish_bundle uses."""

    def __init__(self, root: Path):
        self.root = root
        self.repos: dict[str, dict] = {}

    @staticmethod
    def _oid(index: int) -> str:
        return f"{index + 1:040x}"

    def _index(self, repo_id: str, revision: str | None) -> int:
        repo = self.repos[repo_id]
        if revision in (None, "main"):
            return len(repo["commits"]) - 1
        if revision in repo["tags"]:
            return repo["tags"][revision]
        return int(revision, 16) - 1

    def files(self, repo_id: str, revision: str | None = "main") -> dict[str, bytes]:
        index = self._index(repo_id, revision)
        return self.repos[repo_id]["commits"][index] if index >= 0 else {}

    token: dict = {"role": "write"}

    def whoami(self):
        return {"name": "someone", "auth": {"accessToken": self.token}}

    def repo_exists(self, repo_id):
        return repo_id in self.repos

    def create_repo(self, repo_id, private=False, exist_ok=False):
        self.repos.setdefault(
            repo_id, {"commits": [{".gitattributes": b"*"}], "tags": {}}
        )

    def list_repo_refs(self, repo_id):
        tags = self.repos[repo_id]["tags"]
        return SimpleNamespace(
            tags=[
                SimpleNamespace(name=n, target_commit=self._oid(i))
                for n, i in tags.items()
            ]
        )

    def list_repo_files(self, repo_id):
        return sorted(self.files(repo_id))

    def create_commit(self, repo_id, operations, commit_message):
        files = dict(self.files(repo_id))
        for op in operations:
            if hasattr(op, "path_or_fileobj"):
                files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
            else:
                files.pop(op.path_in_repo)
        self.repos[repo_id]["commits"].append(files)
        return SimpleNamespace(oid=self._oid(len(self.repos[repo_id]["commits"]) - 1))

    def create_tag(self, repo_id, tag, revision, tag_message=None):
        assert tag not in self.repos[repo_id]["tags"], "tags must be deleted first"
        self.repos[repo_id]["tags"][tag] = self._index(repo_id, revision)

    def delete_tag(self, repo_id, tag):
        del self.repos[repo_id]["tags"][tag]

    def _materialize(self, repo_id, revision, target: Path) -> Path:
        for rel, data in self.files(repo_id, revision).items():
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            (target / rel).write_bytes(data)
        return target

    def hf_hub_download(self, repo_id, filename, revision, cache_dir):
        path = Path(cache_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files(repo_id, revision)[filename])
        return str(path)

    def snapshot_download(self, repo_id, revision, cache_dir):
        return str(self._materialize(repo_id, revision, Path(cache_dir) / "snapshot"))


def _fill_card(bundle_dir: Path) -> None:
    path = bundle_dir / "README.md"
    text = re.sub(r"<!-- aemu:todo: [^>]*-->", "Author text.", path.read_text())
    text = text.replace(
        "library_name: astro-emulators-toolkit\n",
        "library_name: astro-emulators-toolkit\nlicense: cc-by-4.0\n",
        1,
    )
    path.write_text(text)


@pytest.fixture
def released(tmp_path) -> Path:
    bundle = tmp_path / "bundle"
    shutil.copytree(ASSETS / "reference_bundle_release", bundle)
    write_model_card(bundle)
    _fill_card(bundle)
    return bundle


def _rerelease(bundle: Path, version: str) -> Path:
    return prepare_bundle_release(
        bundle,
        path=bundle.parent / f"release-{version}",
        release_name="payne-flux-reference-example",
        release_version=version,
    )


def _errors(findings) -> list[str]:
    return [f.message for f in findings if f.level == "error"]


def test_plan_sorts_contract_documentation_skipped_and_misplaced(released):
    (released / "docs").mkdir()
    (released / "docs" / "validation.md").write_text("residuals")
    (released / "checkpoints").mkdir()
    (released / "checkpoints" / "state").write_text("x")
    (released / ".DS_Store").write_text("x")
    (released / "extras" / "plot.png").write_text("x")

    plan = plan_bundle_upload(released)

    assert "weights/weights.safetensors" in plan.contract
    assert "bundle_integrity.json" in plan.contract
    assert plan.documentation == ("README.md", "docs/validation.md")
    assert set(plan.skipped) == {".DS_Store", "checkpoints/state"}
    assert plan.misplaced == ("extras/plot.png",)
    assert any(
        "contract folders" in m for m in _errors(check_bundle_for_publishing(released))
    )


def test_default_card_has_front_matter_prompts_and_generated_interface(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(ASSETS / "reference_bundle_release", bundle)

    text = write_model_card(bundle, repo_id=REPO, revision="v0.1.0").read_text()

    assert text.startswith("---\nlibrary_name: astro-emulators-toolkit\n")
    assert text.count(TODO_MARKER) == 7
    assert "| `parameters` | `teff` | effective temperature | K | 4500 – 7000 |" in text
    assert f'from_pretrained("{REPO}", revision="v0.1.0")' in text
    assert 'pip install "astro-emulators-toolkit>=0.1.0"' in text


def test_generated_usage_snippet_runs(released):
    section = render_generated_section(released, repo_id=REPO, revision="v0.1.0")
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    code = code.replace(
        f'Emulator.from_pretrained("{REPO}", revision="v0.1.0")',
        f'Emulator.from_bundle(r"{released}")',
    )
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 - generated, test-controlled code
    assert np.asarray(namespace["y_physical"]["flux"]).shape == (1, 500)


def test_card_refresh_keeps_author_text_and_front_matter(released):
    card = released / "README.md"
    card.write_text(card.read_text().replace("Author text.", "My validation story.", 1))

    write_model_card(released, repo_id="other/name", revision="v9.9.9")

    text = card.read_text()
    assert "My validation story." in text
    assert "license: cc-by-4.0" in text
    assert 'from_pretrained("other/name", revision="v9.9.9")' in text
    assert text.count(GENERATED_BEGIN) == 1


def test_check_lists_what_blocks_publishing(tmp_path, released):
    raw = tmp_path / "raw"
    shutil.copytree(ASSETS / "reference_bundle_raw", raw)
    raw_errors = " ".join(_errors(check_bundle_for_publishing(raw)))
    assert "no release identity" in raw_errors
    assert "no release fingerprint" in raw_errors
    assert "No README.md model card" in raw_errors

    write_model_card(raw)
    assert any("aemu:todo" in m for m in _errors(check_bundle_for_publishing(raw)))

    assert _errors(check_bundle_for_publishing(released)) == []


def test_dry_run_changes_nothing(released, tmp_path):
    hub = FakeHub(tmp_path)
    card_before = (released / "README.md").read_text()

    result = publish_bundle(released, REPO, dry_run=True, api=hub)

    assert (result.tag, result.commit, result.errors) == ("v0.1.0", None, [])
    assert hub.repos == {}
    assert (released / "README.md").read_text() == card_before


def test_first_release_uploads_tags_and_verifies(released, tmp_path):
    hub = FakeHub(tmp_path)

    result = publish_bundle(released, REPO, api=hub, log=lambda _: None)

    assert result.errors == []
    assert hub.repos[REPO]["tags"] == {"v0.1.0": 1}
    files = hub.files(REPO, "v0.1.0")
    assert set(files) == set(result.plan.uploads) | {".gitattributes"}
    card = files["README.md"].decode()
    assert f'from_pretrained("{REPO}", revision="v0.1.0")' in card
    assert card == (released / "README.md").read_text()  # local == Hub


def test_tags_are_not_moved_unless_asked(released, tmp_path):
    hub = FakeHub(tmp_path)
    publish_bundle(released, REPO, api=hub, log=lambda _: None)

    with pytest.raises(PublishError, match="already exists"):
        publish_bundle(released, REPO, api=hub, log=lambda _: None)

    publish_bundle(released, REPO, api=hub, overwrite_tag=True, log=lambda _: None)
    assert hub.repos[REPO]["tags"] == {"v0.1.0": 2}


def test_an_older_release_is_refused(released, tmp_path):
    hub = FakeHub(tmp_path)
    publish_bundle(_rerelease(released, "0.2.0"), REPO, api=hub, log=lambda _: None)

    with pytest.raises(PublishError, match="older than the latest release v0.2.0"):
        publish_bundle(released, REPO, api=hub, log=lambda _: None)


def test_docs_only_update_requires_an_unchanged_bundle(released, tmp_path):
    hub = FakeHub(tmp_path)
    publish_bundle(released, REPO, api=hub, log=lambda _: None)

    (released / "LICENSE").write_text("CC-BY-4.0")
    result = publish_bundle(released, REPO, api=hub, tag=False, log=lambda _: None)
    assert result.tag is None and result.errors == []
    assert "LICENSE" in hub.files(REPO) and "LICENSE" not in hub.files(REPO, "v0.1.0")

    with pytest.raises(PublishError, match="must not change the bundle"):
        publish_bundle(
            _rerelease(released, "0.2.0"), REPO, api=hub, tag=False, log=lambda _: None
        )


def test_mirror_deletes_removed_files_only_when_asked(released, tmp_path):
    hub = FakeHub(tmp_path)
    (released / "notes.md").write_text("draft")
    publish_bundle(released, REPO, api=hub, log=lambda _: None)
    (released / "notes.md").unlink()

    kept = publish_bundle(released, REPO, api=hub, tag=False, log=lambda _: None)
    assert kept.stale == ("notes.md",) and "notes.md" in hub.files(REPO)

    publish_bundle(released, REPO, api=hub, tag=False, mirror=True, log=lambda _: None)
    assert "notes.md" not in hub.files(REPO)
    assert ".gitattributes" in hub.files(REPO)


def test_an_interface_change_needs_a_major_version(released, tmp_path):
    metadata = json.loads((released / "metadata.json").read_text())
    previous = json.loads(json.dumps(metadata))
    previous["spec"]["inputs"]["channel_names_tree"]["parameters"][0] = "temperature"
    api = SimpleNamespace(
        hf_hub_download=lambda **kw: _write_json(tmp_path / "prev.json", previous)
    )
    assert interface_signature(previous["spec"]) != interface_signature(
        metadata["spec"]
    )

    def findings(tag, **flags):
        return _versioning_findings(
            api,
            REPO,
            metadata,
            tag,
            {"v1.0.0": "c"},
            overwrite_tag=False,
            allow_interface_change=flags.get("allow", False),
        )

    assert any("new major version (2.0.0)" in m for m in _errors(findings("v1.1.0")))
    assert _errors(findings("v2.0.0")) == []
    assert _errors(findings("v1.1.0", allow=True)) == []


def _write_json(path: Path, payload) -> str:
    path.write_text(json.dumps(payload))
    return str(path)


def test_cli_exit_codes(tmp_path, monkeypatch):
    # main() sets JAX_ENABLE_X64 for its own process; scope it to this test so
    # later tests that start example subprocesses do not inherit it.
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    bundle = tmp_path / "bundle"
    shutil.copytree(ASSETS / "reference_bundle_release", bundle)
    assert aemu(["card", str(bundle)]) == 0
    assert aemu(["check", str(bundle)]) == 1  # prompts not filled
    _fill_card(bundle)
    assert aemu(["check", str(bundle)]) == 0

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: FakeHub(tmp_path))
    assert aemu(["publish", str(bundle), REPO, "--dry-run"]) == 0


def test_from_pretrained_records_where_it_came_from(tmp_path, monkeypatch):
    commit = "ab" * 20
    snapshot = tmp_path / "snapshots" / commit
    shutil.copytree(ASSETS / "reference_bundle_release", snapshot)
    monkeypatch.setattr(
        "astro_emulators_toolkit.emulator.snapshot_download",
        lambda repo_id, revision=None, cache_dir=None: snapshot,
    )

    emu = Emulator.from_pretrained(REPO, revision="v0.1.0")

    assert emu.bundle_metadata["hub_source"] == {
        "repo_id": REPO,
        "requested_revision": "v0.1.0",
        "commit": commit,
    }


def test_hub_config_is_deprecated():
    with pytest.warns(DeprecationWarning, match="config.hub"):
        HubConfig(repo_id="owner/name")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        HubConfig()


def test_minimum_loading_toolkit_version():
    def metadata(path: Path) -> dict:
        return json.loads((path / "metadata.json").read_text())

    assert (
        minimum_loading_toolkit_version(metadata(ASSETS / "reference_bundle_release"))
        == "0.1.0"
    )
    # 0.2.0.dev0 stamped config schema v1 but 0.1.x cannot load its configs.
    assert minimum_loading_toolkit_version(metadata(DEV_FIXTURE)) == "0.2.0"
    current = {
        **metadata(ASSETS / "reference_bundle_release"),
        "config_schema_version": 2,
    }
    assert minimum_loading_toolkit_version(current) == "0.2.0"
    assert (
        minimum_loading_toolkit_version({**current, "config_schema_version": 99})
        is None
    )


def test_publishing_explains_a_token_that_cannot_write(released, tmp_path):
    hub = FakeHub(tmp_path)

    hub.token = {"role": "read", "displayName": "laptop"}
    with pytest.raises(PublishError, match="read-only"):
        publish_bundle(released, REPO, api=hub, log=lambda _: None)

    # Fine-grained token scoped to another repository: cannot create this one.
    hub.token = {
        "role": "fineGrained",
        "fineGrained": {
            "global": [],
            "scoped": [
                {"entity": {"name": "someone/other"}, "permissions": ["repo.write"]},
                {"entity": {"name": "someone"}, "permissions": []},
            ],
        },
    }
    dry = publish_bundle(released, REPO, api=hub, dry_run=True)
    assert any(
        "create repositories under 'someone'" in m for m in _errors(dry.findings)
    )
    assert hub.repos == {}

    hub.token["fineGrained"]["scoped"][1]["permissions"] = ["repo.write"]
    assert publish_bundle(released, REPO, api=hub, log=lambda _: None).errors == []

    def not_logged_in():
        raise OSError("no token")

    hub.whoami = not_logged_in
    with pytest.raises(PublishError, match="Not logged in"):
        publish_bundle(_rerelease(released, "0.2.0"), REPO, api=hub, log=lambda _: None)


def test_a_refused_upload_is_reported_without_a_traceback(released, tmp_path):
    from huggingface_hub.errors import HfHubHTTPError

    hub = FakeHub(tmp_path)

    def refuse(*args, **kwargs):
        response = SimpleNamespace(status_code=403, headers={}, request=None)
        raise HfHubHTTPError("403 Forbidden: no rights", response=response)

    hub.create_repo = refuse
    with pytest.raises(PublishError, match=r"refused the upload \(403\)"):
        publish_bundle(released, REPO, api=hub, log=lambda _: None)
    assert hub.repos == {}

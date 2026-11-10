"""Prepare, check and publish a released bundle on the Hugging Face Hub.

The Python-API version of `aemu card` / `aemu check` / `aemu publish`.

Data: the shipped released reference bundle `examples/assets/reference_bundle_release`.
Creates: `examples/development/runs/publish_demo/bundle/` (a copy with a README.md
model card). With `--repo-id` it also does a dry-run publish (read-only Hub
queries); with `--repo-id ... --publish` it creates the repository and uploads.
Runtime: a few seconds, plus upload time when publishing.
Requires: the base install; publishing needs a Hugging Face login (`hf auth login`).

    python examples/development/19_publish_bundle_to_hub.py
    python examples/development/19_publish_bundle_to_hub.py --repo-id YOU/aemu-demo
    python examples/development/19_publish_bundle_to_hub.py --repo-id YOU/aemu-demo --publish
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from astro_emulators_toolkit.bundle import (
    check_bundle_for_publishing,
    publish_bundle,
    write_model_card,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE = SCRIPT_DIR.parent / "assets" / "reference_bundle_release"
BUNDLE = SCRIPT_DIR / "runs" / "publish_demo" / "bundle"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", help="OWNER/NAME; enables the (dry-run) publish")
    parser.add_argument("--publish", action="store_true", help="really upload")
    args = parser.parse_args()

    shutil.rmtree(BUNDLE, ignore_errors=True)
    shutil.copytree(SOURCE, BUNDLE)

    # 1. The model card: front matter, your sections, a generated block.
    card = write_model_card(BUNDLE, repo_id=args.repo_id)
    # A real author replaces each prompt with their own text; the demo
    # fills them so that the checklist passes.
    text = re.sub(r"<!-- aemu:todo: [^>]*-->", "Demo text.", card.read_text())
    card.write_text(
        text.replace("library_name:", "license: cc-by-4.0\nlibrary_name:", 1)
    )
    print(f"Model card: {card}")

    # 2. The pre-publish checklist: errors block publishing.
    findings = check_bundle_for_publishing(BUNDLE)
    for finding in findings:
        print(f"{finding.level}: {finding.message}")
    blocked = any(finding.level == "error" for finding in findings)
    print(
        "Checklist: errors block publishing."
        if blocked
        else "Checklist: ready to publish."
    )

    # 3. Publish: one commit plus the tag v<release_version>, then a
    #    download-back verification. dry_run=True only reads from the Hub.
    if args.repo_id:
        result = publish_bundle(BUNDLE, args.repo_id, dry_run=not args.publish)
        for finding in result.findings:
            print(f"{finding.level}: {finding.message}")
        state = "Dry run for" if result.dry_run else "Published"
        print(f"{state} {args.repo_id} as {result.tag}: {result.url}")


if __name__ == "__main__":
    main()

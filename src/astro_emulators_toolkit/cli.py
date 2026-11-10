"""The ``aemu`` command: model cards, pre-publish checks, Hub publishing.

    aemu card DIR [--repo-id OWNER/NAME] [--revision vX.Y.Z]
    aemu check DIR [--allow-fingerprint-mismatch]
    aemu publish DIR OWNER/NAME [--dry-run] [--private] [--no-tag]
                 [--overwrite-tag] [--mirror] [--allow-interface-change]
                 [--allow-fingerprint-mismatch] [--no-verify] [--message TEXT]

Each subcommand wraps a function in ``astro_emulators_toolkit.bundle``
(``write_model_card``, ``check_bundle_for_publishing``, ``publish_bundle``),
so the same steps can be scripted from Python.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from typing import Any

from ._version import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aemu",
        description="astro-emulators-toolkit bundle tools: model cards, checks, "
        "and publishing to the Hugging Face Hub.",
    )
    parser.add_argument("--version", action="version", version=f"aemu {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    card = sub.add_parser(
        "card",
        help="create README.md for a bundle, or refresh its generated block",
    )
    card.add_argument("bundle_dir")
    card.add_argument("--repo-id", help="Hub repository shown in the usage snippet")
    card.add_argument("--revision", help="revision shown in the usage snippet")

    check = sub.add_parser("check", help="run the pre-publish checklist")
    check.add_argument("bundle_dir")
    check.add_argument("--allow-fingerprint-mismatch", action="store_true")

    publish = sub.add_parser(
        "publish", help="publish a released bundle directory to the Hub"
    )
    publish.add_argument("bundle_dir")
    publish.add_argument("repo_id", help="OWNER/NAME on the Hugging Face Hub")
    publish.add_argument(
        "--dry-run", action="store_true", help="show the plan; change nothing"
    )
    publish.add_argument(
        "--private", action="store_true", help="create the repository as private"
    )
    publish.add_argument(
        "--no-tag",
        action="store_true",
        help="documentation-only update of main; the bundle must be unchanged",
    )
    publish.add_argument(
        "--overwrite-tag",
        action="store_true",
        help="replace an existing v<release_version> tag",
    )
    publish.add_argument(
        "--mirror",
        action="store_true",
        help="also delete files on the Hub that this directory does not have",
    )
    publish.add_argument(
        "--allow-interface-change",
        action="store_true",
        help="publish an interface change without a major version bump",
    )
    publish.add_argument("--allow-fingerprint-mismatch", action="store_true")
    publish.add_argument(
        "--no-verify",
        action="store_true",
        help="skip downloading the published bundle back for verification",
    )
    publish.add_argument("--message", help="commit message")
    return parser


def _report(findings: Sequence[Any]) -> int:
    errors = 0
    for finding in findings:
        errors += finding.level == "error"
        print(f"{finding.level}: {finding.message}")
    return errors


def _print_plan(plan: Any) -> None:
    print(f"Bundle contract ({len(plan.contract)} files): " + ", ".join(plan.contract))
    print(
        "Documentation: "
        + (", ".join(plan.documentation) if plan.documentation else "none")
    )
    if plan.skipped:
        print("Skipped: " + ", ".join(plan.skipped))


def main(argv: Sequence[str] | None = None) -> int:
    # TransformerPayne bundles need float64 wavelengths; enable x64 before
    # anything imports JAX. Float32 models are unaffected.
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    args = _parser().parse_args(argv)

    from .bundle.card import TODO_MARKER, write_model_card
    from .bundle.publish import (
        PublishError,
        check_bundle_for_publishing,
        plan_bundle_upload,
        publish_bundle,
    )

    if args.command == "card":
        path = write_model_card(
            args.bundle_dir, repo_id=args.repo_id, revision=args.revision
        )
        print(f"Wrote {path}")
        todos = path.read_text().count(TODO_MARKER)
        if todos:
            print(
                f"Replace its {todos} `aemu:todo` prompt(s) with your own text, "
                "then run `aemu check`."
            )
        return 0

    if args.command == "check":
        _print_plan(plan_bundle_upload(args.bundle_dir))
        errors = _report(
            check_bundle_for_publishing(
                args.bundle_dir,
                allow_fingerprint_mismatch=args.allow_fingerprint_mismatch,
            )
        )
        print("Ready to publish." if not errors else f"{errors} error(s).")
        return 1 if errors else 0

    try:
        result = publish_bundle(
            args.bundle_dir,
            args.repo_id,
            private=args.private,
            tag=not args.no_tag,
            overwrite_tag=args.overwrite_tag,
            mirror=args.mirror,
            allow_interface_change=args.allow_interface_change,
            allow_fingerprint_mismatch=args.allow_fingerprint_mismatch,
            dry_run=args.dry_run,
            verify=not args.no_verify,
            commit_message=args.message,
        )
    except PublishError as exc:
        _report(exc.findings)
        print("Nothing was published.")
        return 1
    except Exception as exc:
        from huggingface_hub.errors import HfHubHTTPError

        if not isinstance(exc, (HfHubHTTPError, OSError)):
            raise
        # Network or Hub failures: a short message, not a traceback.
        print(f"error: could not reach or use the Hugging Face Hub: {exc}")
        return 1
    _print_plan(result.plan)
    if result.deleted:
        print("Delete on the Hub: " + ", ".join(result.deleted))
    errors = _report(result.findings)
    target = f"{result.repo_id} as {result.tag}" if result.tag else result.repo_id
    if result.dry_run:
        print(
            f"Dry run: would publish to {target}."
            if not errors
            else f"Dry run: {errors} error(s) would block publishing."
        )
    elif errors:
        print(f"Published to {target}, but verification found {errors} error(s).")
    else:
        print(f"Published to {target}: {result.url}")
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

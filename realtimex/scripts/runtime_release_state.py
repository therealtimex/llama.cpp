#!/usr/bin/env python3
"""Classify a RealTimeX runtime release from its manifest and asset names."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


RELEASE_RE = re.compile(r"^b\d+$")

# Releases published before explicit completeness metadata used this fixed matrix.
# Keep this list stable if future releases add runtime variants.
LEGACY_ASSET_TEMPLATES = (
    "llama-server-darwin-arm64-{tag}.zip",
    "llama-server-darwin-x64-{tag}.zip",
    "llama-server-linux-arm64-{tag}.zip",
    "llama-server-linux-arm64-cuda-{tag}.zip",
    "llama-server-linux-arm64-vulkan-{tag}.zip",
    "llama-server-linux-x64-{tag}.zip",
    "llama-server-linux-x64-cuda-{tag}.zip",
    "llama-server-linux-x64-vulkan-{tag}.zip",
    "llama-server-win32-arm64-{tag}.zip",
    "llama-server-win32-x64-{tag}.zip",
    "llama-server-win32-x64-cuda-{tag}.zip",
    "llama-server-win32-x64-vulkan-{tag}.zip",
)


@dataclass(frozen=True)
class RuntimeReleaseState:
    state: str
    expected_asset_names: list[str]
    missing_release_asset_names: list[str]
    missing_manifest_asset_names: list[str]
    reason: str = ""

    @property
    def complete(self) -> bool:
        return self.state == "complete"


def legacy_expected_asset_names(tag: str) -> list[str]:
    if not RELEASE_RE.match(tag):
        raise ValueError(f"invalid release tag {tag!r}")
    return sorted(template.format(tag=tag) for template in LEGACY_ASSET_TEMPLATES)


def _string_list(value: Any) -> Optional[list[str]]:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, str) and item for item in value):
        return None
    return sorted(set(value))


def _manifest_asset_names(manifest: dict[str, Any]) -> Optional[set[str]]:
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        return None
    names: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
            return None
        names.add(asset["name"])
    return names


def classify_runtime_release(
    tag: str,
    release_asset_names: set[str],
    manifest: Optional[dict[str, Any]],
) -> RuntimeReleaseState:
    """Return complete, incomplete, or invalid for locally available release data."""
    legacy_expected = legacy_expected_asset_names(tag)
    if manifest is None:
        missing_release = sorted(set(legacy_expected) - release_asset_names)
        return RuntimeReleaseState(
            state="incomplete",
            expected_asset_names=legacy_expected,
            missing_release_asset_names=missing_release,
            missing_manifest_asset_names=legacy_expected,
            reason="runtime-manifest.json is missing",
        )

    if manifest.get("tag") != f"realtimex-{tag}":
        return RuntimeReleaseState("invalid", [], [], [], "manifest tag is inconsistent")
    if manifest.get("llamaCppRelease") != tag:
        return RuntimeReleaseState(
            "invalid", [], [], [], "manifest llamaCppRelease is inconsistent"
        )

    manifest_asset_names = _manifest_asset_names(manifest)
    if manifest_asset_names is None:
        return RuntimeReleaseState("invalid", [], [], [], "manifest assets are invalid")

    if "complete" in manifest:
        if not isinstance(manifest["complete"], bool):
            return RuntimeReleaseState(
                "invalid", [], [], [], "manifest complete field is invalid"
            )
        expected = _string_list(manifest.get("expectedAssetNames"))
        if expected is None:
            return RuntimeReleaseState(
                "invalid", [], [], [], "manifest expectedAssetNames are invalid"
            )
        published_missing = manifest.get("missingAssetNames")
        if not isinstance(published_missing, list) or not all(
            isinstance(item, str) and item for item in published_missing
        ):
            return RuntimeReleaseState(
                "invalid", [], [], [], "manifest missingAssetNames are invalid"
            )
    else:
        expected = legacy_expected
        published_missing = []

    expected_set = set(expected)
    missing_release = sorted(expected_set - release_asset_names)
    missing_manifest = sorted(expected_set - manifest_asset_names)
    if manifest.get("complete") is False:
        return RuntimeReleaseState(
            "incomplete",
            expected,
            missing_release,
            missing_manifest,
            "manifest reports an assembling release",
        )
    if published_missing:
        return RuntimeReleaseState(
            "incomplete",
            expected,
            missing_release,
            missing_manifest,
            "manifest still reports missing runtime assets",
        )
    if missing_release or missing_manifest:
        return RuntimeReleaseState(
            "incomplete",
            expected,
            missing_release,
            missing_manifest,
            "required runtime assets are missing",
        )

    return RuntimeReleaseState("complete", expected, [], [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="llama.cpp bNNNN tag")
    parser.add_argument(
        "--assets-file",
        required=True,
        help="File containing one GitHub release asset name per line",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Downloaded runtime-manifest.json; omit when the asset is absent",
    )
    parser.add_argument("--github-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_names = {
        line.strip()
        for line in Path(args.assets_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    manifest = None
    if args.manifest:
        try:
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            state = RuntimeReleaseState("invalid", [], [], [], str(exc))
        else:
            if isinstance(manifest, dict):
                state = classify_runtime_release(args.tag, asset_names, manifest)
            else:
                state = RuntimeReleaseState(
                    "invalid", [], [], [], "manifest root must be an object"
                )
    else:
        state = classify_runtime_release(args.tag, asset_names, None)

    payload = asdict(state)
    payload["complete"] = state.complete
    print(json.dumps(payload, indent=2))

    if args.github_output:
        output_path = os.environ.get("GITHUB_OUTPUT")
        if not output_path:
            raise SystemExit("GITHUB_OUTPUT is required with --github-output")
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"release_state={state.state}\n")
            output.write(f"release_complete={'true' if state.complete else 'false'}\n")


if __name__ == "__main__":
    main()

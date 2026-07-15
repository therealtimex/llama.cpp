#!/usr/bin/env python3
"""Compare ggml-org/llama.cpp tags vs RealTimeX promoted releases.

Used by realtimex-watch-upstream.yml to decide whether to dispatch promote.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from runtime_release_state import (
    classify_runtime_release,
    legacy_expected_asset_names,
)

RELEASE_RE = re.compile(r"^b\d+$")
REALTIMEX_TAG_RE = re.compile(r"^realtimex-(b\d+)$")


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def github_api(
    path: str,
    token: str = "",
    accept: str = "application/vnd.github+json",
) -> Any:
    url = f"https://api.github.com{path}"
    headers = {
        "Accept": accept,
        "User-Agent": "therealtimex-llama-cpp-watch",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        die(f"GitHub API {path} failed: HTTP {exc.code}: {body[:300]}")


def parse_b_number(tag: str) -> int:
    return int(tag[1:])


def latest_upstream_tag(token: str, repo: str) -> str:
    """Prefer /releases/latest; fall back to scanning releases for bNNNN."""
    try:
        data = github_api(f"/repos/{repo}/releases/latest", token)
        tag = str(data.get("tag_name") or "")
        if RELEASE_RE.match(tag):
            return tag
    except SystemExit:
        pass

    releases = github_api(f"/repos/{repo}/releases?per_page=30", token)
    candidates = [
        str(r.get("tag_name") or "")
        for r in releases
        if RELEASE_RE.match(str(r.get("tag_name") or ""))
    ]
    if not candidates:
        die(f"no bNNNN releases found on {repo}")
    candidates.sort(key=parse_b_number, reverse=True)
    return candidates[0]


def latest_promoted_release(token: str, repo: str) -> Optional[dict[str, Any]]:
    """Return the latest realtimex-bNNNN release, or None."""
    releases = github_api(f"/repos/{repo}/releases?per_page=50", token)
    best: Optional[dict[str, Any]] = None
    best_n = -1
    for rel in releases:
        if rel.get("draft"):
            continue
        tag = str(rel.get("tag_name") or "")
        m = REALTIMEX_TAG_RE.match(tag)
        if not m:
            continue
        btag = m.group(1)
        n = parse_b_number(btag)
        if n > best_n:
            best_n = n
            best = rel
    return best


def latest_promoted_tag(token: str, repo: str) -> Optional[str]:
    """Return llama.cpp bNNNN from latest realtimex-bNNNN release, or None."""
    release = latest_promoted_release(token, repo)
    if release is None:
        return None
    match = REALTIMEX_TAG_RE.match(str(release.get("tag_name") or ""))
    return match.group(1) if match else None


def expected_runtime_asset_names(tag: str) -> set[str]:
    return set(legacy_expected_asset_names(tag))


def fetch_release_manifest(
    release: dict[str, Any], token: str
) -> tuple[str, Optional[dict[str, Any]]]:
    manifest_asset = next(
        (
            asset
            for asset in release.get("assets") or []
            if asset.get("name") == "runtime-manifest.json"
        ),
        None,
    )
    if not manifest_asset or not manifest_asset.get("url"):
        return "missing", None
    path = urllib.parse.urlparse(str(manifest_asset["url"])).path
    try:
        manifest = github_api(path, token, accept="application/octet-stream")
    except SystemExit:
        return "unknown", None
    if not isinstance(manifest, dict):
        return "unknown", None
    return "available", manifest


def runtime_release_state(release: dict[str, Any], token: str) -> str:
    match = REALTIMEX_TAG_RE.match(str(release.get("tag_name") or ""))
    if not match:
        return "unknown"
    asset_names = {
        str(asset.get("name") or "") for asset in release.get("assets") or []
    }
    fetch_state, manifest = fetch_release_manifest(release, token)
    if fetch_state == "unknown":
        return "unknown"
    classified = classify_runtime_release(match.group(1), asset_names, manifest)
    return "unknown" if classified.state == "invalid" else classified.state


def release_manifest_complete(release: dict[str, Any], token: str) -> Optional[bool]:
    state = runtime_release_state(release, token)
    if state == "unknown":
        return None
    return state == "complete"


def release_is_complete(release: dict[str, Any], token: str) -> bool:
    return runtime_release_state(release, token) == "complete"


def write_github_output(values: dict[str, str]) -> None:
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        return
    with open(out_path, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--upstream-repo",
        default="ggml-org/llama.cpp",
        help="Upstream llama.cpp repo",
    )
    p.add_argument(
        "--packaging-repo",
        default="therealtimex/llama.cpp",
        help="RealTimeX packaging repo (releases realtimex-b*)",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub token (optional for public reads; required if rate-limited)",
    )
    p.add_argument(
        "--github-output",
        action="store_true",
        help="Write updated/latest/promoted to GITHUB_OUTPUT",
    )
    p.add_argument(
        "--force-tag",
        default="",
        help="Check this bNNNN tag instead of the latest upstream release",
    )
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or ""

    forced = (args.force_tag or "").strip()
    if forced and not RELEASE_RE.match(forced):
        die(f"force tag must look like b10012, got {forced!r}")

    latest = forced or latest_upstream_tag(token, args.upstream_repo)
    promoted_release = latest_promoted_release(token, args.packaging_repo)
    promoted_match = (
        REALTIMEX_TAG_RE.match(str(promoted_release.get("tag_name") or ""))
        if promoted_release
        else None
    )
    promoted = promoted_match.group(1) if promoted_match else None

    target_release = None
    if promoted_release and promoted == latest:
        target_release = promoted_release
    elif forced:
        releases = github_api(
            f"/repos/{args.packaging_repo}/releases?per_page=50", token
        )
        target_tag = f"realtimex-{latest}"
        target_release = next(
            (release for release in releases if release.get("tag_name") == target_tag),
            None,
        )

    target_state = (
        runtime_release_state(target_release, token) if target_release else "absent"
    )
    if target_state == "unknown":
        die(f"could not determine release state for realtimex-{latest}")
    target_complete = target_state == "complete"
    repair_existing = target_state == "incomplete"
    if forced:
        updated = target_release is None or repair_existing
    else:
        updated = (
            promoted is None
            or parse_b_number(latest) > parse_b_number(promoted)
            or repair_existing
        )

    payload = {
        "upstreamRepo": args.upstream_repo,
        "packagingRepo": args.packaging_repo,
        "latestUpstream": latest,
        "latestPromoted": promoted or "",
        "updated": updated,
        "releaseComplete": target_complete,
        "releaseState": target_state,
        "repairExisting": repair_existing,
        "releaseTag": f"realtimex-{latest}",
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"upstream latest : {latest} ({args.upstream_repo})")
        print(f"promoted latest : {promoted or '(none)'} ({args.packaging_repo})")
        print(f"update needed   : {updated}")
        print(f"release complete: {target_complete}")
        print(f"release state   : {target_state}")
        print(f"repair existing : {repair_existing}")
        if updated:
            print(f"would promote   : {latest} -> release {payload['releaseTag']}")

    if args.github_output:
        write_github_output(
            {
                "updated": "true" if updated else "false",
                "releaseComplete": "true" if target_complete else "false",
                "releaseState": target_state,
                "repairExisting": "true" if repair_existing else "false",
                "latestUpstream": latest,
                "latestPromoted": promoted or "",
                "releaseTag": payload["releaseTag"],
            }
        )

    # Exit 0 always - workflow branches on outputs, not exit code.
    raise SystemExit(0)


if __name__ == "__main__":
    main()

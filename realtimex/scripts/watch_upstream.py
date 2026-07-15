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
import urllib.request
from typing import Any, Optional

RELEASE_RE = re.compile(r"^b\d+$")
REALTIMEX_TAG_RE = re.compile(r"^realtimex-(b\d+)$")


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def github_api(path: str, token: str = "") -> Any:
    url = f"https://api.github.com{path}"
    headers = {
        "Accept": "application/vnd.github+json",
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


def latest_promoted_tag(token: str, repo: str) -> Optional[str]:
    """Return llama.cpp bNNNN from latest realtimex-bNNNN release, or None."""
    releases = github_api(f"/repos/{repo}/releases?per_page=50", token)
    best: Optional[str] = None
    best_n = -1
    for rel in releases:
        tag = str(rel.get("tag_name") or "")
        m = REALTIMEX_TAG_RE.match(tag)
        if not m:
            continue
        btag = m.group(1)
        n = parse_b_number(btag)
        if n > best_n:
            best_n = n
            best = btag
    return best


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
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or ""

    latest = latest_upstream_tag(token, args.upstream_repo)
    promoted = latest_promoted_tag(token, args.packaging_repo)
    updated = promoted is None or parse_b_number(latest) > parse_b_number(promoted)

    payload = {
        "upstreamRepo": args.upstream_repo,
        "packagingRepo": args.packaging_repo,
        "latestUpstream": latest,
        "latestPromoted": promoted or "",
        "updated": updated,
        "releaseTag": f"realtimex-{latest}",
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"upstream latest : {latest} ({args.upstream_repo})")
        print(f"promoted latest : {promoted or '(none)'} ({args.packaging_repo})")
        print(f"update needed   : {updated}")
        if updated:
            print(f"would promote   : {latest} → release {payload['releaseTag']}")

    if args.github_output:
        write_github_output(
            {
                "updated": "true" if updated else "false",
                "latestUpstream": latest,
                "latestPromoted": promoted or "",
                "releaseTag": payload["releaseTag"],
            }
        )

    # Exit 0 always — workflow branches on outputs, not exit code.
    raise SystemExit(0)


if __name__ == "__main__":
    main()

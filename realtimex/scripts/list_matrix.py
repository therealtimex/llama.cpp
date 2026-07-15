#!/usr/bin/env python3
"""Print packaging matrix as JSON (for CI / debugging)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_packaging import OFFICIAL_MATRIX, SELF_BUILD_MATRIX, format_template, validate_release_tag  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="")
    p.add_argument("--kind", choices=("official", "self-build", "all"), default="all")
    args = p.parse_args()
    tag = validate_release_tag(args.tag) if args.tag else ""

    rows = []
    if args.kind in ("official", "all"):
        rows.extend(OFFICIAL_MATRIX)
    if args.kind in ("self-build", "all"):
        rows.extend(SELF_BUILD_MATRIX)

    if tag:
        expanded = []
        for row in rows:
            item = dict(row)
            if "upstream" in item:
                item["upstream"] = format_template(item["upstream"], tag)
            item["asset"] = format_template(item["asset"], tag)
            item["tag"] = tag
            expanded.append(item)
        rows = expanded

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

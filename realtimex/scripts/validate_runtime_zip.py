#!/usr/bin/env python3
"""Validate a RealTimeX llama-server runtime zip (meta + binary co-location)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_packaging import validate_runtime_zip  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", required=True, help="Path to llama-server-*.zip")
    p.add_argument("--expect-release", default="", help="Expected bNNNN release")
    p.add_argument("--expect-gpu", default="", help="Expected gpu: metal|cuda|vulkan|cpu|false")
    p.add_argument("--expect-platform", default="", help="darwin|linux|win32|mac|win")
    p.add_argument("--expect-arch", default="", help="arm64|x64|armv7l")
    p.add_argument("--json", action="store_true", help="Print result as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_runtime_zip(
        Path(args.zip),
        expect_release=args.expect_release,
        expect_gpu=args.expect_gpu,
        expect_platform=args.expect_platform,
        expect_arch=args.expect_arch,
    )
    payload = {
        "ok": result.ok,
        "metadataPath": result.metadata_path,
        "binaryPath": result.binary_path,
        "release": result.release,
        "gpu": result.gpu,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"OK {args.zip}: binary={result.binary_path} "
            f"meta={result.metadata_path} release={result.release} gpu={result.gpu}"
        )


if __name__ == "__main__":
    main()

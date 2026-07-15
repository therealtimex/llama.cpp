#!/usr/bin/env python3
"""Write RealTimeX-compatible _nlcBuildMetadata.json next to llama-server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_packaging import write_nlc_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Output path for _nlcBuildMetadata.json")
    p.add_argument("--platform", required=True, help="mac | linux | win (or darwin/win32)")
    p.add_argument("--arch", required=True, help="arm64 | x64 | armv7l")
    p.add_argument(
        "--gpu",
        required=True,
        help='metal | cuda | vulkan | false/cpu (use "false" for CPU)',
    )
    p.add_argument("--release", required=True, help="llama.cpp release tag, e.g. b10012")
    p.add_argument("--repo", default="ggml-org/llama.cpp")
    p.add_argument("--source", default="official-mirrored")
    p.add_argument("--upstream-asset", default="")
    p.add_argument("--upstream-digest", default="")
    p.add_argument("--platform-info-name", default="")
    p.add_argument("--platform-info-version", default="ci")
    p.add_argument("--print", action="store_true", help="Print JSON to stdout")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gpu = args.gpu
    if isinstance(gpu, str) and gpu.lower() in ("false", "cpu", "0", "off", "none"):
        gpu = False

    metadata = write_nlc_metadata(
        Path(args.out),
        platform=args.platform,
        arch=args.arch,
        gpu=gpu,
        release=args.release,
        repo=args.repo,
        source=args.source,
        upstream_asset=args.upstream_asset,
        upstream_digest=args.upstream_digest,
        platform_info_name=args.platform_info_name,
        platform_info_version=args.platform_info_version,
    )
    print(f"wrote {args.out}", flush=True)
    if args.print:
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

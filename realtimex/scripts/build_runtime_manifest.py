#!/usr/bin/env python3
"""Build runtime-manifest.json from a directory of RealTimeX runtime zips."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_packaging import (  # noqa: E402
    get_metadata_gpu,
    sha256_file,
    validate_release_tag,
    validate_runtime_zip,
    write_runtime_manifest,
)

ASSET_RE = re.compile(
    r"^llama-server-(?P<platform>darwin|linux|win32)-(?P<arch>arm64|x64|armv7l)"
    r"(?:-(?P<backend>cuda|vulkan))?-(?P<release>b\d+)\.zip$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dist-dir", required=True)
    p.add_argument("--tag", required=True, help="Release tag for RealTimeX, e.g. realtimex-b10012")
    p.add_argument("--llama-cpp-release", required=True, help="Upstream bNNNN")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    llama_release = validate_release_tag(args.llama_cpp_release)
    dist = Path(args.dist_dir)
    assets = []
    for zip_path in sorted(dist.glob("llama-server-*.zip")):
        m = ASSET_RE.match(zip_path.name)
        if not m:
            print(f"skip unexpected name: {zip_path.name}", flush=True)
            continue
        result = validate_runtime_zip(zip_path, expect_release=llama_release)
        backend = m.group("backend") or result.gpu
        if backend == "cpu":
            backend = ""
        # Prefer metal for darwin default packages
        if not m.group("backend") and result.gpu == "metal":
            backend = "metal"
        assets.append(
            {
                "name": zip_path.name,
                "platform": m.group("platform"),
                "arch": m.group("arch"),
                "backend": backend,
                "source": (result.metadata.get("provenance") or {}).get("source", ""),
                "sha256": sha256_file(zip_path),
                "size": zip_path.stat().st_size,
                "gpu": result.gpu,
                "release": result.release,
            }
        )

    write_runtime_manifest(
        Path(args.out),
        tag=args.tag,
        llama_cpp_release=llama_release,
        assets=assets,
    )
    print(f"wrote {args.out} with {len(assets)} assets", flush=True)
    print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()

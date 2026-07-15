#!/usr/bin/env python3
"""Package a cmake build tree's llama-server into a RealTimeX runtime zip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_packaging import (  # noqa: E402
    copy_runtime_payload,
    find_server_binary,
    format_template,
    matrix_entry_by_id,
    validate_release_tag,
    validate_runtime_zip,
    write_nlc_metadata,
    zip_runtime_dir,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True, help="llama.cpp release tag, e.g. b10012")
    p.add_argument(
        "--matrix-id",
        default="",
        help="Optional SELF_BUILD_MATRIX id, e.g. linux-x64-cuda",
    )
    p.add_argument("--build-dir", required=True, help="cmake build directory (contains bin/)")
    p.add_argument("--out-dir", default="realtimex-dist")
    p.add_argument("--platform", default="", help="Override meta platform")
    p.add_argument("--arch", default="", help="Override meta arch")
    p.add_argument("--gpu", default="", help="Override meta gpu")
    p.add_argument("--asset-name", default="", help="Override output zip name")
    p.add_argument("--source", default="realtimex-built")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = validate_release_tag(args.tag)
    row = matrix_entry_by_id(args.matrix_id) if args.matrix_id else None

    platform = args.platform or (row["meta_platform"] if row else "")
    arch = args.arch or (row["meta_arch"] if row else "")
    gpu = args.gpu or (row["meta_gpu"] if row else "")
    if not platform or not arch or gpu == "":
        raise SystemExit("platform, arch, and gpu are required (via --matrix-id or flags)")

    if gpu in ("false", "cpu", "0", "off", "none"):
        gpu_value: object = False
    else:
        gpu_value = gpu

    build_dir = Path(args.build_dir).resolve()
    server = find_server_binary(build_dir)
    if server is None:
        raise SystemExit(f"llama-server not found under {build_dir}")

    work_runtime = build_dir / "_realtimex_runtime_package"
    if work_runtime.exists():
        import shutil

        shutil.rmtree(work_runtime)
    copy_runtime_payload(server, work_runtime)
    write_nlc_metadata(
        work_runtime / "_nlcBuildMetadata.json",
        platform=platform,
        arch=arch,
        gpu=gpu_value,
        release=tag,
        source=args.source,
        upstream_asset="",
        upstream_digest="",
    )

    if args.asset_name:
        asset_name = args.asset_name
    elif row:
        asset_name = format_template(row["asset"], tag)
    else:
        backend = "" if gpu_value is False else str(gpu_value)
        asset_platform = {"mac": "darwin", "win": "win32"}.get(platform, platform)
        if backend in ("metal", "cpu"):
            backend = ""
        from runtime_packaging import build_asset_name

        asset_name = build_asset_name(asset_platform, arch, tag, backend)

    out_zip = Path(args.out_dir).resolve() / asset_name
    zip_runtime_dir(work_runtime, out_zip)
    validate_runtime_zip(
        out_zip,
        expect_release=tag,
        expect_gpu="cpu" if gpu_value is False else str(gpu_value),
        expect_platform={"mac": "darwin", "win": "win32"}.get(platform, platform),
        expect_arch=arch,
    )
    print(f"wrote {out_zip}", flush=True)
    print(f"ASSET_NAME={asset_name}")
    print(f"ASSET_PATH={out_zip}")


if __name__ == "__main__":
    main()

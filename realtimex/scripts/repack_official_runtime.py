#!/usr/bin/env python3
"""Download an official llama.cpp release pack and repack as RealTimeX runtime zip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_packaging import (  # noqa: E402
    copy_runtime_payload,
    download_file,
    extract_archive,
    find_server_binary,
    format_template,
    github_release_asset_url,
    matrix_entry_by_id,
    sha256_file,
    validate_release_tag,
    validate_runtime_zip,
    write_nlc_metadata,
    zip_runtime_dir,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True, help="Upstream llama.cpp tag, e.g. b10012")
    p.add_argument(
        "--matrix-id",
        required=True,
        help="Row id from OFFICIAL_MATRIX, e.g. darwin-arm64, win32-x64-cuda",
    )
    p.add_argument(
        "--upstream-repo",
        default="ggml-org/llama.cpp",
        help="GitHub repo for official assets",
    )
    p.add_argument("--work-dir", default="realtimex-work", help="Working directory")
    p.add_argument("--out-dir", default="realtimex-dist", help="Output directory for zips")
    p.add_argument(
        "--upstream-file",
        default="",
        help="Use a local upstream archive instead of downloading",
    )
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = validate_release_tag(args.tag)
    row = matrix_entry_by_id(args.matrix_id)
    if row.get("source") != "official-mirrored" and "upstream" not in row:
        # allow using official-shaped rows only
        pass
    if "upstream" not in row:
        raise SystemExit(f"matrix id {args.matrix_id!r} is not an official-mirrored row")

    upstream_name = format_template(row["upstream"], tag)
    asset_name = format_template(row["asset"], tag)
    work_dir = Path(args.work_dir).resolve() / args.matrix_id
    out_dir = Path(args.out_dir).resolve()
    extract_dir = work_dir / "extract"
    runtime_dir = work_dir / "runtime"
    upstream_path = Path(args.upstream_file).resolve() if args.upstream_file else work_dir / upstream_name

    work_dir.mkdir(parents=True, exist_ok=True)
    if runtime_dir.exists():
        import shutil

        shutil.rmtree(runtime_dir)
    if extract_dir.exists():
        import shutil

        shutil.rmtree(extract_dir)

    if not args.skip_download and not args.upstream_file:
        url = github_release_asset_url(args.upstream_repo, tag, upstream_name)
        print(f"downloading {url}", flush=True)
        download_file(url, upstream_path)
    elif not upstream_path.is_file():
        raise SystemExit(f"upstream archive not found: {upstream_path}")

    digest = f"sha256:{sha256_file(upstream_path)}"
    print(f"upstream digest {digest}", flush=True)

    extract_archive(upstream_path, extract_dir)
    server = find_server_binary(extract_dir)
    if server is None:
        raise SystemExit(f"llama-server not found inside {upstream_path.name}")
    print(f"found server binary {server}", flush=True)

    copy_runtime_payload(server, runtime_dir)
    gpu = row["meta_gpu"]
    if gpu == "false":
        gpu_value: object = False
    else:
        gpu_value = gpu

    write_nlc_metadata(
        runtime_dir / "_nlcBuildMetadata.json",
        platform=row["meta_platform"],
        arch=row["meta_arch"],
        gpu=gpu_value,
        release=tag,
        source=row.get("source", "official-mirrored"),
        upstream_asset=upstream_name,
        upstream_digest=digest,
    )

    out_zip = out_dir / asset_name
    zip_runtime_dir(runtime_dir, out_zip)
    validate_runtime_zip(
        out_zip,
        expect_release=tag,
        expect_gpu=str(gpu_value) if gpu_value is not False else "cpu",
        expect_platform=row["asset_platform"],
        expect_arch=row["asset_arch"],
    )
    print(f"wrote {out_zip}", flush=True)
    print(f"ASSET_NAME={asset_name}")
    print(f"ASSET_PATH={out_zip}")


if __name__ == "__main__":
    main()

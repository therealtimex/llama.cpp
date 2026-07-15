#!/usr/bin/env python3
"""Build runtime-manifest.json from a directory of RealTimeX runtime zips."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_packaging import (  # noqa: E402
    OFFICIAL_MATRIX,
    format_template,
    get_metadata_gpu,
    matrix_entry_by_id,
    sha256_file,
    validate_release_tag,
    validate_runtime_zip,
    write_runtime_manifest,
)

ASSET_RE = re.compile(
    r"^llama-server-(?P<platform>darwin|linux|win32)-(?P<arch>arm64|x64|armv7l)"
    r"(?:-(?P<backend>cuda|vulkan))?-(?P<release>b\d+)\.zip$"
)


def expected_asset_names(
    llama_release: str,
    *,
    require_official: bool,
    require_matrix_ids: list[str],
) -> list[str]:
    required_rows = list(OFFICIAL_MATRIX) if require_official else []
    required_ids = {row["id"] for row in required_rows}
    for matrix_id in require_matrix_ids:
        if matrix_id in required_ids:
            continue
        row = matrix_entry_by_id(matrix_id)
        required_rows.append(row)
        required_ids.add(matrix_id)
    return sorted(
        {format_template(row["asset"], llama_release) for row in required_rows}
    )


def missing_required_asset_names(
    assets: list[dict[str, object]], expected_names: list[str]
) -> list[str]:
    actual_names = {str(asset["name"]) for asset in assets}
    return [name for name in expected_names if name not in actual_names]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dist-dir", required=True)
    p.add_argument("--tag", required=True, help="Release tag for RealTimeX, e.g. realtimex-b10012")
    p.add_argument("--llama-cpp-release", required=True, help="Upstream bNNNN")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--require-official",
        action="store_true",
        help="Include every official matrix asset in the completeness check",
    )
    p.add_argument(
        "--require-matrix-id",
        action="append",
        default=[],
        help="Additional matrix row required for completeness (repeatable)",
    )
    p.add_argument(
        "--allow-missing-matrix-id",
        action="append",
        default=[],
        help=(
            "Additional matrix row expected in the final release but allowed to be "
            "missing from this publication phase (repeatable)"
        ),
    )
    p.add_argument(
        "--github-output",
        action="store_true",
        help="Write complete/expectedCount/missingCount to GITHUB_OUTPUT",
    )
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

    expected_names = expected_asset_names(
        llama_release,
        require_official=args.require_official,
        require_matrix_ids=args.require_matrix_id + args.allow_missing_matrix_id,
    )
    missing_names = missing_required_asset_names(assets, expected_names)
    complete = not missing_names
    allowed_missing_names = set(
        expected_asset_names(
            llama_release,
            require_official=False,
            require_matrix_ids=args.allow_missing_matrix_id,
        )
    )
    blocking_missing_names = [
        name for name in missing_names if name not in allowed_missing_names
    ]
    ready = not blocking_missing_names

    write_runtime_manifest(
        Path(args.out),
        tag=args.tag,
        llama_cpp_release=llama_release,
        assets=assets,
        complete=complete,
        expected_asset_names=expected_names,
        missing_asset_names=missing_names,
    )

    print(f"wrote {args.out} with {len(assets)} assets", flush=True)
    print(json.dumps(assets, indent=2))
    print(
        f"completeness: complete={str(complete).lower()} "
        f"ready={str(ready).lower()} expected={len(expected_names)} "
        f"missing={len(missing_names)} blocking={len(blocking_missing_names)}",
        flush=True,
    )
    for name in missing_names:
        print(f"missing required asset: {name}", flush=True)

    if args.github_output:
        output_path = os.environ.get("GITHUB_OUTPUT")
        if not output_path:
            raise SystemExit("GITHUB_OUTPUT is required with --github-output")
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(f"complete={'true' if complete else 'false'}\n")
            fh.write(f"ready={'true' if ready else 'false'}\n")
            fh.write(f"expectedCount={len(expected_names)}\n")
            fh.write(f"missingCount={len(missing_names)}\n")
            fh.write(f"blockingMissingCount={len(blocking_missing_names)}\n")


if __name__ == "__main__":
    main()

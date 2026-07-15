#!/usr/bin/env python3
"""Shared helpers for RealTimeX llama-server runtime packaging.

Asset naming and _nlcBuildMetadata.json shape must stay compatible with:
  - RealTimeX server/utils/AiProviders/nodeLlamaCpp/runtimeStatus.js
  - RealTimeX scripts/startup-go/startup/llama_server.go
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


BACKEND_IDS = ("metal", "cuda", "vulkan", "cpu")
RELEASE_RE = re.compile(r"^b\d+$")

# Official ggml-org/llama.cpp release asset → RealTimeX product asset.
OFFICIAL_MATRIX: list[dict[str, str]] = [
    {
        "id": "darwin-arm64",
        "upstream": "llama-{tag}-bin-macos-arm64.tar.gz",
        "asset": "llama-server-darwin-arm64-{tag}.zip",
        "meta_platform": "mac",
        "meta_arch": "arm64",
        "meta_gpu": "metal",
        "asset_platform": "darwin",
        "asset_arch": "arm64",
        "asset_backend": "",
        "source": "official-mirrored",
    },
    {
        "id": "darwin-x64",
        "upstream": "llama-{tag}-bin-macos-x64.tar.gz",
        "asset": "llama-server-darwin-x64-{tag}.zip",
        "meta_platform": "mac",
        "meta_arch": "x64",
        "meta_gpu": "metal",
        "asset_platform": "darwin",
        "asset_arch": "x64",
        "asset_backend": "",
        "source": "official-mirrored",
    },
    {
        "id": "linux-x64",
        "upstream": "llama-{tag}-bin-ubuntu-x64.tar.gz",
        "asset": "llama-server-linux-x64-{tag}.zip",
        "meta_platform": "linux",
        "meta_arch": "x64",
        "meta_gpu": "false",
        "asset_platform": "linux",
        "asset_arch": "x64",
        "asset_backend": "",
        "source": "official-mirrored",
    },
    {
        "id": "linux-arm64",
        "upstream": "llama-{tag}-bin-ubuntu-arm64.tar.gz",
        "asset": "llama-server-linux-arm64-{tag}.zip",
        "meta_platform": "linux",
        "meta_arch": "arm64",
        "meta_gpu": "false",
        "asset_platform": "linux",
        "asset_arch": "arm64",
        "asset_backend": "",
        "source": "official-mirrored",
    },
    {
        "id": "linux-x64-vulkan",
        "upstream": "llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz",
        "asset": "llama-server-linux-x64-vulkan-{tag}.zip",
        "meta_platform": "linux",
        "meta_arch": "x64",
        "meta_gpu": "vulkan",
        "asset_platform": "linux",
        "asset_arch": "x64",
        "asset_backend": "vulkan",
        "source": "official-mirrored",
    },
    {
        "id": "linux-arm64-vulkan",
        "upstream": "llama-{tag}-bin-ubuntu-vulkan-arm64.tar.gz",
        "asset": "llama-server-linux-arm64-vulkan-{tag}.zip",
        "meta_platform": "linux",
        "meta_arch": "arm64",
        "meta_gpu": "vulkan",
        "asset_platform": "linux",
        "asset_arch": "arm64",
        "asset_backend": "vulkan",
        "source": "official-mirrored",
    },
    {
        "id": "win32-x64",
        "upstream": "llama-{tag}-bin-win-cpu-x64.zip",
        "asset": "llama-server-win32-x64-{tag}.zip",
        "meta_platform": "win",
        "meta_arch": "x64",
        "meta_gpu": "false",
        "asset_platform": "win32",
        "asset_arch": "x64",
        "asset_backend": "",
        "source": "official-mirrored",
    },
    {
        "id": "win32-arm64",
        "upstream": "llama-{tag}-bin-win-cpu-arm64.zip",
        "asset": "llama-server-win32-arm64-{tag}.zip",
        "meta_platform": "win",
        "meta_arch": "arm64",
        "meta_gpu": "false",
        "asset_platform": "win32",
        "asset_arch": "arm64",
        "asset_backend": "",
        "source": "official-mirrored",
    },
    {
        "id": "win32-x64-cuda",
        "upstream": "llama-{tag}-bin-win-cuda-12.4-x64.zip",
        "asset": "llama-server-win32-x64-cuda-{tag}.zip",
        "meta_platform": "win",
        "meta_arch": "x64",
        "meta_gpu": "cuda",
        "asset_platform": "win32",
        "asset_arch": "x64",
        "asset_backend": "cuda",
        "source": "official-mirrored",
    },
    {
        "id": "win32-x64-vulkan",
        "upstream": "llama-{tag}-bin-win-vulkan-x64.zip",
        "asset": "llama-server-win32-x64-vulkan-{tag}.zip",
        "meta_platform": "win",
        "meta_arch": "x64",
        "meta_gpu": "vulkan",
        "asset_platform": "win32",
        "asset_arch": "x64",
        "asset_backend": "vulkan",
        "source": "official-mirrored",
    },
]

SELF_BUILD_MATRIX: list[dict[str, str]] = [
    {
        "id": "linux-x64-cuda",
        "asset": "llama-server-linux-x64-cuda-{tag}.zip",
        "meta_platform": "linux",
        "meta_arch": "x64",
        "meta_gpu": "cuda",
        "asset_platform": "linux",
        "asset_arch": "x64",
        "asset_backend": "cuda",
        "source": "realtimex-built",
    },
    {
        "id": "linux-arm64-cuda",
        "asset": "llama-server-linux-arm64-cuda-{tag}.zip",
        "meta_platform": "linux",
        "meta_arch": "arm64",
        "meta_gpu": "cuda",
        "asset_platform": "linux",
        "asset_arch": "arm64",
        "asset_backend": "cuda",
        "source": "realtimex-built",
    },
]


def die(message: str, code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def validate_release_tag(tag: str) -> str:
    tag = (tag or "").strip()
    if not RELEASE_RE.match(tag):
        die(f"invalid llama.cpp release tag {tag!r}; expected b<digits> e.g. b10012")
    return tag


def normalize_gpu_for_meta(gpu: Any) -> Any:
    """Return value suitable for buildOptions.gpu in _nlcBuildMetadata.json."""
    if gpu is False:
        return False
    if gpu is None:
        return False
    text = str(gpu).strip().lower()
    if text in ("", "false", "0", "off", "none", "cpu", "disable", "disabled"):
        return False
    if text in BACKEND_IDS:
        return text
    die(f"unsupported gpu setting {gpu!r}")


def normalize_gpu_for_compare(gpu: Any) -> str:
    value = normalize_gpu_for_meta(gpu)
    if value is False:
        return "cpu"
    return str(value)


def build_asset_name(
    platform: str,
    arch: str,
    release: str,
    backend: str = "",
) -> str:
    """Match Go buildLlamaServerRuntimeAssetName."""
    backend = (backend or "").strip().lower()
    if backend in ("", "cpu", "metal", "false"):
        return f"llama-server-{platform}-{arch}-{release}.zip"
    if backend in ("cuda", "vulkan"):
        return f"llama-server-{platform}-{arch}-{backend}-{release}.zip"
    die(f"backend {backend!r} cannot appear in asset name (only cuda/vulkan)")


def format_template(template: str, tag: str) -> str:
    return template.replace("{tag}", tag)


def matrix_entry_by_id(entry_id: str) -> dict[str, str]:
    for row in OFFICIAL_MATRIX + SELF_BUILD_MATRIX:
        if row["id"] == entry_id:
            return dict(row)
    die(f"unknown matrix id {entry_id!r}")


def write_nlc_metadata(
    out_path: Path,
    *,
    platform: str,
    arch: str,
    gpu: Any,
    release: str,
    repo: str = "ggml-org/llama.cpp",
    source: str = "official-mirrored",
    upstream_asset: str = "",
    upstream_digest: str = "",
    platform_info_name: str = "",
    platform_info_version: str = "ci",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    release = validate_release_tag(release)
    gpu_value = normalize_gpu_for_meta(gpu)

    platform = platform.strip().lower()
    if platform in ("darwin", "macos", "osx"):
        platform = "mac"
    elif platform in ("windows", "win32"):
        platform = "win"
    elif platform not in ("mac", "linux", "win"):
        die(f"unsupported meta platform {platform!r}")

    arch = arch.strip().lower()
    if arch == "amd64":
        arch = "x64"
    elif arch == "aarch64":
        arch = "arm64"
    elif arch == "arm":
        arch = "armv7l"

    if not platform_info_name:
        platform_info_name = {
            "mac": "macOS",
            "linux": "Linux",
            "win": "Windows",
        }[platform]

    metadata: dict[str, Any] = {
        "buildOptions": {
            "platform": platform,
            "platformInfo": {
                "name": platform_info_name,
                "version": platform_info_version,
            },
            "arch": arch,
            "gpu": gpu_value,
            "llamaCpp": {
                "repo": repo,
                "release": release,
            },
            "customCmakeOptions": {
                "LLAMA_BUILD_SERVER": "ON",
                "LLAMA_CURL": "OFF",
            },
        },
        "provenance": {
            "source": source,
            "upstreamRepo": repo,
            "upstreamTag": release,
            "upstreamAsset": upstream_asset,
            "upstreamDigest": upstream_digest,
            "builtBy": "therealtimex",
            "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    if extra:
        metadata = deep_merge(metadata, extra)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return metadata


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_metadata_release(metadata: dict[str, Any]) -> Optional[str]:
    build = metadata.get("buildOptions") or {}
    llama = build.get("llamaCpp") or {}
    if llama.get("release"):
        return str(llama["release"])
    top = metadata.get("llamaCpp") or {}
    if top.get("release"):
        return str(top["release"])
    return None


def get_metadata_gpu(metadata: dict[str, Any]) -> str:
    build = metadata.get("buildOptions") or {}
    return normalize_gpu_for_compare(build.get("gpu"))


def normalize_metadata_platform(platform: str) -> str:
    normalized = (platform or "").strip().lower()
    if normalized == "mac":
        return "darwin"
    if normalized == "win":
        return "win32"
    return normalized


SERVER_NAMES = ("llama-server", "llama-server.exe")
LIB_GLOBS = (
    "libllama*",
    "libggml*",
    "libmtmd*",
    "ggml*",
    "*.dll",
    "*.dylib",
    "*.so",
    "*.so.*",
    "*.metal",
)


def find_server_binary(root: Path) -> Optional[Path]:
    candidates: list[Path] = []
    for name in SERVER_NAMES:
        for path in root.rglob(name):
            if path.is_file() and not path.is_symlink():
                candidates.append(path)
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, str]:
        parts = [p.lower() for p in path.parts]
        # Prefer non-test, release-ish locations.
        penalty = 0
        if "test" in parts:
            penalty += 10
        if "debug" in parts:
            penalty += 5
        # Prefer shallower paths and bin/Release folders.
        bonus = 0
        if path.parent.name.lower() in ("bin", "release", "runtime"):
            bonus -= 2
        return (penalty + bonus, len(path.parts), str(path))

    candidates.sort(key=score)
    return candidates[0]


def is_probably_runtime_lib(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    if lower in SERVER_NAMES:
        return False
    if lower.endswith((".dll", ".dylib", ".so", ".metal")):
        return True
    if ".so." in lower:
        return True
    if lower.startswith(("libllama", "libggml", "libmtmd", "ggml", "llama", "mtmd")):
        return True
    return False


def collect_runtime_files(server_path: Path) -> list[Path]:
    """Collect llama-server + sibling shared libraries from its directory tree."""
    files: list[Path] = [server_path]
    search_roots = [server_path.parent]
    # Official packs often put libs next to the binary or one level up.
    if server_path.parent.parent != server_path.parent:
        search_roots.append(server_path.parent.parent)

    seen: set[Path] = {server_path.resolve()}
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            if is_probably_runtime_lib(path):
                # Skip huge unrelated tools; only libs near the server binary tree.
                try:
                    path.relative_to(server_path.parent.parent if server_path.parent.parent.exists() else server_path.parent)
                except ValueError:
                    continue
                seen.add(resolved)
                files.append(path)
    return files


def copy_runtime_payload(server_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = collect_runtime_files(server_path)
    # Flatten into dest_dir while preserving only basenames; collide-safe with prefix.
    used_names: set[str] = set()
    for src in files:
        name = src.name
        if name in used_names:
            # Prefer the server-dir copy of a name if already present.
            continue
        used_names.add(name)
        target = dest_dir / name
        shutil.copy2(src, target)
        if name in SERVER_NAMES:
            mode = target.stat().st_mode
            target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    server_dest = dest_dir / server_path.name
    if not server_dest.exists():
        die(f"failed to copy server binary to {server_dest}")
    return server_dest


def zip_runtime_dir(runtime_dir: Path, zip_path: Path, archive_root: str = "runtime") -> None:
    """Zip contents of runtime_dir under archive_root/, no symlinks/absolute paths."""
    runtime_dir = runtime_dir.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(runtime_dir.rglob("*")):
            if path.is_dir():
                continue
            if path.is_symlink():
                die(f"refusing to package symlink: {path}")
            rel = path.relative_to(runtime_dir).as_posix()
            if rel.startswith("/") or ".." in rel.split("/"):
                die(f"unsafe path while packaging: {rel}")
            arcname = f"{archive_root}/{rel}" if archive_root else rel
            # Preserve executable bit for Unix extractors via external_attr.
            mode = path.stat().st_mode
            info = zipfile.ZipInfo(arcname)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | (mode & 0o777)) << 16
            with path.open("rb") as fh:
                zf.writestr(info, fh.read())


def assert_safe_zip_entries(names: Iterable[str]) -> None:
    for name in names:
        entry = name.replace("\\", "/")
        if not entry or entry.endswith("/"):
            continue
        if "\0" in entry:
            die(f"zip entry has null byte: {entry!r}")
        if entry.startswith("/") or re.match(r"^[A-Za-z]:/", entry):
            die(f"zip entry is absolute: {entry!r}")
        if any(part == ".." for part in entry.split("/")):
            die(f"zip entry has path traversal: {entry!r}")


@dataclass
class ZipValidationResult:
    ok: bool
    metadata_path: str
    binary_path: str
    metadata: dict[str, Any]
    release: str
    gpu: str
    message: str = ""


def validate_runtime_zip(
    zip_path: Path,
    *,
    expect_release: str = "",
    expect_gpu: str = "",
    expect_platform: str = "",
    expect_arch: str = "",
) -> ZipValidationResult:
    if not zip_path.is_file():
        die(f"zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert_safe_zip_entries(names)
        # Symlink detection: Unix symlinks have symlink mode bits.
        for info in zf.infolist():
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                die(f"zip contains symlink entry: {info.filename}")

        meta_entries = [
            n for n in names if n.replace("\\", "/").endswith("_nlcBuildMetadata.json") and not n.endswith("/")
        ]
        if not meta_entries:
            die("zip missing _nlcBuildMetadata.json")

        for meta_name in meta_entries:
            raw = zf.read(meta_name)
            try:
                metadata = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                die(f"invalid JSON in {meta_name}: {exc}")

            release = get_metadata_release(metadata) or ""
            gpu = get_metadata_gpu(metadata)
            build = metadata.get("buildOptions") or {}
            meta_platform = normalize_metadata_platform(str(build.get("platform") or ""))
            meta_arch = str(build.get("arch") or "")

            if expect_release and release != expect_release:
                continue
            if expect_gpu and gpu != normalize_gpu_for_compare(expect_gpu):
                # metal satisfies cpu on darwin — handled by caller if needed
                if not (
                    normalize_gpu_for_compare(expect_gpu) == "cpu"
                    and gpu == "metal"
                    and meta_platform == "darwin"
                ):
                    continue
            if expect_platform:
                want = expect_platform if expect_platform not in ("mac", "win") else normalize_metadata_platform(expect_platform)
                if meta_platform and meta_platform != want and meta_platform != expect_platform:
                    # allow meta mac vs expect darwin
                    if not (
                        (meta_platform == "darwin" and expect_platform in ("darwin", "mac"))
                        or (meta_platform == "win32" and expect_platform in ("win32", "win"))
                    ):
                        continue
            if expect_arch and meta_arch and meta_arch != expect_arch:
                if not (meta_arch == "arm" and expect_arch == "armv7l"):
                    continue

            meta_dir = meta_name.replace("\\", "/").rsplit("/", 1)[0] if "/" in meta_name.replace("\\", "/") else ""
            candidate_dirs = [meta_dir]
            if meta_dir:
                candidate_dirs.append(f"{meta_dir}/Release")
            else:
                candidate_dirs.append("Release")

            file_set = {n.replace("\\", "/") for n in names if not n.endswith("/")}
            for candidate_dir in candidate_dirs:
                for binary_name in SERVER_NAMES:
                    binary_path = f"{candidate_dir}/{binary_name}" if candidate_dir else binary_name
                    binary_path = binary_path.replace("//", "/")
                    if binary_path in file_set:
                        if expect_release and release != expect_release:
                            die(
                                f"metadata release {release!r} does not match expected {expect_release!r}"
                            )
                        return ZipValidationResult(
                            ok=True,
                            metadata_path=meta_name,
                            binary_path=binary_path,
                            metadata=metadata,
                            release=release,
                            gpu=gpu,
                        )

        die(
            "zip has metadata but no co-located llama-server binary "
            f"(expect_release={expect_release!r}, expect_gpu={expect_gpu!r})"
        )


def run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        run(["unzip", "-q", "-o", str(archive_path), "-d", str(dest_dir)])
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        run(["tar", "-xzf", str(archive_path), "-C", str(dest_dir)])
    elif name.endswith(".tar.xz"):
        run(["tar", "-xJf", str(archive_path), "-C", str(dest_dir)])
    else:
        die(f"unsupported archive type: {archive_path.name}")


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["curl", "-fsSL", "-o", str(dest), url])


def github_release_asset_url(repo: str, tag: str, asset_name: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def write_runtime_manifest(
    out_path: Path,
    *,
    tag: str,
    llama_cpp_release: str,
    assets: list[dict[str, Any]],
) -> None:
    payload = {
        "tag": tag,
        "llamaCppRelease": llama_cpp_release,
        "upstreamRepo": "ggml-org/llama.cpp",
        "publishedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assets": assets,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

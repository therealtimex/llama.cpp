# RealTimeX llama-server runtime packaging

Hybrid pipeline that turns a pinned [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) release into RealTimeX-managed `llama-server` zips with `_nlcBuildMetadata.json`.

## Why

RealTimeX no longer needs Node native addons for local LLM. It still expects:

| Contract | Example |
|----------|---------|
| Asset name | `llama-server-darwin-arm64-b10012.zip` |
| Zip layout | `runtime/llama-server` + `runtime/_nlcBuildMetadata.json` (+ libs) |
| Meta | `buildOptions.platform/arch/gpu` + `buildOptions.llamaCpp.release` |

Official llama.cpp release tarballs do **not** include that metadata. This tree repacks (or self-builds) them.

## GitHub Actions on this fork

This fork is a **RealTimeX packaging surface**, not upstream CI.

Workflows kept:

- `.github/workflows/realtimex-promote-runtime.yml` - build/repack/publish runtimes
- `.github/workflows/realtimex-watch-upstream.yml` - watch `ggml-org/llama.cpp` for new `b*` tags

All other upstream Actions (build matrices, release, lint bots, UI, Docker, etc.) are removed so pushes/syncs do not burn CI. Re-evaluate after merging from `ggml-org/llama.cpp` if workflows are restored.


## Workflows

### Promote runtime

[`.github/workflows/realtimex-promote-runtime.yml`](../.github/workflows/realtimex-promote-runtime.yml)

```bash
# Dry run (artifacts only)
gh workflow run realtimex-promote-runtime.yml \
  --repo therealtimex/llama.cpp \
  -f llama_cpp_tag=b10012 \
  -f publish=false

# Publish release realtimex-b10012
gh workflow run realtimex-promote-runtime.yml \
  --repo therealtimex/llama.cpp \
  -f llama_cpp_tag=b10012 \
  -f publish=true \
  -f build_linux_cuda_x64=true \
  -f build_linux_cuda_arm64=true
```

Publishing requires all 12 runtime variants, including both Linux CUDA self-builds. The CUDA inputs may be disabled only for artifact-only dry runs. Published `realtimex-b*` releases are immutable; use a new tag rather than rerunning a published release.

### Watch upstream (from node-llama-cpp "Watch llama.cpp")

[`.github/workflows/realtimex-watch-upstream.yml`](../.github/workflows/realtimex-watch-upstream.yml)

Polls `ggml-org/llama.cpp` for newer `b*` tags than the latest `realtimex-b*` release.

| Mode | Behavior |
|------|----------|
| Cron (every 3h) | Detect and report only; does not dispatch an expensive build |
| Manual | Can force tag, toggle promote/publish/cuda |

```bash
# Check + dry-run promote if behind
gh workflow run realtimex-watch-upstream.yml \
  --repo therealtimex/llama.cpp \
  -f dispatch_promote=true \
  -f publish=false

# Force a specific tag dry-run
gh workflow run realtimex-watch-upstream.yml \
  --repo therealtimex/llama.cpp \
  -f force_tag=b10012 \
  -f dispatch_promote=true \
  -f publish=false
```

Local compare:

```bash
python3 realtimex/scripts/watch_upstream.py --json
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/runtime_packaging.py` | Shared matrix, meta writer, zip helpers |
| `scripts/write_nlc_metadata.py` | CLI for `_nlcBuildMetadata.json` |
| `scripts/validate_runtime_zip.py` | Validate zip for app compatibility |
| `scripts/repack_official_runtime.py` | Download official pack -> RealTimeX zip |
| `scripts/package_built_runtime.py` | Package local cmake `llama-server` build |
| `scripts/build_runtime_manifest.py` | Emit `runtime-manifest.json` and verify the required matrix |
| `scripts/list_matrix.py` | Dump packaging matrix |
| `scripts/watch_upstream.py` | Compare upstream `b*` vs `realtimex-b*` releases |

### Local repack example

```bash
python3 realtimex/scripts/repack_official_runtime.py \
  --tag b10012 \
  --matrix-id darwin-arm64 \
  --out-dir realtimex-dist
```

### Local package of a cmake build

```bash
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
cmake --build build --target llama-server
python3 realtimex/scripts/package_built_runtime.py \
  --tag b10012 \
  --matrix-id linux-x64-cuda \
  --build-dir build \
  --out-dir realtimex-dist
```

## Asset matrix (summary)

- **Official -> repack:** macOS arm64/x64, Linux CPU/Vulkan, Windows CPU/CUDA/Vulkan
- **Self-build:** Linux x64 CUDA and Linux arm64 CUDA (default on)

GPU naming rules (must match RealTimeX Go/JS):

- `cuda` / `vulkan` appear in the filename
- `metal` / `cpu` do **not** appear in the filename (meta still has `"gpu": "metal"` or `false`)

## Linux CUDA builds

`build-linux-cuda-x64` uses a GitHub-hosted runner and optimizes for CI time:

| Setting | Value | Why |
|---------|-------|-----|
| CUDA arches | `75-real;80-real;86-real;89-real;90` | SASS for common deployed GPUs plus SM90 PTX for forward compatibility |
| ccache | on | Faster re-runs / same-tag rebuilds |
| Target | `llama-server` only | Avoid compiling unused tools |
| Toolkit cache | `use-github-cache: true` | Skip re-downloading CUDA |
| Transfer artifact retention | 1 day | Matrix artifacts are temporary inputs to assembly |
| Combined artifact retention | 7 days | Preserve dry-run output without long-term duplication |

The generic Linux x64 CUDA package covers Turing (`75`), Ampere (`80`/`86`), Ada (`89`), and Hopper (`90`). The Linux arm64 CUDA package uses CUDA 13 and includes native code for Jetson Orin (`87`) and DGX Spark (`121a`).

`build-linux-cuda-arm64` runs on the RealTimeX DGX Spark runner. The repository-scoped runner must be online with the default `self-hosted`, `Linux`, and `ARM64` labels plus `dgx-spark` and `llama-cpp`.

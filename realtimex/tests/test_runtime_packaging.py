#!/usr/bin/env python3

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_runtime_manifest import (  # noqa: E402
    expected_asset_names,
    missing_required_asset_names,
)
from runtime_packaging import (  # noqa: E402
    OFFICIAL_MATRIX,
    SELF_BUILD_MATRIX,
    collect_runtime_files,
    format_template,
    is_probably_runtime_lib,
    write_runtime_manifest,
)
from runtime_release_state import classify_runtime_release  # noqa: E402
import watch_upstream  # noqa: E402
from watch_upstream import (  # noqa: E402
    expected_runtime_asset_names,
    release_is_complete,
    runtime_release_state,
)


class RuntimePayloadTests(unittest.TestCase):
    def test_runtime_library_filter_rejects_build_files(self) -> None:
        accepted = (
            "ggml-metal.metal",
            "libggml.so",
            "libggml.so.0.16.0",
            "libllama.dylib",
            "cudart64_13.dll",
        )
        rejected = (
            "ggml-cuda.cu.o",
            "libllama-common.a",
            "ggml-config.cmake",
            "llama.pc",
        )

        for name in accepted:
            with self.subTest(name=name):
                self.assertTrue(is_probably_runtime_lib(Path(name)))
        for name in rejected:
            with self.subTest(name=name):
                self.assertFalse(is_probably_runtime_lib(Path(name)))

    def test_collect_runtime_files_keeps_only_runtime_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = Path(temp_dir) / "build" / "bin"
            bin_dir.mkdir(parents=True)
            for name in (
                "llama-server",
                "libggml.so.0.16.0",
                "libllama.so.0.0.1",
                "ggml-cuda.cu.o",
                "libllama-common.a",
                "llama-config.cmake",
            ):
                (bin_dir / name).write_bytes(b"fixture")

            files = collect_runtime_files(bin_dir / "llama-server")

        self.assertEqual(
            {path.name for path in files},
            {"llama-server", "libggml.so.0.16.0", "libllama.so.0.0.1"},
        )


class RuntimeManifestTests(unittest.TestCase):
    @staticmethod
    def managed_manifest(
        asset_names: list[str],
        *,
        complete: bool,
        expected_asset_names: list[str] | None = None,
    ) -> dict[str, object]:
        expected = expected_asset_names or asset_names
        missing = sorted(set(expected) - set(asset_names))
        return {
            "tag": "realtimex-b10017",
            "llamaCppRelease": "b10017",
            "complete": complete,
            "expectedAssetNames": expected,
            "missingAssetNames": missing,
            "assets": [{"name": name} for name in asset_names],
        }

    def test_manifest_cli_allows_only_late_x64_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dist = root / "dist"
            dist.mkdir()
            for row in OFFICIAL_MATRIX + SELF_BUILD_MATRIX:
                if row["id"] == "linux-x64-cuda":
                    continue
                name = format_template(row["asset"], "b10017")
                gpu = row["meta_gpu"]
                if gpu == "false":
                    gpu = False
                metadata = {
                    "buildOptions": {
                        "platform": row["meta_platform"],
                        "arch": row["meta_arch"],
                        "gpu": gpu,
                        "llamaCpp": {"release": "b10017"},
                    },
                    "provenance": {"source": row["source"]},
                }
                binary_name = (
                    "llama-server.exe"
                    if row["meta_platform"] == "win"
                    else "llama-server"
                )
                with zipfile.ZipFile(dist / name, "w") as runtime_zip:
                    runtime_zip.writestr(
                        "runtime/_nlcBuildMetadata.json",
                        json.dumps(metadata),
                    )
                    runtime_zip.writestr(f"runtime/{binary_name}", b"fixture")

            manifest_path = root / "runtime-manifest.json"
            github_output = root / "github-output"
            env = dict(os.environ, GITHUB_OUTPUT=str(github_output))
            subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parents[1]
                        / "scripts"
                        / "build_runtime_manifest.py"
                    ),
                    "--dist-dir",
                    str(dist),
                    "--tag",
                    "realtimex-b10017",
                    "--llama-cpp-release",
                    "b10017",
                    "--out",
                    str(manifest_path),
                    "--require-official",
                    "--require-matrix-id",
                    "linux-arm64-cuda",
                    "--allow-missing-matrix-id",
                    "linux-x64-cuda",
                    "--github-output",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            outputs = dict(
                line.split("=", 1)
                for line in github_output.read_text().splitlines()
            )
            manifest = json.loads(manifest_path.read_text())

        self.assertEqual(outputs["ready"], "true")
        self.assertEqual(outputs["complete"], "false")
        self.assertEqual(outputs["missingCount"], "1")
        self.assertEqual(outputs["blockingMissingCount"], "0")
        self.assertEqual(
            manifest["missingAssetNames"],
            ["llama-server-linux-x64-cuda-b10017.zip"],
        )

    def test_incremental_manifest_reports_late_x64_cuda_asset(self) -> None:
        expected = expected_asset_names(
            "b10017",
            require_official=True,
            require_matrix_ids=["linux-arm64-cuda", "linux-x64-cuda"],
        )
        present = [{"name": name} for name in expected if "linux-x64-cuda" not in name]
        missing = missing_required_asset_names(present, expected)

        self.assertEqual(
            missing,
            ["llama-server-linux-x64-cuda-b10017.zip"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "runtime-manifest.json"
            write_runtime_manifest(
                manifest_path,
                tag="realtimex-b10017",
                llama_cpp_release="b10017",
                assets=present,
                complete=False,
                expected_asset_names=expected,
                missing_asset_names=missing,
            )
            manifest = json.loads(manifest_path.read_text())

        self.assertEqual(manifest["status"], "assembling")
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["expectedAssetNames"], expected)
        self.assertEqual(manifest["missingAssetNames"], missing)

    def test_watcher_requires_full_assets_and_completed_manifest(self) -> None:
        expected = sorted(expected_runtime_asset_names("b10017"))
        assets = [
            {"name": name, "url": f"https://api.github.test/assets/{index}"}
            for index, name in enumerate(expected)
        ]
        assets.append(
            {
                "name": "runtime-manifest.json",
                "url": "https://api.github.test/assets/manifest",
            }
        )
        release = {"tag_name": "realtimex-b10017", "assets": assets}
        complete_manifest = self.managed_manifest(expected, complete=True)
        assembling_manifest = self.managed_manifest(expected[:-1], complete=False)
        legacy_manifest = {
            "tag": "realtimex-b10017",
            "llamaCppRelease": "b10017",
            "assets": [{"name": name} for name in expected],
        }

        with patch("watch_upstream.github_api", return_value=complete_manifest):
            self.assertTrue(release_is_complete(release, "token"))
        with patch("watch_upstream.github_api", return_value=assembling_manifest):
            self.assertFalse(release_is_complete(release, "token"))
        with patch("watch_upstream.github_api", return_value=legacy_manifest):
            self.assertTrue(release_is_complete(release, "token"))
        with patch(
            "watch_upstream.github_api",
            return_value={"name": "runtime-manifest.json"},
        ):
            self.assertFalse(release_is_complete(release, "token"))

        release["assets"] = [
            asset
            for asset in release["assets"]
            if asset["name"] != "llama-server-linux-x64-cuda-b10017.zip"
        ]
        with patch("watch_upstream.github_api", return_value=complete_manifest):
            self.assertFalse(release_is_complete(release, "token"))

    def test_release_state_uses_manifest_declared_future_matrix(self) -> None:
        expected = ["llama-server-future-b10017.zip"]
        manifest = self.managed_manifest(
            expected,
            complete=True,
            expected_asset_names=expected,
        )
        state = classify_runtime_release("b10017", set(expected), manifest)

        self.assertTrue(state.complete)
        self.assertEqual(state.expected_asset_names, expected)

    def test_release_state_cli_classifies_complete_manifest(self) -> None:
        expected = sorted(expected_runtime_asset_names("b10017"))
        manifest = self.managed_manifest(expected, complete=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets_path = root / "assets.txt"
            manifest_path = root / "runtime-manifest.json"
            assets_path.write_text("\n".join(expected + ["runtime-manifest.json"]))
            manifest_path.write_text(json.dumps(manifest))
            result = subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parents[1]
                        / "scripts"
                        / "runtime_release_state.py"
                    ),
                    "--tag",
                    "b10017",
                    "--assets-file",
                    str(assets_path),
                    "--manifest",
                    str(manifest_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["state"], "complete")
        self.assertTrue(payload["complete"])

    def test_watcher_treats_manifest_fetch_failure_as_unknown(self) -> None:
        release = {
            "tag_name": "realtimex-b10017",
            "assets": [
                {
                    "name": "runtime-manifest.json",
                    "url": "https://api.github.test/assets/manifest",
                }
            ],
        }
        with patch("watch_upstream.github_api", side_effect=SystemExit(1)):
            self.assertEqual(runtime_release_state(release, "token"), "unknown")

    def test_watcher_dispatches_x64_repair_for_assembling_release(self) -> None:
        release = {"tag_name": "realtimex-b10017", "assets": []}
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                ["watch_upstream.py", "--force-tag", "b10017", "--json"],
            ),
            patch("watch_upstream.latest_promoted_release", return_value=release),
            patch("watch_upstream.runtime_release_state", return_value="incomplete"),
            redirect_stdout(output),
        ):
            with self.assertRaises(SystemExit) as exit_context:
                watch_upstream.main()

        self.assertEqual(exit_context.exception.code, 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["updated"])
        self.assertTrue(payload["repairExisting"])
        self.assertFalse(payload["releaseComplete"])
        self.assertEqual(payload["releaseState"], "incomplete")

    def test_watcher_does_not_repair_unknown_release_state(self) -> None:
        release = {"tag_name": "realtimex-b10017", "assets": []}
        error_output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                ["watch_upstream.py", "--force-tag", "b10017", "--json"],
            ),
            patch("watch_upstream.latest_promoted_release", return_value=release),
            patch("watch_upstream.runtime_release_state", return_value="unknown"),
            redirect_stderr(error_output),
        ):
            with self.assertRaises(SystemExit) as exit_context:
                watch_upstream.main()

        self.assertEqual(exit_context.exception.code, 1)
        self.assertIn("could not determine release state", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()

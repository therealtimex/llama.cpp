#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from runtime_packaging import collect_runtime_files, is_probably_runtime_lib  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()

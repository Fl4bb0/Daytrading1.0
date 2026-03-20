from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kvant.utils.pipeline_config import load_pipeline_config, list_from_config


class PipelineConfigTests(unittest.TestCase):
    def test_load_merges_with_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[prepare]",
                        "lookback = 42",
                        "",
                        "[train]",
                        "device = \"cuda\"",
                    ]
                )
            )
            cfg, used_path = load_pipeline_config(cfg_path)

            self.assertEqual(used_path, cfg_path.resolve())
            self.assertEqual(int(cfg["prepare"]["lookback"]), 42)
            self.assertEqual(cfg["train"]["device"], "cuda")
            self.assertEqual(cfg["predict"]["split"], "test")

    def test_validation_rejects_invalid_split_fracs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[prepare]",
                        "val_frac = 0.6",
                        "test_frac = 0.5",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_list_from_config_handles_supported_types(self) -> None:
        self.assertEqual(list_from_config(None), None)
        self.assertEqual(list_from_config("AAPL"), ["AAPL"])
        self.assertEqual(list_from_config(["AAPL", "MSFT"]), ["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()


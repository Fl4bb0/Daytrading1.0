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
            self.assertEqual(cfg["ensemble"]["models"], [])
            self.assertFalse(bool(cfg["meta"]["enabled"]))
            self.assertEqual(float(cfg["trading"]["brokerage_fee"]), 0.0008)

    def test_validation_accepts_known_ensemble_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[ensemble]",
                        "models = [\"resnls\", \"conv1d\"]",
                    ]
                )
            )
            cfg, _ = load_pipeline_config(cfg_path)
            self.assertEqual(cfg["ensemble"]["models"], ["resnls", "conv1d"])

    def test_validation_rejects_duplicate_ensemble_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[ensemble]",
                        "models = [\"resnls\", \"resnls\"]",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_rejects_unknown_ensemble_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[ensemble]",
                        "models = [\"resnls\", \"unknown\"]",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

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

    def test_validation_rejects_invalid_predict_probability_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[predict]",
                        "required_buy_probability = 1.5",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_rejects_invalid_execution_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[predict]",
                        "execution_priority = \"highest_prob\"",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_accepts_meta_score_when_meta_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[predict]",
                        "execution_priority = \"meta_score\"",
                        "",
                        "[meta]",
                        "enabled = true",
                    ]
                )
            )
            cfg, _ = load_pipeline_config(cfg_path)
            self.assertEqual(cfg["predict"]["execution_priority"], "meta_score")
            self.assertTrue(bool(cfg["meta"]["enabled"]))

    def test_validation_rejects_meta_score_when_meta_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[predict]",
                        "execution_priority = \"meta_score\"",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_rejects_invalid_meta_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[meta]",
                        "train_split = \"test\"",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_rejects_invalid_meta_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[meta]",
                        "alpha = 0",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_rejects_negative_meta_shrinkage_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[meta]",
                        "shrinkage_k = -1",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_accepts_meta_side_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[meta]",
                        "min_score_buy = 0.01",
                        "min_score_short = 0.02",
                    ]
                )
            )
            cfg, _ = load_pipeline_config(cfg_path)
            self.assertEqual(float(cfg["meta"]["min_score_buy"]), 0.01)
            self.assertEqual(float(cfg["meta"]["min_score_short"]), 0.02)

    def test_validation_rejects_invalid_top_k_per_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[predict]",
                        "top_k_per_timestamp = -1",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_rejects_negative_ticker_cooldown_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[predict]",
                        "ticker_cooldown_minutes = -5",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_accepts_non_negative_brokerage_fee(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[trading]",
                        "brokerage_fee = 0.0012",
                    ]
                )
            )
            cfg, _ = load_pipeline_config(cfg_path)
            self.assertEqual(float(cfg["trading"]["brokerage_fee"]), 0.0012)

    def test_validation_rejects_negative_brokerage_fee(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[trading]",
                        "brokerage_fee = -0.001",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_rejects_conv3d_inside_ensemble_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[ensemble]",
                        "models = [\"resnls\", \"conv3d\"]",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_rejects_unknown_train_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[train]",
                        "model = \"unknown\"",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)

    def test_validation_rejects_unknown_predict_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pipeline.toml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "[predict]",
                        "model = \"unknown\"",
                    ]
                )
            )
            with self.assertRaises(SystemExit):
                load_pipeline_config(cfg_path)


if __name__ == "__main__":
    unittest.main()

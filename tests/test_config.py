from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from face_lora_eval.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_basic_yaml_without_requiring_pyyaml_features(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                """
person_id: xiaohan
reference_root: "/tmp/ref"
candidate_root: "/tmp/candidate"
output_dir: "/tmp/out"
insightface:
  root: "/tmp/models"
  model_name: "buffalo_l"
  det_size: 1024
  providers:
    - "CPUExecutionProvider"
reference_quality:
  min_det_score: 0.50
  min_face_area_ratio: 0.001
ranking:
  mean_weight: 0.45
  median_weight: 0.25
  p25_weight: 0.20
  fail_rate_weight: 0.10
""".strip(),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.person_id, "xiaohan")
        self.assertEqual(config.insightface["root"], Path("/tmp/models"))
        self.assertEqual(config.insightface["providers"], ["CPUExecutionProvider"])
        self.assertEqual(config.ranking["mean_weight"], 0.45)


if __name__ == "__main__":
    unittest.main()


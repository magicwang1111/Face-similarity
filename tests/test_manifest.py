from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from face_lora_eval.manifest import build_manifest


class ManifestTests(unittest.TestCase):
    def test_build_manifest_handles_chinese_and_spaces(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "0417-人脸采集 小韩"
            candidate = root / "批量测试输出"
            lora = candidate / "20260506xiaohan_flux2_lora16_000000300"
            scene = lora / "女装_灰色套装_秀场"
            reference.mkdir(parents=True)
            scene.mkdir(parents=True)

            (reference / "2026-04-17 124429.jpg").write_bytes(b"not-a-real-image")
            (scene / "女性_灰色上衣白裤_01_loop_0.png").write_bytes(b"not-a-real-image")

            records = build_manifest(reference, candidate, person_id="xiaohan")

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].role, "reference")
        self.assertEqual(records[1].role, "candidate")
        self.assertEqual(records[1].lora_id, "20260506xiaohan_flux2_lora16_000000300")
        self.assertEqual(records[1].scene, "女装_灰色套装_秀场")

    def test_manifest_limits_candidates_per_lora(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "ref"
            candidate = root / "cand"
            reference.mkdir()
            for lora_index in range(2):
                scene = candidate / f"lora_{lora_index}" / "scene"
                scene.mkdir(parents=True)
                for image_index in range(3):
                    (scene / f"{image_index}.png").write_bytes(b"x")
            (reference / "a.jpg").write_bytes(b"x")

            records = build_manifest(reference, candidate, "p", limit_candidates_per_lora=2)

        candidates = [record for record in records if record.role == "candidate"]
        self.assertEqual(len(candidates), 4)


if __name__ == "__main__":
    unittest.main()


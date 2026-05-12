from __future__ import annotations

from dataclasses import dataclass
import unittest

from face_lora_eval.face_model import select_best_face


@dataclass
class DummyFace:
    bbox: tuple[float, float, float, float]
    det_score: float


class FaceSelectionTests(unittest.TestCase):
    def test_selects_large_center_face_over_edge_face(self) -> None:
        edge_face = DummyFace(bbox=(0, 0, 100, 100), det_score=0.99)
        center_face = DummyFace(bbox=(350, 250, 650, 550), det_score=0.90)

        face, quality = select_best_face([edge_face, center_face], image_width=1000, image_height=800)

        self.assertIs(face, center_face)
        self.assertIsNotNone(quality)
        self.assertEqual(quality.num_faces, 2)


if __name__ == "__main__":
    unittest.main()


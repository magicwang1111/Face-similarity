from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from face_lora_eval.face_model import resolve_onnx_providers


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_PIL = importlib.util.find_spec("PIL") is not None


class LVFaceProviderTests(unittest.TestCase):
    def test_provider_resolution_prefers_available_cuda_then_cpu(self) -> None:
        providers = resolve_onnx_providers(
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            available=["CPUExecutionProvider"],
        )
        self.assertEqual(providers, ["CPUExecutionProvider"])


@unittest.skipUnless(HAS_NUMPY, "numpy is required for LVFace preprocessing tests")
class LVFacePreprocessTests(unittest.TestCase):
    def test_preprocess_shape_dtype_and_range(self) -> None:
        import numpy as np

        from face_lora_eval.face_model import preprocess_lvface_aligned_bgr

        image = np.zeros((112, 112, 3), dtype=np.uint8)
        batch = preprocess_lvface_aligned_bgr(image, input_size=112)

        self.assertEqual(batch.shape, (1, 3, 112, 112))
        self.assertEqual(batch.dtype, np.float32)
        self.assertAlmostEqual(float(batch.min()), -1.0, places=5)
        self.assertAlmostEqual(float(batch.max()), -1.0, places=5)

    def test_normalize_onnx_embedding_flattens_and_l2_normalizes(self) -> None:
        import numpy as np

        from face_lora_eval.face_model import normalize_onnx_embedding

        embedding = normalize_onnx_embedding([np.array([[3.0, 4.0]], dtype=np.float32)])

        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)
        self.assertAlmostEqual(float(embedding[0]), 0.6, places=5)
        self.assertAlmostEqual(float(embedding[1]), 0.8, places=5)


@unittest.skipUnless(HAS_NUMPY and HAS_PIL, "numpy and Pillow are required for LVFace extraction tests")
class LVFaceMockSessionTests(unittest.TestCase):
    def test_extract_uses_mock_session_output(self) -> None:
        import numpy as np
        from PIL import Image

        from face_lora_eval.face_model import LVFaceOnnxEmbedder

        @dataclass
        class DummyFace:
            bbox: tuple[float, float, float, float]
            det_score: float
            kps: object

        class DummyDetector:
            def get(self, image):
                return [
                    DummyFace(
                        bbox=(10.0, 10.0, 90.0, 90.0),
                        det_score=0.99,
                        kps=np.array(
                            [[30.0, 35.0], [70.0, 35.0], [50.0, 55.0], [35.0, 75.0], [65.0, 75.0]],
                            dtype=np.float32,
                        ),
                    )
                ]

        class DummyInput:
            name = "input"

        class DummySession:
            def get_inputs(self):
                return [DummyInput()]

            def run(self, output_names, feeds):
                self.last_batch = feeds["input"]
                return [np.array([[3.0, 4.0]], dtype=np.float32)]

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "face.png"
            Image.new("RGB", (100, 100), (128, 128, 128)).save(image_path)

            session = DummySession()
            embedder = LVFaceOnnxEmbedder(
                detector_root=Path(tmp),
                model_path=Path(tmp) / "missing.onnx",
            )
            embedder._detector_app = DummyDetector()
            embedder._session = session
            embedder._input_name = "input"
            embedder._align_face = lambda image, face: np.zeros((112, 112, 3), dtype=np.uint8)

            result = embedder.extract(image_path)

        self.assertIsNotNone(result)
        self.assertEqual(session.last_batch.shape, (1, 3, 112, 112))
        self.assertAlmostEqual(float(result.embedding[0]), 0.6, places=5)
        self.assertAlmostEqual(float(result.embedding[1]), 0.8, places=5)


if __name__ == "__main__":
    unittest.main()


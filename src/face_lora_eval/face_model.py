from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .image_io import load_bgr_array


@dataclass(frozen=True)
class FaceQuality:
    det_score: float
    face_area_ratio: float
    face_width: float
    face_height: float
    center_offset: float
    selection_score: float
    num_faces: int


@dataclass(frozen=True)
class FaceEmbedding:
    embedding: Any
    quality: FaceQuality


def l2_normalize(vector: Any):
    import numpy as np

    vector = np.asarray(vector, dtype="float32")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Embedding norm is zero.")
    return vector / norm


def select_best_face(faces: list[Any], image_width: int, image_height: int) -> tuple[Any | None, FaceQuality | None]:
    if not faces:
        return None, None

    image_area = max(float(image_width * image_height), 1.0)
    half_diag = max(((image_width / 2) ** 2 + (image_height / 2) ** 2) ** 0.5, 1.0)
    image_center_x = image_width / 2
    image_center_y = image_height / 2

    best_face = None
    best_quality = None
    best_score = -1.0

    for face in faces:
        bbox = [float(value) for value in face.bbox]
        x1, y1, x2, y2 = bbox
        face_width = max(x2 - x1, 0.0)
        face_height = max(y2 - y1, 0.0)
        area_ratio = (face_width * face_height) / image_area
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        center_offset = (((face_center_x - image_center_x) ** 2 + (face_center_y - image_center_y) ** 2) ** 0.5) / half_diag
        centered = max(0.0, 1.0 - center_offset)
        det_score = float(getattr(face, "det_score", 0.0))
        area_score = min(area_ratio / 0.08, 1.0)
        selection_score = 0.55 * det_score + 0.35 * area_score + 0.10 * centered
        quality = FaceQuality(
            det_score=det_score,
            face_area_ratio=area_ratio,
            face_width=face_width,
            face_height=face_height,
            center_offset=center_offset,
            selection_score=selection_score,
            num_faces=len(faces),
        )
        if selection_score > best_score:
            best_score = selection_score
            best_face = face
            best_quality = quality

    return best_face, best_quality


class InsightFaceEmbedder:
    def __init__(
        self,
        root: Path,
        model_name: str = "buffalo_l",
        det_size: int = 1024,
        providers: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.model_name = model_name
        self.det_size = int(det_size)
        self.providers = providers or ["CPUExecutionProvider"]
        self._app = None

    def _resolve_providers(self) -> list[str]:
        try:
            import onnxruntime as ort
        except ImportError:
            return self.providers

        available = set(ort.get_available_providers())
        resolved = [provider for provider in self.providers if provider in available]
        if not resolved and "CPUExecutionProvider" in available:
            resolved = ["CPUExecutionProvider"]
        if not resolved:
            raise RuntimeError(
                f"No requested ONNX Runtime providers are available. "
                f"requested={self.providers}, available={sorted(available)}"
            )
        skipped = [provider for provider in self.providers if provider not in available]
        if skipped:
            print(f"Skipping unavailable ONNX Runtime providers: {skipped}. Available: {sorted(available)}")
        return resolved

    def prepare(self) -> None:
        model_dir = self.root / "models" / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(
                f"InsightFace model directory not found: {model_dir}. "
                "Set --insightface-root to the directory that contains models/<model_name>."
            )

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency `insightface`. Install runtime dependencies with "
                "`python3 -m pip install -e .` or `python3 -m pip install -r requirements.txt`."
            ) from exc

        providers = self._resolve_providers()
        print(f"Using ONNX Runtime providers: {providers}")
        app = FaceAnalysis(name=self.model_name, root=str(self.root), providers=providers)
        app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
        self._app = app

    def extract(self, image_path: Path) -> FaceEmbedding | None:
        if self._app is None:
            self.prepare()

        image = load_bgr_array(image_path)
        image_height, image_width = image.shape[:2]
        faces = self._app.get(image)
        face, quality = select_best_face(faces, image_width=image_width, image_height=image_height)
        if face is None or quality is None:
            return None

        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            raise RuntimeError(f"No embedding returned for {image_path}")

        return FaceEmbedding(embedding=l2_normalize(embedding), quality=quality)

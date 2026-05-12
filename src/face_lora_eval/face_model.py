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


def resolve_onnx_providers(requested: list[str], available: list[str] | None = None) -> list[str]:
    if available is None:
        try:
            import onnxruntime as ort
        except ImportError:
            return requested
        available = list(ort.get_available_providers())

    available_set = set(available)
    resolved = [provider for provider in requested if provider in available_set]
    if not resolved and "CPUExecutionProvider" in available_set:
        resolved = ["CPUExecutionProvider"]
    if not resolved:
        raise RuntimeError(
            f"No requested ONNX Runtime providers are available. requested={requested}, available={sorted(available_set)}"
        )
    skipped = [provider for provider in requested if provider not in available_set]
    if skipped:
        print(f"Skipping unavailable ONNX Runtime providers: {skipped}. Available: {sorted(available_set)}")
    return resolved


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


def create_face_analysis(root: Path, model_name: str, providers: list[str], allowed_modules: list[str] | None = None):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `insightface`. Install runtime dependencies with "
            "`python3 -m pip install -e .` or `python3 -m pip install -r requirements.txt`."
        ) from exc

    if allowed_modules is None:
        return FaceAnalysis(name=model_name, root=str(root), providers=providers)
    try:
        return FaceAnalysis(name=model_name, root=str(root), providers=providers, allowed_modules=allowed_modules)
    except TypeError:
        return FaceAnalysis(name=model_name, root=str(root), providers=providers)


def preprocess_lvface_aligned_bgr(aligned_bgr: Any, input_size: int = 112):
    import numpy as np

    aligned_bgr = np.asarray(aligned_bgr)
    if aligned_bgr.shape[:2] != (input_size, input_size):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to resize LVFace aligned crops.") from exc
        aligned_bgr = cv2.resize(aligned_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    rgb = aligned_bgr[:, :, ::-1].astype("float32")
    normalized = (rgb - 127.5) / 127.5
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype("float32")


def normalize_onnx_embedding(output: Any):
    import numpy as np

    if isinstance(output, (list, tuple)):
        output = output[0]
    return l2_normalize(np.asarray(output, dtype="float32").reshape(-1))


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
        return resolve_onnx_providers(self.providers)

    def prepare(self) -> None:
        model_dir = self.root / "models" / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(
                f"InsightFace model directory not found: {model_dir}. "
                "Set --insightface-root to the directory that contains models/<model_name>."
            )

        providers = self._resolve_providers()
        print(f"Using ONNX Runtime providers: {providers}")
        app = create_face_analysis(self.root, self.model_name, providers)
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


class LVFaceOnnxEmbedder:
    def __init__(
        self,
        detector_root: Path,
        detector_model_name: str = "buffalo_l",
        detector_det_size: int = 1024,
        detector_providers: list[str] | None = None,
        model_path: Path | str = "models/lvface/LVFace-B_Glint360K/LVFace-B_Glint360K.onnx",
        providers: list[str] | None = None,
        input_size: int = 112,
    ) -> None:
        self.detector_root = Path(detector_root)
        self.detector_model_name = detector_model_name
        self.detector_det_size = int(detector_det_size)
        self.detector_providers = detector_providers or ["CPUExecutionProvider"]
        self.model_path = Path(model_path)
        self.providers = providers or ["CPUExecutionProvider"]
        self.input_size = int(input_size)
        self._detector_app = None
        self._session = None
        self._input_name = ""

    def prepare(self) -> None:
        detector_model_dir = self.detector_root / "models" / self.detector_model_name
        if not detector_model_dir.exists():
            raise FileNotFoundError(
                f"InsightFace detector model directory not found: {detector_model_dir}. "
                "LVFace still needs InsightFace detection/alignment."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"LVFace ONNX model not found: {self.model_path}. "
                "Run `python scripts/download_lvface.py` or set --lvface-model-path."
            )

        detector_providers = resolve_onnx_providers(self.detector_providers)
        print(f"Using InsightFace detector providers: {detector_providers}")
        detector_app = create_face_analysis(
            self.detector_root,
            self.detector_model_name,
            detector_providers,
            allowed_modules=["detection"],
        )
        detector_app.prepare(ctx_id=0, det_size=(self.detector_det_size, self.detector_det_size))

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency `onnxruntime`. Install runtime dependencies with "
                "`python3 -m pip install -e .` or `python3 -m pip install -r requirements.txt`."
            ) from exc

        lvface_providers = resolve_onnx_providers(self.providers)
        print(f"Using LVFace ONNX providers: {lvface_providers}")
        session = ort.InferenceSession(str(self.model_path), providers=lvface_providers)
        self._input_name = session.get_inputs()[0].name
        self._session = session
        self._detector_app = detector_app

    def _align_face(self, image_bgr: Any, face: Any):
        landmarks = getattr(face, "kps", None)
        if landmarks is None:
            raise RuntimeError("Selected face has no 5-point landmarks for LVFace alignment.")
        try:
            from insightface.utils import face_align
        except ImportError as exc:
            raise RuntimeError("Missing InsightFace face alignment utilities.") from exc
        return face_align.norm_crop(image_bgr, landmark=landmarks, image_size=self.input_size)

    def extract(self, image_path: Path) -> FaceEmbedding | None:
        if self._detector_app is None or self._session is None:
            self.prepare()

        image = load_bgr_array(image_path)
        image_height, image_width = image.shape[:2]
        faces = self._detector_app.get(image)
        face, quality = select_best_face(faces, image_width=image_width, image_height=image_height)
        if face is None or quality is None:
            return None

        aligned = self._align_face(image, face)
        batch = preprocess_lvface_aligned_bgr(aligned, input_size=self.input_size)
        output = self._session.run(None, {self._input_name: batch})
        return FaceEmbedding(embedding=normalize_onnx_embedding(output), quality=quality)


def create_embedder(config: Any):
    backend = getattr(config, "backend", "insightface")
    if backend == "insightface":
        return InsightFaceEmbedder(
            root=Path(config.insightface["root"]),
            model_name=str(config.insightface["model_name"]),
            det_size=int(config.insightface["det_size"]),
            providers=list(config.insightface["providers"]),
        )
    if backend == "lvface_onnx":
        return LVFaceOnnxEmbedder(
            detector_root=Path(config.insightface["root"]),
            detector_model_name=str(config.insightface["model_name"]),
            detector_det_size=int(config.insightface["det_size"]),
            detector_providers=list(config.insightface["providers"]),
            model_path=Path(config.lvface["model_path"]),
            providers=list(config.lvface["providers"]),
            input_size=int(config.lvface["input_size"]),
        )
    raise ValueError(f"Unsupported embedding backend: {backend}")

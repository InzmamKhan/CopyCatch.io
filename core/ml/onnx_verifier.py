import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image
from config.settings import settings

class ONNXVerifier:
    def __init__(self):
        self._ensure_model_exists()
        self.session = ort.InferenceSession(str(settings.model_path), providers=['CPUExecutionProvider'])

    def _ensure_model_exists(self):
        """Auto-downloads quantized ONNX model if not locally present."""
        if not settings.model_path.exists():
            settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Downloading ONNX fallback model to {settings.model_path}...")
            urllib.request.urlretrieve(settings.MODEL_URL, settings.model_path)

    def extract_embedding(self, image_path) -> np.ndarray:
        """Preprocesses image and extracts normalized embedding vector."""
        with Image.open(image_path) as img:
            resized = img.convert("RGB").resize((224, 224))
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            # NCHW layout
            tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, :]

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: tensor})[0].flatten()
        norm = np.linalg.norm(outputs)
        return outputs / norm if norm > 0 else outputs

    @staticmethod
    def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes Cosine Similarity between two normalized vectors."""
        return float(np.dot(emb1, emb2))
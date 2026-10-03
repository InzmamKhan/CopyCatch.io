from pathlib import Path
from dataclasses import dataclass

@dataclass
class Settings:
    # Hash distance thresholds
    DIRECT_DUPLICATE_THRESHOLD: int = 5    # Hamming distance <= 5 -> Direct duplicate
    AMBIGUOUS_THRESHOLD: int = 15          # Hamming distance 6-15 -> ONNX Fallback
    
    # Cosine similarity threshold for ONNX feature vectors
    COSINE_SIMILARITY_THRESHOLD: float = 0.85

    # Display / UI thumbnail size
    THUMBNAIL_SIZE: tuple[int, int] = (150, 150)
    
    # Supported file formats
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"
    )

    # ONNX Fallback Model configuration
    MODEL_DIR: Path = Path(__file__).parent.parent / "models"
    MODEL_NAME: str = "mobilenetv3_small_quantized.onnx"
    MODEL_URL: str = (
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    )

    @property
    def model_path(self) -> Path:
        return self.MODEL_DIR / self.MODEL_NAME

settings = Settings()
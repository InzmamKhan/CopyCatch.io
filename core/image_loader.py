from PIL import Image
from pathlib import Path
import io
from typing import Union, Any

class ImageLoader:
    """Utility class for loading images and generating thumbnails from disk paths or uploaded file streams."""

    @staticmethod
    def load_image(source: Union[str, Path, Any]) -> Image.Image:
        """Loads an image from either a file path or a Streamlit UploadedFile object."""
        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")
        else:
            source.seek(0)
            return Image.open(io.BytesIO(source.read())).convert("RGB")

    @staticmethod
    def create_thumbnail(source: Union[str, Path, Any], max_size: tuple = (200, 200)) -> Image.Image:
        """Generates a PIL image thumbnail for UI previews."""
        img = ImageLoader.load_image(source)
        img.thumbnail(max_size)
        return img
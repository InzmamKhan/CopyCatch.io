from pathlib import Path
from typing import List
from PIL import Image
from config.settings import settings
from core.models import ImageItem

class ImageLoader:
    @staticmethod
    def scan_directory(folder_path: str | Path) -> List[ImageItem]:
        """Recursively scans directory for supported image files."""
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return []

        image_items = []
        item_id = 0
        
        # Recursive scanning using rglob
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in settings.SUPPORTED_EXTENSIONS:
                image_items.append(
                    ImageItem(
                        id=item_id,
                        path=path,
                        filename=path.name
                    )
                )
                item_id += 1

        return image_items

    @staticmethod
    def create_thumbnail(image_path: Path) -> Image.Image:
        """Generates a lightweight in-memory PIL Thumbnail."""
        with Image.open(image_path) as img:
            img_copy = img.convert("RGB").copy()
            img_copy.thumbnail(settings.THUMBNAIL_SIZE)
            return img_copy
from abc import ABC, abstractmethod
from pathlib import Path

class BaseHasher(ABC):
    @abstractmethod
    def compute_hash(self, image_path: str | Path) -> int:
        """
        Abstract method to compute a 64-bit binary perceptual signature (hash)
        for a given image file path.
        """
        pass
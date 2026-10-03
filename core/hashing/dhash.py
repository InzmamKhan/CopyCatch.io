from pathlib import Path
import numpy as np
from PIL import Image
from core.hashing.base import BaseHasher

class DifferenceHasher(BaseHasher):
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size

    def compute_hash(self, image_path: str | Path) -> int:
        """
        Computes a 64-bit Difference Hash (dHash).
        Resizes to (hash_size + 1) x hash_size, converts to grayscale,
        and measures relative pixel intensity gradients between adjacent columns.
        """
        with Image.open(image_path) as img:
            # Resize to 9x8 for an 8x8 difference matrix (64 bits)
            resized = img.convert("L").resize(
                (self.hash_size + 1, self.hash_size), 
                Image.Resampling.LANCZOS
            )
            pixels = np.asarray(resized, dtype=np.int32)

        # Compute differences between adjacent column pixels (left > right)
        diff = pixels[:, :-1] > pixels[:, 1:]

        # Convert boolean matrix to 64-bit integer
        bit_string = "".join(["1" if bit else "0" for bit in diff.flatten()])
        return int(bit_string, 2)
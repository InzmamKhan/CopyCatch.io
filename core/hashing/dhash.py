from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
from core.hashing.base import BaseHasher

class dhash(BaseHasher):
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size

    def compute_hash(self, image: Union[Image.Image, str, Path]) -> int:
        """
        Computes a 64-bit Difference Hash (dHash).
        Accepts either a pre-loaded PIL Image or an image file path.
        """
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                resized = img.convert("L").resize(
                    (self.hash_size + 1, self.hash_size), 
                    Image.Resampling.LANCZOS
                )
        else:
            resized = image.convert("L").resize(
                (self.hash_size + 1, self.hash_size), 
                Image.Resampling.LANCZOS
            )

        pixels = np.asarray(resized, dtype=np.int32)

        diff = pixels[:, :-1] > pixels[:, 1:]

        bit_string = "".join(["1" if bit else "0" for bit in diff.flatten()])
        return int(bit_string, 2)
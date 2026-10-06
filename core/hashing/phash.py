from pathlib import Path
from typing import Union
import numpy as np
import scipy.fftpack
from PIL import Image
from core.hashing.base import BaseHasher

class phash(BaseHasher):
    def __init__(self, hash_size: int = 8, highfreq_factor: int = 4):
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor

    def compute_hash(self, image: Union[Image.Image, str, Path]) -> int:
        """
        Computes a 64-bit Perceptual Hash (pHash) using Discrete Cosine Transform (DCT).
        Accepts either a pre-loaded PIL Image or an image file path.
        """
        img_size = self.hash_size * self.highfreq_factor
        
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                gray_img = img.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
        else:
            gray_img = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)

        img_array = np.asarray(gray_img, dtype=np.float32)

        dct = scipy.fftpack.dct(scipy.fftpack.dct(img_array, axis=0, norm='ortho'), axis=1, norm='ortho')
        dct_low_freq = dct[:self.hash_size, :self.hash_size]

        med = np.median(dct_low_freq)
        diff = dct_low_freq > med

        bit_string = "".join(["1" if val else "0" for val in diff.flatten()])
        return int(bit_string, 2)
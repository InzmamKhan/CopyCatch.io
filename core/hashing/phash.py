from pathlib import Path
import numpy as np
from PIL import Image
import scipy.fftpack
from core.hashing.base import BaseHasher

class PerceptualHasher(BaseHasher):
    def __init__(self, hash_size: int = 8, highfreq_factor: int = 4):
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor

    def compute_hash(self, image_path: str | Path) -> int:
        """
        Computes a 64-bit Perceptual Hash (pHash) using Discrete Cosine Transform (DCT).
        Resizes to 32x32, extracts low-frequency DCT components, and thresholds by median.
        """
        img_size = self.hash_size * self.highfreq_factor
        
        with Image.open(image_path) as img:
            gray_img = img.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
            img_array = np.asarray(gray_img, dtype=np.float32)

        # 2D DCT calculation
        dct = scipy.fftpack.dct(scipy.fftpack.dct(img_array, axis=0, norm='ortho'), axis=1, norm='ortho')
        dct_low_freq = dct[:self.hash_size, :self.hash_size]

        # Median thresholding
        med = np.median(dct_low_freq)
        diff = dct_low_freq > med

        # Pack boolean matrix into 64-bit integer
        bit_string = "".join(["1" if val else "0" for val in diff.flatten()])
        return int(bit_string, 2)
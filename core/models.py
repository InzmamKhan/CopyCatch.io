from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np

@dataclass
class ImageItem:
    id: int
    path: Path
    filename: str
    hash_val: Optional[int] = None
    embedding: Optional[np.ndarray] = None

@dataclass
class DuplicateGroup:
    group_id: int
    images: List[ImageItem] = field(default_factory=list)
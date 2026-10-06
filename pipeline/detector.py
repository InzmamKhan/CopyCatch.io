from pathlib import Path
from typing import List, Union
import numpy as np

from core.models import ImageItem, DuplicateGroup
from core.image_loader import ImageLoader
from core.hashing.phash import pHash as PHash
from core.hashing.dhash import dHash as DHash
from core.indexer.faiss_index import BinaryFaissIndex
from core.ml.onnx_verifier import ONNXVerifier
from core.grouping.graph_cluster import GraphClusterer
from config.settings import settings

class DuplicateDetectorPipeline:
    """Core pipeline coordinating hashing, indexing, verification, and clustering."""

    def __init__(self):
        self.phash = PHash()
        self.dhash = DHash()
        self.verifier = ONNXVerifier()
        self.clusterer = GraphClusterer()

    def run(self, folder_path: Union[str, Path]) -> List[DuplicateGroup]:
        """Scans a local directory for duplicate images."""
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return []

        image_paths = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in settings.SUPPORTED_EXTENSIONS
        ]

        processed_images = []
        for path in image_paths:
            try:
                img = ImageLoader.load_image(path)
                phash_val = self.phash.compute_hash(img)
                dhash_val = self.dhash.compute_hash(img)
                combined_hash = (phash_val << 64) | dhash_val
                
                item = ImageItem(
                    path=path,
                    filename=path.name,
                    hash_val=combined_hash
                )
                processed_images.append(item)
            except Exception:
                continue

        return self._process_items(processed_images)

    def run_from_uploads(self, uploaded_files) -> List[DuplicateGroup]:
        """Processes in-memory uploaded image streams from Streamlit."""
        processed_images = []

        for file_obj in uploaded_files:
            try:
                img = ImageLoader.load_image(file_obj)
                phash_val = self.phash.compute_hash(img)
                dhash_val = self.dhash.compute_hash(img)
                combined_hash = (phash_val << 64) | dhash_val

                item = ImageItem(
                    path=file_obj,
                    filename=file_obj.name,
                    hash_val=combined_hash
                )
                processed_images.append(item)
            except Exception:
                continue

        return self._process_items(processed_images)

    def _process_items(self, items: List[ImageItem]) -> List[DuplicateGroup]:
        """Executes indexing, FAISS search, ONNX verification, and graph clustering."""
        if not items:
            return []

        indexer = BinaryFaissIndex(dimension_bits=128)
        hashes = [item.hash_val for item in items]
        indexer.build_index(hashes)

        candidate_pairs = indexer.search_range(max_distance=settings.AMBIGUOUS_THRESHOLD)

        confirmed_edges = []
        for i, j, dist in candidate_pairs:
            if dist <= settings.DIRECT_DUPLICATE_THRESHOLD:
                confirmed_edges.append((i, j))
            else:
                img_i = ImageLoader.load_image(items[i].path)
                img_j = ImageLoader.load_image(items[j].path)

                emb_i = self.verifier.extract_embedding(img_i)
                emb_j = self.verifier.extract_embedding(img_j)

                similarity = self.verifier.compute_cosine_similarity(emb_i, emb_j)
                if similarity >= settings.COSINE_SIMILARITY_THRESHOLD:
                    confirmed_edges.append((i, j))

        num_nodes = len(items)
        clusters = self.clusterer.cluster(num_nodes, confirmed_edges)

        duplicate_groups = []
        group_id_counter = 1
        for cluster_indices in clusters:
            group_images = [items[idx] for idx in cluster_indices]
            duplicate_groups.append(
                DuplicateGroup(group_id=group_id_counter, images=group_images)
            )
            group_id_counter += 1

        return duplicate_groups
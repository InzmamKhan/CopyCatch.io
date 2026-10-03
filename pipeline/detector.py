from typing import List
from config.settings import settings
from core.models import ImageItem, DuplicateGroup
from core.image_loader import ImageLoader
from core.hashing.phash import PerceptualHasher
from core.indexer.faiss_index import BinaryFaissIndex
from core.ml.onnx_verifier import ONNXVerifier
from core.grouping.graph_cluster import GraphClusterer

class DuplicateDetectorPipeline:
    def __init__(self):
        self.hasher = PerceptualHasher()
        self.verifier = None

    def run(self, folder_path: str) -> List[DuplicateGroup]:
        # Step 1: Scan Directory
        images = ImageLoader.scan_directory(folder_path)
        if not images:
            return []

        # Step 2: Compute pHash for all images
        hashes = []
        for img in images:
            img.hash_val = self.hasher.compute_hash(img.path)
            hashes.append(img.hash_val)

        # Step 3: Fast Binary Indexing in FAISS
        faiss_idx = BinaryFaissIndex(bit_dim=64)
        faiss_idx.add_hashes(hashes)
        
        limits, distances, indices = faiss_idx.range_search(settings.AMBIGUOUS_THRESHOLD)

        # Step 4: Evaluate Pairs & Trigger ONNX for edge cases
        match_pairs = []
        
        for i in range(len(images)):
            start, end = limits[i], limits[i + 1]
            for j in range(start, end):
                neighbor_idx = indices[j]
                dist = distances[j]

                if neighbor_idx <= i:
                    continue  # Skip self-matches and duplicate symmetry

                if dist <= settings.DIRECT_DUPLICATE_THRESHOLD:
                    match_pairs.append((i, neighbor_idx))
                elif settings.DIRECT_DUPLICATE_THRESHOLD < dist <= settings.AMBIGUOUS_THRESHOLD:
                    # Edge Case: Fall back to ONNX embedding check
                    if self.verifier is None:
                        self.verifier = ONNXVerifier()

                    if images[i].embedding is None:
                        images[i].embedding = self.verifier.extract_embedding(images[i].path)
                    if images[neighbor_idx].embedding is None:
                        images[neighbor_idx].embedding = self.verifier.extract_embedding(images[neighbor_idx].path)

                    sim = self.verifier.calculate_similarity(images[i].embedding, images[neighbor_idx].embedding)
                    if sim >= settings.COSINE_SIMILARITY_THRESHOLD:
                        match_pairs.append((i, neighbor_idx))

        # Step 5: Connected Components Graph Clustering (Returns only duplicate groups)
        return GraphClusterer.cluster_matches(images, match_pairs)
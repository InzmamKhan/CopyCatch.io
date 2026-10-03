import numpy as np
import faiss

class BinaryFaissIndex:
    def __init__(self, bit_dim: int = 64):
        self.bit_dim = bit_dim
        self.num_bytes = bit_dim // 8
        # IndexBinaryFlat uses bitwise Hamming distance (XOR + POPCNT)
        self.index = faiss.IndexBinaryFlat(self.bit_dim)

    def add_hashes(self, hashes: list[int]):
        """Converts uint64 hashes into uint8 byte arrays and adds to FAISS index."""
        byte_matrix = np.zeros((len(hashes), self.num_bytes), dtype=np.uint8)
        
        for idx, h in enumerate(hashes):
            for b in range(self.num_bytes):
                byte_matrix[idx, b] = (h >> (8 * (self.num_bytes - 1 - b))) & 0xFF

        self.index.add(byte_matrix)

    def range_search(self, threshold: int):
        """Performs Hamming range search returning pairs within max distance threshold."""
        n_elements = self.index.ntotal
        if n_elements == 0:
            return [], [], []

        # Extract vectors back from FAISS for self-search
        byte_matrix = np.zeros((n_elements, self.num_bytes), dtype=np.uint8)
        self.index.reconstruct_n(0, n_elements, byte_matrix)

        # Search within max Hamming distance range
        limits, distances, indices = self.index.range_search(byte_matrix, threshold + 1)
        return limits, distances, indices
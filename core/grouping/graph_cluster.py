import networkx as nx
from typing import List, Dict
from core.models import ImageItem, DuplicateGroup

class GraphClusterer:
    @staticmethod
    def cluster_matches(images: List[ImageItem], match_pairs: list[tuple[int, int]]) -> List[DuplicateGroup]:
        """
        Uses Connected Components graph algorithm to group matched image pairs.
        Returns only duplicate groups (clusters with 2 or more images). Distinct images are ignored.
        """
        G = nx.Graph()
        
        # Add all image IDs as nodes
        for img in images:
            G.add_node(img.id)

        # Add edges between matched pairs
        for u, v in match_pairs:
            G.add_edge(u, v)

        img_map: Dict[int, ImageItem] = {img.id: img for img in images}
        
        duplicate_groups = []
        group_id = 1

        for component in nx.connected_components(G):
            group_images = [img_map[node_id] for node_id in component]
            
            # Only keep clusters containing duplicates (size >= 2)
            if len(group_images) > 1:
                duplicate_groups.append(
                    DuplicateGroup(group_id=group_id, images=group_images)
                )
                group_id += 1

        return duplicate_groups
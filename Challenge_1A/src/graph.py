import networkx as nx
from scipy.spatial import cKDTree
def centroid(block):
    x0, y0, x1, y1 = block["bbox"]
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

def build_page_graph(blocks, k=4):
    """
    Given a list of text-block dicts (one page), return a networkx.Graph
    where each node is an index into `blocks` and edges connect each node
    to its k nearest neighbors by centroid distance.
    """
    # 1. Compute centroids list
    pts = [centroid(b) for b in blocks]
    
    # 2. Build a KD‑tree for fast neighbor lookup
    tree = cKDTree(pts)
    
    # 3. Query k+1 nearest (first is itself at distance 0)
    dists, nbrs = tree.query(pts, k=k+1)
    # dists: array shape (n_blocks, k+1), nbrs: same shape with indices
    
    # 4. Construct graph
    G = nx.Graph()
    for idx, block in enumerate(blocks):
        # Store the block metadata on the node
        G.add_node(idx, meta=block)
    
    # 5. Add edges (skip the zero‑distance self‑link)
    n = len(blocks)
    for i in range(n):
        for j in nbrs[i][1:]:      # skip nbrs[i][0] == i itself
            G.add_edge(i, int(j))
    
    return G

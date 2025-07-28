import re
import numpy as np
import pandas as pd

def centroid(block):
    x0, y0, x1, y1 = block["bbox"]
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

# --- Node‑Level Features ---

def node_level_features(block):
    spans = [s for line in block["lines"] for s in line["spans"]]
    sizes = [s["size"] for s in spans]
    flags = [s["flags"] for s in spans]
    text = "".join(s["text"] for s in spans).strip()

    # Basic typographic
    font_size = float(np.mean(sizes)) if sizes else 0.0
    font_is_bold = any((f & 2) != 0 for f in flags)  # flag bit 2 = bold

    # Positional (normalized)
    x0, y0, x1, y1 = block["bbox"]
    w, h = block["_page_width"], block["_page_height"]
    norm_x0 = x0 / w
    norm_y0 = y0 / h

    # Content‑based
    word_count = len(text.split())
    char_count = len(text)
    uppercase_ratio = sum(c.isupper() for c in text) / max(1, len(text))
    ends_with_punctuation = text.endswith(('.', '?', '!'))
    numbering_pattern = bool(re.match(r'^(\d+(\.\d+)*|[A-Za-z]\)|\([ivx]+\))\s', text))

    return {
        "font_size": font_size,
        "font_is_bold": int(font_is_bold),
        "norm_x0": norm_x0,
        "norm_y0": norm_y0,
        "word_count": word_count,
        "char_count": char_count,
        "uppercase_ratio": uppercase_ratio,
        "ends_with_punctuation": int(ends_with_punctuation),
        "numbering_pattern": int(numbering_pattern),
    }

# --- Relational (Graph‑Derived) Features ---

def relational_features(node_idx, G, k=None):
    """
    G: networkx Graph where G.nodes[i]['meta'] is the block dict.
    k: number of neighbors (inferred from G) or len(G[node_idx])
    """
    block = G.nodes[node_idx]["meta"]
    spans = [s for line in block["lines"] for s in line["spans"]]
    font_size = float(np.mean([s["size"] for s in spans])) if spans else 0.0

    neighbors = list(G[node_idx])
    neigh_sizes = []
    neigh_centroids = []
    for j in neighbors:
        b2 = G.nodes[j]["meta"]
        spans2 = [s for line in b2["lines"] for s in line["spans"]]
        if not spans2: continue
        neigh_sizes.append(np.mean([s["size"] for s in spans2]))
        neigh_centroids.append(centroid(b2))

    # Basic topological
    node_degree = len(neighbors)
    avg_neighbor_distance = np.mean(
        [np.linalg.norm(np.array(centroid(block)) - np.array(c)) for c in neigh_centroids]
    ) if neigh_centroids else 0.0

    # Differential typographic
    mean_neigh_size = np.mean(neigh_sizes) if neigh_sizes else font_size
    font_size_ratio = font_size / mean_neigh_size if mean_neigh_size else 1.0

    # Bold differential
    block_bold = any((s["flags"] & 2) != 0 for line in block["lines"] for s in line["spans"])
    neigh_bolds = [any((s["flags"] & 2) != 0 for line in G.nodes[j]["meta"]["lines"] for s in line["spans"]) for j in neighbors]
    bold_vs_neighbors = int(block_bold and not any(neigh_bolds))

    # Spatial differential
    # Find vertical distance to closest block above
    x0, y0, x1, y1 = block["bbox"]
    above_dists = []
    for j in neighbors:
        b2 = G.nodes[j]["meta"]
        _, y02, _, _ = b2["bbox"]
        if y02 < y0:
            above_dists.append(y0 - y02)
    space_above = min(above_dists) if above_dists else 0.0

    # Indentation vs. block below
    below_x0s = []
    for j in neighbors:
        b2 = G.nodes[j]["meta"]
        _, y02, _, _ = b2["bbox"]
        if y02 > y0:
            below_x0s.append(b2["bbox"][0])
    indentation_vs_below = (b2["bbox"][0] - x0) if below_x0s else 0.0

    return {
        "node_degree": node_degree,
        "avg_neighbor_distance": avg_neighbor_distance,
        "font_size_ratio": font_size_ratio,
        "bold_vs_neighbors": bold_vs_neighbors,
        "space_above": space_above,
        "indentation_vs_below": indentation_vs_below,
    }

# --- DataFrame Builder ---

def build_feature_dataframe(page_graphs):
    """
    page_graphs: list of networkx.Graphs, one per page.
    Returns a pandas.DataFrame with one row per node.
    """
    rows = []
    for pg_idx, G in enumerate(page_graphs, start=1):
        for i in G.nodes:
            feats = {}
            feats.update(node_level_features(G.nodes[i]["meta"]))
            feats.update(relational_features(i, G))
            # preserve page and block id
            feats["page_idx"] = pg_idx
            feats["node_idx"] = i
            rows.append(feats)
    return pd.DataFrame(rows)

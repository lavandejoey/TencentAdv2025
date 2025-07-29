# utils/embeddings.py
import numpy as np
from pathlib import Path

def load_vec(path: str):
    """
    Load space- or tab-delimited .vec file:
      first field = ID, rest = floats.
    Returns (np.ndarray [N,D], id2idx dict).
    """
    ids, vecs = [], []
    with open(Path(path), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 2: continue
            ids.append(parts[0])
            vecs.append([float(x) for x in parts[1:]])
    mat = np.vstack(vecs)
    id2idx = {nid:i for i,nid in enumerate(ids)}
    return mat, id2idx

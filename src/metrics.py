"""
Retrieval metrics and risk thresholds (NumPy only, no deep-learning dependency).

Evaluation protocol
-------------------
Every image in the evaluated set is used once as a query. The gallery is every
*other* image in the same set. A retrieved image is relevant when it shares the
query's parent_id (i.e. it is the same mark, possibly modified).
"""

import numpy as np


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def retrieval_eval(emb, labels, ks=(1, 5, 10)):
    """Per-query retrieval results.

    Returns a dict of NumPy arrays (one value per query):
      recall@k : True if at least one same-parent image is in the top-k
      ap       : average precision over the full ranking
      n_pos    : number of relevant images in the gallery for that query
    """
    emb = l2_normalize(emb)
    labels = np.asarray(labels)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)                     # never retrieve yourself
    order = np.argsort(-sim, axis=1)[:, :-1]           # drop self (ranked last)
    rel = labels[order] == labels[:, None]             # (N, N-1) relevance matrix

    out = {f"recall@{k}": rel[:, :k].any(axis=1) for k in ks}
    n_pos = rel.sum(axis=1)
    hits_so_far = np.cumsum(rel, axis=1)
    ranks = np.arange(1, rel.shape[1] + 1)
    precision_at_hit = (hits_so_far / ranks) * rel
    out["ap"] = precision_at_hit.sum(axis=1) / np.maximum(n_pos, 1)
    out["n_pos"] = n_pos
    return out


def summarize(per_query):
    """Average the per-query results into a single dict (mAP = mean of ap)."""
    summary = {}
    for key, values in per_query.items():
        if key == "n_pos":
            continue
        name = "mAP" if key == "ap" else key
        summary[name] = float(np.mean(values))
    return summary


def top_k(emb, query_idx, k=5):
    """Indices and scores of the k nearest gallery items for each query index."""
    emb = l2_normalize(emb)
    sim = emb[query_idx] @ emb.T
    sim[np.arange(len(query_idx)), query_idx] = -np.inf
    idx = np.argsort(-sim, axis=1)[:, :k]
    scores = np.take_along_axis(sim, idx, axis=1)
    return idx, scores


def pair_scores(emb, labels):
    """Cosine similarity of every unique pair, split into same-mark / different-mark."""
    emb = l2_normalize(emb)
    labels = np.asarray(labels)
    sim = emb @ emb.T
    iu = np.triu_indices(len(labels), k=1)
    scores = sim[iu]
    same = labels[iu[0]] == labels[iu[1]]
    return scores[same], scores[~same]


def choose_thresholds(pos_scores, neg_scores, high_precision=0.95, medium_precision=0.70):
    """Pick score cut-offs for the 'high' and 'medium' risk labels.

    high   = lowest score at which >= high_precision of the pairs above it are the same mark
    medium = lowest score at which >= medium_precision of the pairs above it are the same mark
    """
    pos = np.sort(pos_scores)
    neg = np.sort(neg_scores)
    grid = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 1000)
    n_pos_above = len(pos) - np.searchsorted(pos, grid, side="left")
    n_neg_above = len(neg) - np.searchsorted(neg, grid, side="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = n_pos_above / (n_pos_above + n_neg_above)

    def first_reaching(target):
        ok = np.where(precision >= target)[0]
        return float(grid[ok[0]]) if len(ok) else float(grid[-1])

    high = first_reaching(high_precision)
    medium = min(first_reaching(medium_precision), high)
    recall_high = float((pos >= high).mean())
    recall_medium = float((pos >= medium).mean())
    return {"high": round(high, 4), "medium": round(medium, 4),
            "high_precision_target": high_precision,
            "medium_precision_target": medium_precision,
            "recall_of_true_pairs_at_high": round(recall_high, 4),
            "recall_of_true_pairs_at_medium": round(recall_medium, 4)}


def risk_label(score, high, medium):
    if score >= high:
        return "High"
    if score >= medium:
        return "Medium"
    return "Low"

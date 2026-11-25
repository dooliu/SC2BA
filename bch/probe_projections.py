"""Same-frame projections and quantitative neighborhood agreement."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

from .upper_state.bktree_distance import distance_matrices

VIEW_NAMES = ("cnn", "bk-position", "bk-health")
NEIGHBORS = 5


def _tsne(distances, perplexity, seed):
    return TSNE(
        n_components=2, metric="precomputed", perplexity=perplexity, random_state=seed,
        init="random", learning_rate="auto", n_iter=1500, method="exact",
    ).fit_transform(distances.copy()).astype(float)


def _neighbors(distances, candidates, k):
    """Marginal inclusion probability under uniform tie breaking at the k-th neighbor."""
    eligible = candidates.sum(axis=1) >= k
    weights = np.zeros_like(distances)
    if not np.any(eligible):
        return weights, eligible
    choices = candidates[eligible]
    values = distances[eligible]
    kth = np.partition(np.where(choices, values, np.inf), k - 1, axis=1)[:, [k - 1]]
    tied = choices & np.isclose(values, kth, atol=1e-10, rtol=1e-9)
    closer = choices & (values < kth) & ~tied
    tie_weight = (k - closer.sum(axis=1)) / tied.sum(axis=1)
    weights[eligible] = closer + tied * tie_weight[:, None]
    return weights, eligible


def _overlap(left, right, anchors, k):
    if not np.any(anchors):
        return None
    return round(float((left[anchors] * right[anchors]).sum(axis=1).mean() / k), 6)


def _rank_agreement(left, right):
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    return round(float(spearmanr(left, right).statistic), 6)


def _agreement(matrices, projected, roster):
    n = len(roster)
    if n < 2:
        return None
    k = min(NEIGHBORS, n - 1)
    same = (roster[:, None, :] == roster[None, :, :]).all(axis=2)
    other = ~np.eye(n, dtype=bool)
    within = same & other
    pairs = np.triu(other, k=1)
    same_pairs = pairs & same
    weights = {name: _neighbors(matrix, other, k)[0] for name, matrix in matrices.items()}
    projected_weights = {name: _neighbors(matrix, other, k)[0] for name, matrix in projected.items()}
    within_weights = {name: _neighbors(matrix, within, k)[0] for name, matrix in matrices.items()}
    within_anchors = within.sum(axis=1) >= k
    return {
        "frames": n, "pairs": int(pairs.sum()), "neighbors": k,
        "same_roster_pairs": int(same_pairs.sum()), "same_roster_anchors": int(within_anchors.sum()),
        "random_overlap": round(k / (n - 1), 6),
        "within_roster_random_overlap": (
            round(float((k / within.sum(axis=1)[within_anchors]).mean()), 6)
            if np.any(within_anchors) else None
        ),
        "roster_chance": round(float(within.mean() * n / (n - 1)), 6),
        "roster_purity": {name: round(float((values * within).sum() / (n * k)), 6)
                          for name, values in weights.items()},
        "comparisons": {
            name: {
                "spearman": _rank_agreement(matrices["cnn"][pairs], matrices[name][pairs]),
                "same_roster_spearman": _rank_agreement(matrices["cnn"][same_pairs], matrices[name][same_pairs]),
                "neighbor_overlap": _overlap(weights["cnn"], weights[name], other.any(axis=1), k),
                "projected_overlap": _overlap(projected_weights["cnn"], projected_weights[name], other.any(axis=1), k),
                "same_roster_neighbor_overlap": _overlap(within_weights["cnn"], within_weights[name], within_anchors, k),
            }
            for name in ("bk-position", "bk-health")
        },
    }


def compare_projections(encoder, observations, relation_features, roster, scopes, seed):
    latents = encoder.embed_many(observations)
    position, hp = distance_matrices(observations)
    matrices = {
        "relation": cdist(relation_features, relation_features),
        "cnn": cdist(latents, latents),
        "bk-position": position,
        "bk-health": hp,
    }
    perplexity = min(20.0, len(observations) / 4)
    xy = {name: _tsne(matrices[name], perplexity, seed) for name in VIEW_NAMES}
    projected = {name: cdist(coords, coords) for name, coords in xy.items()}
    results = {}
    roster = np.asarray(roster)
    for name, indices in scopes.items():
        subset = np.ix_(indices, indices)
        results[name] = _agreement(
            {method: distance[subset] for method, distance in matrices.items()},
            {method: distance[subset] for method, distance in projected.items()},
            roster[indices],
        )
    meta = encoder.fit_manifest
    return {
        "coordinates": {name: coords.round(5).tolist() for name, coords in xy.items()},
        "perplexity": perplexity, "agreement": results,
        "cnn": {"sha256": encoder.fingerprint, **{key: meta[key] for key in
                                                ("seed", "latent_dim", "epochs", "best_epoch", "sample_counts")}},
    }, matrices

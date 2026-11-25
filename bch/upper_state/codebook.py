"""Weighted k-medoids for matched code budgets across representations."""

import numpy as np
from scipy.spatial.distance import cdist


def hash_features(value):
    return np.frombuffer(value.encode("ascii"), dtype=np.uint8)


def fit_medoids(features, size, metric, seed, iterations=16):
    points, weights = np.unique(features, return_counts=True, axis=0)
    if not 1 <= size <= len(points):
        raise ValueError(f"Cannot fit {size} medoids to {len(points)} distinct training inputs.")
    costs = cdist(points, points, metric=metric)
    rng = np.random.default_rng(seed)
    medoids = [int(rng.choice(len(points), p=weights / weights.sum()))]
    nearest = costs[:, medoids[0]].copy()
    while len(medoids) < size:
        probabilities = nearest * weights
        index = int(rng.choice(len(points), p=probabilities / probabilities.sum()))
        medoids.append(index)
        np.minimum(nearest, costs[:, index], out=nearest)
    medoids = np.asarray(medoids)
    for _ in range(iterations):
        groups = costs[:, medoids].argmin(axis=1)
        updated = []
        for index in range(size):
            members = np.flatnonzero(groups == index)
            within = costs[np.ix_(members, members)] @ weights[members]
            updated.append(members[within.argmin()])
        if np.array_equal(updated, medoids):
            break
        medoids = np.asarray(updated)
    return points[medoids], float(np.average(costs[:, medoids].min(axis=1), weights=weights))

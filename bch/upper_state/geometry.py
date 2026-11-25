"""Current-frame relationship targets and coverage diagnostics for Marines."""

import numpy as np
from scipy.spatial.distance import cdist


def _directed_labels(distances, attackers, targets):
    radii = np.array([unit.radius for unit in attackers])[:, None] + [unit.radius for unit in targets]
    in_range = distances <= 5.0 + radii + 1e-6
    near = np.isclose(distances, distances.min(axis=1, keepdims=True), atol=1e-6, rtol=0)
    far = np.isclose(distances, distances.max(axis=1, keepdims=True), atol=1e-6, rtol=0)
    return in_range.astype(np.uint8) | (near.astype(np.uint8) << 1) | (far.astype(np.uint8) << 2)


def relation_labels(friendly, enemy):
    """Range, nearest and farthest flags in each direction, in world units."""
    if not friendly or not enemy:
        return np.zeros((len(friendly), len(enemy)), dtype=np.uint8)
    distance = cdist([unit.xy for unit in friendly], [unit.xy for unit in enemy])
    return _directed_labels(distance, friendly, enemy) | (_directed_labels(distance.T, enemy, friendly).T << 3)


def geometry_target(observation):
    own, enemy = observation.friendly, observation.enemy
    counts = np.bincount(relation_labels(own, enemy).ravel(), minlength=64)
    return [len(own) * 16 / observation.influence_scale, len(enemy) * 16 / observation.influence_scale,
            *(counts / max(1, counts.sum()))]

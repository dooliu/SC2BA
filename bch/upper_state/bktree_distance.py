"""The two historical BK-tree dissimilarities, evaluated on SC2BA observations."""

from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def normalized_armies(observation):
    # Q-learning_new_bktree.py scales a 128-cell coordinate frame around 64.
    return tuple(
        np.asarray([(unit.x / 64 - 1, 1 - (observation.map_height - unit.y) / 64, unit.health / 45)
                    for unit in sorted(side, key=lambda unit: unit.tag)], dtype=float).reshape(-1, 3)
        for side in (observation.friendly, observation.enemy)
    )


def army_distances(left, right):
    """The positional assignment and its HP discrepancy, with the original dummy costs."""
    size = max(len(left), len(right))
    if not size:
        return 0.0, 0.0
    if not len(left) or not len(right):
        return float(size), 1.0
    cost = np.ones((size, size))
    cost[:len(left), :len(right)] = cdist(left[:, :2], right[:, :2])
    health_left = np.pad(left[:, 2], (0, size - len(left)))
    health_right = np.pad(right[:, 2], (0, size - len(right)))
    rows, cols = linear_sum_assignment(cost)
    hp_forward = np.abs(health_left[rows] - health_right[cols]).sum()
    # Tied positional assignments may yield different HP sums when the query direction changes.
    reverse_rows, reverse_cols = linear_sum_assignment(cost.T)
    hp_backward = np.abs(health_left[reverse_cols] - health_right[reverse_rows]).sum()
    return float(cost[rows, cols].sum()), float((hp_forward + hp_backward) / 2)


def distance_matrices(observations):
    armies = [normalized_armies(observation) for observation in observations]
    shape = (len(armies), len(armies))
    positions, health = np.zeros(shape), np.zeros(shape)
    for i, j in combinations(range(len(armies)), 2):
        position, hp = map(sum, zip(*(army_distances(left, right) for left, right in zip(armies[i], armies[j]))))
        positions[i, j] = positions[j, i] = position
        health[i, j] = health[j, i] = hp
    return positions, health

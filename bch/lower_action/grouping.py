"""The five original CBS grouping rules and the lower Q state."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from ..observation import Unit

GROUPINGS = ("k_means_000", "k_means_025", "k_means_050", "k_means_075", "k_means_100")


@dataclass(frozen=True)
class Group:
    center: Tuple[float, float]
    uniformity: float
    crowding: float
    units: Tuple[Unit, ...]


@dataclass(frozen=True)
class Formation:
    count: int
    variance: float
    groups: Tuple[Group, ...]

    @property
    def table(self):
        return self.count, self.variance

    @property
    def key(self):
        return "".join(format(int(value * 15.9), "X")
                       for group in self.groups for value in (group.uniformity, group.crowding)) or "X"


def center(units):
    return tuple(np.mean([unit.xy for unit in units], axis=0))


def describe_group(units):
    centroid = center(units)
    if len(units) == 1:
        return Group(centroid, 0.0, 1.0, units)
    points = [unit.xy for unit in units]
    radii = np.linalg.norm(np.asarray(points) - centroid, axis=1)
    uniformity = 1.0 - radii.std() / radii.mean() if radii.mean() else 1.0
    crowding = 1.0
    if radii.max() > 0:
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        perimeter = 2 * len(points) * radii.max() * np.sin(np.pi / len(points))
        crowding -= distances.min(axis=1).sum() / perimeter
    return Group(centroid, round(float(uniformity), 2), round(float(crowding), 2), units)


def group_units(units, rule, rng):
    index = GROUPINGS.index(rule)
    count = len(units) if index == 0 else 1 if index == 4 else max(1, len(units) * index // 4)
    if not units:
        return Formation(count, 0.0, ())
    if index == 0:
        teams = [(unit,) for unit in units]
    elif count == 1:
        teams = [units]
    else:
        labels = KMeans(n_clusters=count, n_init=1, random_state=int(rng.integers(2**31 - 1))).fit_predict(
            [unit.xy for unit in units]
        )
        teams = [tuple(unit for unit, label in zip(units, labels) if label == group) for group in sorted(set(labels))]
    groups = tuple(describe_group(team) for team in teams)
    variance = float(np.var([group.center for group in groups], axis=0).sum())
    return Formation(count, round(variance * 2) / 2, groups)

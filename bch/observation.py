"""Plain data shared by the environment, encoder and action library."""

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class Unit:
    tag: int
    x: float
    y: float
    health: float
    health_max: float
    radius: float
    cooldown: float

    @property
    def xy(self):
        return self.x, self.y


@dataclass(frozen=True)
class Observation:
    friendly: Tuple[Unit, ...]
    enemy: Tuple[Unit, ...]
    step: int
    game_loop: int
    map_height: int
    influence_scale: float

    @property
    def margin(self):
        return sum(unit.health for unit in self.friendly) - sum(unit.health for unit in self.enemy)

    def record(self):
        return asdict(self)

    @classmethod
    def from_record(cls, row):
        row = dict(row)
        for side in ("friendly", "enemy"):
            row[side] = tuple(Unit(**unit) for unit in row[side])
        return cls(**row)

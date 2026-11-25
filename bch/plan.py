"""Explicit scenario and layout folds shared by collection and policy evaluation."""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Optional, Tuple

import numpy as np

from .lower_action.grouping import GROUPINGS
from .lower_action.macros import MACROS
from .records import FOLDS

Point = Tuple[float, float]


@dataclass(frozen=True)
class Layout:
    id: str
    split: str
    red: Optional[Point]
    blue: Optional[Point]
    red_units: Tuple[Point, ...]
    blue_units: Tuple[Point, ...]

    @property
    def positions(self):
        return self.red, self.blue, tuple(sorted(self.red_units)), tuple(sorted(self.blue_units))


@dataclass(frozen=True)
class Scenario:
    name: str
    opponents: Tuple[str, ...]
    layouts: Tuple[Layout, ...]


def _point(values):
    if len(values) != 2 or not all(isinstance(value, (float, int)) and isfinite(value) for value in values):
        raise ValueError("A spawn must be two finite world coordinates.")
    return tuple(values)


def _layout(record):
    red = _point(record["red"]) if "red" in record else None
    blue = _point(record["blue"]) if "blue" in record else None
    red_units = tuple(map(_point, record.get("red_units", ())))
    blue_units = tuple(map(_point, record.get("blue_units", ())))
    explicit = bool(red_units or blue_units)
    if (explicit and (not red_units or not blue_units or red or blue)) or (not explicit and not red):
        raise ValueError("Use a red center with an optional blue center, or give both exact armies.")
    if len(set(red_units + blue_units)) != len(red_units) + len(blue_units):
        raise ValueError("Each explicitly placed Marine needs a distinct position.")
    ident, split = record["id"], record["split"]
    if not isinstance(ident, str) or not ident.strip() or split not in FOLDS:
        raise ValueError("A layout needs an id and a train, validation or test split.")
    return Layout(ident, split, red, blue, red_units, blue_units)


def read_plan(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    scenarios = {}
    for entry in document["scenarios"]:
        name = entry["name"]
        layouts = tuple(map(_layout, entry["layouts"]))
        opponents = tuple(entry["opponents"])
        if name in scenarios or not layouts or not opponents or len(opponents) != len(set(opponents)):
            raise ValueError(f"Repeated scenario or missing or repeated opponents in {name}.")
        if len({layout.id for layout in layouts}) != len(layouts):
            raise ValueError(f"Repeated layout id in {name}.")
        if len({layout.positions for layout in layouts}) != len(layouts):
            raise ValueError(f"Repeated spawn positions in {name}.")
        scenarios[name] = Scenario(name, opponents, layouts)
    groups = tuple(document.get("groupings", GROUPINGS))
    declared = document.get("behaviors", {"mixed": document.get("macros")})
    behaviors = {name: MACROS if macros is None else tuple(macros) for name, macros in declared.items()}
    if not scenarios or not behaviors or any(not name.strip() for name in behaviors):
        raise ValueError("A plan needs scenarios and named probe behaviors.")
    _validate_actions(groups, GROUPINGS)
    for macros in behaviors.values():
        _validate_actions(macros, MACROS)
    return document, scenarios, groups, behaviors


def _validate_actions(chosen, available):
    if not chosen or len(chosen) != len(set(chosen)) or not set(chosen).issubset(available):
        raise ValueError(f"Invalid or repeated action in {chosen}.")


def plan_digest(document):
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def scenario_seeds(seed, document, scenario):
    index = next(index for index, item in enumerate(document["scenarios"]) if item["name"] == scenario)
    engine, order, actions = np.random.SeedSequence([seed, index]).spawn(3)
    return int(engine.generate_state(1)[0]), np.random.default_rng(order), np.random.default_rng(actions)


def cases(scenario, repeats, rng, split=None):
    pool = [(layout, opponent) for layout in scenario.layouts if split is None or layout.split == split
            for opponent in scenario.opponents]
    if not pool:
        raise ValueError(f"{scenario.name} has no {split} layouts.")
    for repeat in range(repeats):
        for index in rng.permutation(len(pool)):
            layout, opponent = pool[index]
            yield repeat, layout, opponent

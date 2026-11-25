"""Translate the eleven historical macros into ordered SC2-independent commands."""

from dataclasses import dataclass
from math import dist
from typing import Tuple, Union

from .grouping import center


@dataclass(frozen=True)
class Command:
    ability: str
    units: Tuple[int, ...]
    target: Union[int, Tuple[float, float]]
    queued: bool = False


def _order(ability, units, target, queued=False):
    return Command(ability, tuple(unit.tag for unit in units), target, queued)


def _nearest(position, units, *, prefer_health=False):
    return min(units, key=lambda unit: (dist(position, unit.xy), unit.health if prefer_health else 0, unit.tag))


def _group_target(group, enemies, weak=False):
    def rank(enemy):
        distances = [dist(unit.xy, enemy.xy) for unit in group.units]
        return (-sum(d < 5.0 for d in distances), enemy.health if weak else sum(distances), enemy.tag)
    return min(enemies, key=rank)


def _nearest_per_unit(friendly, enemy, groups):
    return [_order("smart", (unit,), _nearest(unit.xy, enemy).tag) for unit in friendly]


def _nearest_per_group(friendly, enemy, groups):
    return _group_attacks(groups, enemy)


def _nearest_weakest(friendly, enemy, groups):
    target = _nearest(center(friendly), enemy, prefer_health=True)
    return [_order("smart", friendly, target.tag)]


def _weakest_per_group(friendly, enemy, groups):
    return _group_attacks(groups, enemy, weak=True)


def _group_attacks(groups, enemy, weak=False):
    commands = []
    for group in groups:
        tag = _group_target(group, enemy, weak).tag
        commands.extend(_order("smart", (unit,), tag) for unit in group.units)
    return commands


def _threatening(friendly, enemy, groups):
    origin = center(friendly)
    target = min(enemy, key=lambda unit: (-unit.health, dist(origin, unit.xy), unit.tag))
    return [_order("smart", friendly, target.tag)]


def _withdraw(friendly, enemy, groups):
    commands = []
    for group in groups:
        threats = {_nearest(unit.xy, enemy).tag for unit in group.units}
        threat = center(tuple(unit for unit in enemy if unit.tag in threats))
        point = tuple(x + 3 * (x - y) for x, y in zip(group.center, threat))
        commands.extend(_order("smart", (unit,), point) for unit in group.units)
    return commands


def _gather(friendly, enemy, groups):
    commands = [_order("smart", group.units, group.center) for group in groups if group.crowding < 0.4]
    return commands + _nearest_per_unit(friendly, enemy, groups)


def _lure(friendly, enemy, groups):
    target = center(enemy)
    separated = tuple(unit for group in groups if group.uniformity < 0.4 for unit in group.units)
    lure = _nearest(target, separated if len(separated) > 1 else friendly)
    rest = tuple(unit for unit in friendly if unit.tag != lure.tag)
    commands = [_order("smart", (lure,), target)]
    own = center(friendly)
    commands.extend(_order("smart", (unit,), own) for unit in rest)
    return commands


def _sacrifice_lure(friendly, enemy, groups):
    own, opposing = center(friendly), center(enemy)
    lure = _nearest(opposing, friendly, prefer_health=True)
    target = _nearest(own, enemy, prefer_health=True).tag
    commands = [_order("smart", (lure,), target)]
    rest = tuple(unit for unit in friendly if unit.tag != lure.tag)
    if rest:
        point = tuple((6 * x - y) / 5 for x, y in zip(own, opposing))
        commands += [_order("smart", rest, target, queued=True), _order("move", rest, point)]
    return commands


def _attack_point(friendly, enemy, groups):
    # In the source, do_randomly is a point attack at feature coordinate (50, 50).
    return [_order("attack", friendly, (50.0, 50.0))]


def _idle(friendly, enemy, groups):
    return []


_MACROS = {
    "action_ATK_nearest": _nearest_per_unit,
    "action_ATK_clu_nearest": _nearest_per_group,
    "action_ATK_nearest_weakest": _nearest_weakest,
    "action_ATK_clu_nearest_weakest": _weakest_per_group,
    "action_ATK_threatening": _threatening,
    "action_DEF_clu_nearest": _withdraw,
    "action_MIX_gather": _gather,
    "action_MIX_lure": _lure,
    "action_MIX_sacrifice_lure": _sacrifice_lure,
    "do_randomly": _attack_point,
    "do_nothing": _idle,
}
MACROS = tuple(_MACROS)


def macro_commands(observation, formation, name):
    if not observation.friendly or not observation.enemy:
        return []
    return _MACROS[name](observation.friendly, observation.enemy, formation.groups)

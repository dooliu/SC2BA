"""The two original Q layers with independently replaceable state and action modules."""

from math import sqrt

import numpy as np

from .lower_action.grouping import GROUPINGS, group_units
from .lower_action.macros import MACROS, macro_commands
from .lower_action.reward import short_reward


class ActionTable:
    def __init__(self, actions):
        self.actions = actions
        self.rows = {}

    def row(self, key):
        if key not in self.rows:
            self.rows[key] = np.zeros(len(self.actions), dtype=np.float64)
        return self.rows[key]

    def choose(self, key, rng, learn, exploit):
        row = self.row(key) if learn else self.rows.get(key)
        if row is None:
            return int(rng.integers(len(self.actions))), True
        if learn and rng.random() >= exploit:
            return int(rng.integers(len(self.actions))), False
        return int(rng.choice(np.flatnonzero(row == row.max()))), False

    def update(self, key, action, reward, next_key):
        row = self.rows[key]
        target = reward if next_key is None else reward + 0.9 * self.row(next_key).max()
        row[action] += 0.1 * (target - row[action])

    def snapshot(self):
        return {key: row.tolist() for key, row in self.rows.items()}

    def restore(self, rows):
        self.rows = {key: np.asarray(row, dtype=np.float64) for key, row in rows.items()}
        if any(row.shape != (len(self.actions),) or not np.isfinite(row).all() for row in self.rows.values()):
            raise ValueError("Q table has invalid rows.")


class HierarchicalPolicy:
    def __init__(self, encoder):
        self.encoder = encoder
        self.upper = ActionTable(GROUPINGS)
        self.lower = {}
        self.episodes = 0
        self.begin_episode(train=False)

    def begin_episode(self, train):
        self.train = train
        self.previous = self.previous_upper = None
        self.previous_lower = {}
        if train:
            self.episodes += 1
        self.exploit = 1 - 0.5 / sqrt(self.episodes) if train else 1.0

    def decide(self, observation, rng, *, terminal=False):
        key = self.encoder.key(observation)
        group_id, upper_unseen = self.upper.choose(key, rng, self.train, self.exploit)
        grouping = GROUPINGS[group_id]
        formation = group_units(observation.friendly, grouping, rng)
        tag, sub_key = formation.table, formation.key
        table = self.lower.get(tag)
        if table is None and self.train:
            table = self.lower[tag] = ActionTable(MACROS)
        if table is None:
            macro_id, lower_unseen = int(rng.integers(len(MACROS))), True
        else:
            macro_id, lower_unseen = table.choose(sub_key, rng, self.train, self.exploit)

        if self.train and self.previous is not None:
            self.upper.update(*self.previous_upper, observation.margin - self.previous.margin,
                              None if terminal else key)
            # The original learner matches a previous lower decision by the current grouping table.
            if tag in self.previous_lower:
                table.update(*self.previous_lower[tag], short_reward(self.previous, observation),
                             None if terminal else sub_key)
        if self.train:
            self.previous = observation
            self.previous_upper = (key, group_id)
            self.previous_lower[tag] = (sub_key, macro_id)

        macro = MACROS[macro_id]
        commands = [] if terminal else macro_commands(observation, formation, macro)
        return commands, (key, upper_unseen, lower_unseen)

    def snapshot(self):
        return {"groupings": list(GROUPINGS), "macros": list(MACROS), "episodes_trained": self.episodes,
                "upper": self.upper.snapshot(),
                "lower": [{"tag": list(tag), "rows": table.snapshot()} for tag, table in sorted(self.lower.items())]}

    def restore(self, record):
        if record["groupings"] != list(GROUPINGS) or record["macros"] != list(MACROS):
            raise ValueError("Q checkpoint action lists do not match.")
        self.episodes = record["episodes_trained"]
        self.upper.restore(record["upper"])
        self.lower.clear()
        for item in record["lower"]:
            table = self.lower[tuple(item["tag"])] = ActionTable(MACROS)
            table.restore(item["rows"])

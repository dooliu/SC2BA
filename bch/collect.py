"""Collect paired observations and outcomes under a mixed probe policy."""

from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import product
import gzip

import numpy as np

from .arena import Arena
from .lower_action.grouping import group_units
from .lower_action.macros import macro_commands
from .plan import cases, plan_digest, scenario_seeds
from .records import digest_file, json_line, write_json
from .upper_state.geometry import relation_labels


def _action_cycle(pairs, rng):
    while True:
        for index in rng.permutation(len(pairs)):
            yield pairs[index]


def _behavior_rounds(scenario, repeats, choices, order_rng, policy_rng):
    behaviors = tuple(choices)
    for repeat, layout, opponent in cases(scenario, repeats, order_rng):
        episode_seed = int(policy_rng.integers(2**32))
        for index in np.random.default_rng(episode_seed).permutation(len(behaviors)):
            yield repeat, layout, opponent, behaviors[index], episode_seed


def _episodes(args, document, scenarios, choices, samples, episodes):
    fields = ("scenario", "opponent", "split", "grouping", "macro", "behavior")
    counts = {field: Counter() for field in (*fields, "alive")}
    relation_profiles = defaultdict(set)
    count = 0
    for scenario in scenarios:
        engine_seed, order_rng, policy_rng = scenario_seeds(args.seed, document, scenario.name)
        schedules = {}
        with Arena(scenario, engine_seed, step_mul=args.step_mul) as arena:
            rounds = _behavior_rounds(scenario, args.repeats, choices, order_rng, policy_rng)
            for repeat, layout, opponent, behavior, episode_seed in rounds:
                case_key = layout.id, opponent, behavior
                if case_key not in schedules:
                    schedules[case_key] = _action_cycle(choices[behavior], np.random.default_rng(episode_seed))
                schedule = schedules[case_key]
                rng = np.random.default_rng(episode_seed)
                metadata = {
                    "episode": f"{scenario.name}/{layout.id}/{opponent}/{behavior}/{repeat}",
                    "scenario": scenario.name, "layout": layout.id, "opponent": opponent,
                    "split": layout.split, "behavior": behavior,
                    "repeat": repeat, "engine_seed": engine_seed, "episode_seed": episode_seed,
                }
                first = current = arena.reset_round(layout, opponent)
                done = False
                while not done:
                    before = current
                    grouping, macro = next(schedule)
                    commands_trace = []
                    for _ in range(args.horizon):
                        formation = group_units(current.friendly, grouping, rng)
                        commands = macro_commands(current, formation, macro)
                        commands_trace.append([asdict(command) for command in commands])
                        current, done, info = arena.step_macros(commands)
                        if done:
                            break
                    row = {
                        **metadata, "grouping": grouping, "macro": macro, "state": before.record(),
                        "hp_change": current.margin - before.margin, "steps": current.step - before.step,
                        "terminal": done, "commands": commands_trace,
                    }
                    samples.write(json_line(row))
                    count += 1
                    own, enemy = before.friendly, before.enemy
                    counts["alive"][f"{len(own)}v{len(enemy)}"] += 1
                    for field in fields:
                        counts[field][row[field]] += 1
                    relations = relation_labels(own, enemy)
                    histogram = tuple(np.bincount(relations.ravel(), minlength=64))
                    relation_profiles[scenario.name].add((len(own), len(enemy), histogram))
                audit = arena.episode_details(first, current, info)
                episodes.write(json_line({**metadata, **audit}))
                episodes.flush()
                print(f'{metadata["episode"]}: {audit["steps"]} steps, {audit["outcome"]}', flush=True)
    return count, counts, {name: len(profiles) for name, profiles in relation_profiles.items()}


def collect(args, document, scenarios, groupings, behaviors):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    sample_path, episode_path = out / "probes.jsonl.gz", out / "episodes.jsonl"
    with gzip.open(sample_path, "wt", encoding="utf-8") as samples, episode_path.open("w", encoding="utf-8") as episodes:
        choices = {name: tuple(product(groupings, macros)) for name, macros in behaviors.items()}
        count, counts, profiles = _episodes(args, document, scenarios, choices, samples, episodes)
    write_json(out / "manifest.json", {
        "format": "bch-sc2ba-probes-v1", "plan": document, "plan_sha256": plan_digest(document),
        "root_seed": args.seed, "repeats": args.repeats, "step_mul": args.step_mul, "horizon": args.horizon,
        "behaviors": list(behaviors),
        "samples": count, "counts": counts, "distinct_strict_relation_histograms": profiles,
        "probes_sha256": digest_file(sample_path), "episodes_sha256": digest_file(episode_path),
    })
    print(f"{count} probes written to {out}", flush=True)

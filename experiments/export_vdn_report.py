"""Export the completed SC2BA experiment and paired bootstrap summaries."""

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict, namedtuple
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import numpy as np


PHASES = ("original", "trained")
RESAMPLES = 10_000
BOOTSTRAP_SEED = 26112402
Outcome = namedtuple("Outcome", "policy opponent x y outcome steps red_return")


def read_outcomes(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            episode = json.loads(line)
            yield Outcome(episode["phase"], episode["opponent"], *episode["red_spawn"],
                          episode["outcome"], episode["steps"], episode["red_return"])


def box(samples):
    q1, median, q3 = np.quantile(samples, [0.25, 0.5, 0.75])
    spread = 1.5 * (q3 - q1)
    inside = samples[(samples >= q1 - spread - 1e-9) & (samples <= q3 + spread + 1e-9)]
    return {
        "q1": float(q1), "median": float(median), "q3": float(q3),
        "low": float(inside.min()), "high": float(inside.max()),
        "outliers": np.unique(samples[(samples < inside.min()) | (samples > inside.max())]).tolist(),
        "interval95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def bootstrap(episodes, comparison, layouts, rng):
    by_policy = defaultdict(dict)
    for row in episodes:
        spawn = (row.x, row.y)
        if spawn in by_policy[row.policy]:
            raise ValueError(f"Duplicate matchup layout: {row}")
        by_policy[row.policy][spawn] = row
    if set(by_policy) != set(PHASES):
        raise ValueError(f"Missing matchup policies: {comparison['opponent']}")

    wins = {}
    for phase in PHASES:
        games = by_policy[phase]
        if set(games) != set(layouts):
            raise ValueError(f"Mismatched layouts: {comparison['opponent']} {phase}")
        counts = Counter(game.outcome for game in games.values())
        if any(counts[outcome] != comparison[f"{phase}_{outcome}"] for outcome in ("win", "loss", "draw")):
            raise ValueError(f"Outcome totals differ from the release: {comparison['opponent']} {phase}")
        if not math.isclose(counts["win"] / len(layouts), comparison[f"{phase}_win_rate"]):
            raise ValueError("Win rate differs from the release")
        wins[phase] = np.array([games[spawn].outcome == "win" for spawn in layouts])

    picks = rng.integers(0, len(layouts), size=(RESAMPLES, len(layouts)))
    original = 100 * wins["original"][picks].mean(axis=1)
    trained = 100 * wins["trained"][picks].mean(axis=1)
    return {"original": box(original), "trained": box(trained), "delta": box(trained - original)}


def export(results, output, completed_at):
    def load(name):
        return json.loads((results / name).read_text(encoding="utf-8"))

    config, status, result = (load(name) for name in ("config.json", "status.json", "result.json"))
    if status["state"] != "complete" or not result["converged"]:
        raise ValueError("The formal experiment has not finished its convergence-gated test")

    episodes = list(read_outcomes(results / "test_trajectories.jsonl.gz"))
    groups = defaultdict(list)
    for row in episodes:
        groups[row.opponent].append(row)
    if set(groups) != {row["opponent"] for row in result["comparisons"]}:
        raise ValueError("Opponent coverage differs from the release")
    layouts = [tuple(point) for point in config["test_layouts"]]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    opponents = [name for name, _ in config["opponents"]]
    cadence = Counter()
    steps = 0
    with (results / "training_episodes.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            cadence[row["opponent"]] += 1
            steps += row["steps"]
    if set(cadence) != set(opponents) or sum(cadence.values()) != result["episodes"] or steps != result["training_steps"]:
        raise ValueError("Training logs differ from the release")

    comparisons = []
    for row in result["comparisons"]:
        name = row["opponent"]
        comparisons.append({
            **row, "training_episodes": cadence[name],
            "box": bootstrap(groups[name], row, layouts, rng),
        })
    for phase in PHASES:
        wins = sum(row[f"{phase}_win"] for row in comparisons)
        if not math.isclose(wins / (len(opponents) * len(layouts)), result[phase]["win_rate"]):
            raise ValueError(f"Overall win rate differs from the release: {phase}")

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "batch": "vdn-formal-20261124", "version": "mixed-opponents-v1",
        "completed_at": datetime.fromisoformat(completed_at).isoformat(),
        "algorithm": "VDN", "scenario": "3m", "game_version": "5.0.16.97425",
        "training_steps": result["training_steps"], "episodes": result["episodes"],
        "updates": result["optimizer_updates"], "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": RESAMPLES},
        "seed": config["seed"], "device": config["device"], "actors": config["actors"],
        "batch_size": config["batch_size"], "replay_size": config["replay_size"],
        "target_update": config["target_update"], "warmup_steps": config["warmup_steps"],
        "initial_lr": config["initial_lr"], "final_lr": config["final_lr"],
        "minimum_steps": config["min_steps"], "training_layouts": config["training_layout_count"],
        "validation_layouts": config["val_rounds"], "test_layouts": len(layouts),
        "plateau": config["plateau"], "validation": result["validation"],
        "original": result["original"], "trained": result["trained"], "comparisons": comparisons,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "test-outcomes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(Outcome._fields)
        writer.writerows(episodes)
    copyfile(results / "matchups.csv", output / "matchups.csv")
    print(f"Verified {len(episodes):,} paired-layout games and {len(comparisons)} matchup rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-at", required=True)
    args = parser.parse_args()
    export(args.results, args.output, args.completed_at)

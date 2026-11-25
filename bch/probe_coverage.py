"""Rebuild the fixed-vs-mixed probe coverage report from an SC2BA archive."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from math import ceil, floor
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from .observation import Observation
from .records import digest_file, write_json
from .upper_state.geometry import relation_labels

FIXED, MIXED = "legacy-nearest", "mixed"
SCOPES = ("all", "later")
TEMPLATE = Path(__file__).with_name("visualization") / "probe_coverage.html"


def _archive(path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    data = path / "probes.jsonl.gz"
    if manifest["format"] != "bch-sc2ba-probes-v1" or digest_file(data) != manifest["probes_sha256"]:
        raise ValueError("The probe archive does not match its manifest.")
    scenes = defaultdict(list)
    with gzip.open(data, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["behavior"] in (FIXED, MIXED):
                scenes[row["scenario"]].append(row)
    if not scenes:
        raise ValueError("The archive has no fixed or mixed probe windows.")
    return manifest, scenes


def _case(frame):
    return frame["layout"], frame["opponent"], frame["repeat"]


def _frames(rows):
    frames, signatures = [], []
    previous = {}
    for row in sorted(rows, key=lambda r: (r["episode"], r["state"]["step"])):
        observation = Observation.from_record(row["state"])
        labels = relation_labels(observation.friendly, observation.enemy)
        histogram = np.bincount(labels.ravel(), minlength=64).tolist()
        signatures.append((*labels.shape, *histogram))
        frame = {field: row[field] for field in
                 ("episode", "layout", "opponent", "repeat", "split", "behavior", "grouping", "macro")}
        frame.update(
            step=observation.step, previous=previous.get(row["episode"]), hp_change=row["hp_change"],
            reach=np.argwhere(labels & 1).tolist(),
            units=[[[round(unit.x, 3), round(unit.y, 3), unit.health] for unit in side]
                   for side in (observation.friendly, observation.enemy)],
        )
        previous[row["episode"]] = row["macro"]
        frames.append(frame)
    by_case_frame = {(*_case(frame), frame["step"], frame["behavior"]): i for i, frame in enumerate(frames)}
    if len(by_case_frame) != len(frames):
        raise ValueError("The archive lists a case, behavior and decision step more than once.")
    for frame in frames:
        opposite = MIXED if frame["behavior"] == FIXED else FIXED
        frame["paired"] = by_case_frame.get((*_case(frame), frame["step"], opposite))
    return frames, signatures


def _project(signatures, seed):
    values = np.asarray(signatures, dtype=float)
    if len(values) < 3:
        raise ValueError("At least three distinct relation profiles are required per scenario.")
    alive, histogram = values[:, :2], values[:, 2:]
    rates = alive / np.maximum(alive.max(axis=0), 1)
    frequencies = histogram / np.maximum(histogram.sum(axis=1, keepdims=True), 1)
    features = np.column_stack((rates, np.sqrt(frequencies / 2)))
    perplexity = min(20.0, len(features) / 4)
    coordinates = TSNE(
        n_components=2, perplexity=perplexity, random_state=seed,
        init="pca", learning_rate="auto", n_iter=1500, method="exact",
    ).fit_transform(features)
    return coordinates.round(5).tolist(), perplexity


def _matched_coverage(frames, trials, seed):
    groups = defaultdict(lambda: defaultdict(list))
    for row in frames:
        groups[_case(row)][row["behavior"]].append(row["point"])
    paired = []
    for modes in groups.values():
        fixed, mixed = modes[FIXED], modes[MIXED]
        quota = min(len(fixed), len(mixed))
        if quota:
            paired.append((fixed, mixed, quota))
    if not paired:
        return None
    sizes = np.empty((trials, 3), dtype=int)
    rng = np.random.default_rng(seed)
    for index in range(trials):
        fixed_covered, mixed_covered, combined = set(), set(), set()
        for fixed, mixed, quota in paired:
            f = rng.choice(fixed, quota, replace=False)
            m = rng.choice(mixed, quota, replace=False)
            # When a case has an odd quota, alternate which behavior gets the extra draw in expectation.
            n_fixed = quota // 2 + int(quota % 2 and rng.integers(2))
            fixed_covered.update(f)
            mixed_covered.update(m)
            combined.update(f[:n_fixed])
            combined.update(m[:quota - n_fixed])
        sizes[index] = len(fixed_covered), len(mixed_covered), len(combined)

    def interval(values):
        low, high = np.quantile(values, [0.05, 0.95])
        return {"mean": round(float(values.mean()), 3), "p05": round(float(low), 3), "p95": round(float(high), 3)}

    return {
        "cases": len(paired), "all_cases": len(groups), "windows": sum(quota for _, _, quota in paired),
        "fixed": interval(sizes[:, 0]), "mixed": interval(sizes[:, 1]), "combined": interval(sizes[:, 2]),
        "combined_minus_fixed": interval(sizes[:, 2] - sizes[:, 0]),
    }


def _scope_report(frames, scope, trials, seed):
    scoped = [frame for frame in frames if scope == "all" or frame["step"] > 0]
    fixed = {frame["point"] for frame in scoped if frame["behavior"] == FIXED}
    mixed = {frame["point"] for frame in scoped if frame["behavior"] == MIXED}
    sizes = Counter(frame["behavior"] for frame in scoped)
    return {
        "fixed_windows": sizes[FIXED], "mixed_windows": sizes[MIXED],
        "fixed_profiles": len(fixed), "mixed_profiles": len(mixed), "shared": len(fixed & mixed),
        "fixed_only": len(fixed - mixed), "mixed_only": len(mixed - fixed), "union": len(fixed | mixed),
        "matched": _matched_coverage(scoped, trials, seed),
    }


def _example(frames, points):
    candidates = []
    for index, frame in enumerate(frames):
        if frame["behavior"] != MIXED or not frame["step"] or frame["paired"] is None:
            continue
        other = frames[frame["paired"]]
        point = points[frame["point"]]
        counterpart = points[other["point"]]
        same_size = point["alive"] == counterpart["alive"]
        both_unique = point["sources"] == [MIXED] and counterpart["sources"] == [FIXED]
        if same_size and both_unique:
            candidates.append((abs(len(frame["reach"]) - len(other["reach"])), -frame["step"], index))
    return max(candidates)[-1] if candidates else 0


def _scene_report(name, rows, trials, seed):
    frames, signatures = _frames(rows)
    types = sorted(set(signatures))
    type_to_id = {signature: index for index, signature in enumerate(types)}
    members = defaultdict(list)
    xy, perplexity = _project(types, seed)
    for index, (signature, frame) in enumerate(zip(signatures, frames)):
        frame["point"] = type_to_id[signature]
        members[frame["point"]].append(index)
    points = [{
        "id": f"{name}-{index + 1:03d}", "xy": xy[index], "alive": list(signature[:2]),
        "histogram": [[label, count] for label, count in enumerate(signature[2:]) if count],
        "frames": members[index], "sources": sorted({frames[i]["behavior"] for i in members[index]}),
    } for index, signature in enumerate(types)]
    units = np.array([unit[:2] for frame in frames for side in frame["units"] for unit in side])
    return {
        "points": points, "frames": frames, "perplexity": perplexity,
        "bounds": [[floor(value) - 1 for value in units.min(axis=0)],
                   [ceil(value) + 1 for value in units.max(axis=0)]],
        "scopes": {scope: _scope_report(frames, scope, trials, seed) for scope in SCOPES},
        "example": _example(frames, points),
    }


def _write_tables(output, report):
    with (output / "profile-points.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scenario", "profile", "t_sne_1", "t_sne_2", "friendly_alive", "enemy_alive",
                         "fixed_all", "mixed_all", "fixed_later", "mixed_later", "relation_histogram"))
        for name, scene in report["scenarios"].items():
            for point in scene["points"]:
                frames = [scene["frames"][index] for index in point["frames"]]
                counts = Counter(frame["behavior"] for frame in frames)
                later = Counter(frame["behavior"] for frame in frames if frame["step"] > 0)
                writer.writerow((name, point["id"], *point["xy"], *point["alive"],
                                 counts[FIXED], counts[MIXED], later[FIXED], later[MIXED],
                                 json.dumps(dict(point["histogram"]))))
    with (output / "coverage-summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        first = next(iter(report["scenarios"].values()))["scopes"]["all"]
        fields = [key for key in first if key != "matched"]
        conditions = ("fixed", "mixed", "combined", "combined_minus_fixed")
        intervals = ("mean", "p05", "p95")
        writer.writerow(("scenario", "scope", *fields, "paired_cases", "matched_windows", "repetitions",
                         *(f"{condition}_{stat}" for condition in conditions for stat in intervals)))
        for name, scene in report["scenarios"].items():
            for scope, stats in scene["scopes"].items():
                matched = stats["matched"] or {}
                writer.writerow((name, scope, *(stats[key] for key in fields), matched.get("cases"),
                                 matched.get("windows"), report["draws"],
                                 *(matched.get(c, {}).get(s) for c in conditions for s in intervals)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1201)
    args = parser.parse_args()
    if args.draws < 1 or not 0 <= args.seed < 2**32:
        parser.error("Use at least one draw and a nonnegative 32-bit seed.")
    manifest, scenes = _archive(args.probes)
    with threadpool_limits(limits=1):
        scene_reports = {name: _scene_report(name, rows, args.draws, args.seed) for name, rows in sorted(scenes.items())}
    report = {
        "draws": args.draws, "seed": args.seed, "root_seed": manifest["root_seed"],
        "window_steps": manifest["horizon"], "probe_sha256": manifest["probes_sha256"],
        "scenarios": scene_reports,
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    _write_tables(output, report)
    write_json(output / "coverage-summary.json", {
        **{key: value for key, value in report.items() if key != "scenarios"},
        "scenarios": {name: {**scene["scopes"], "perplexity": scene["perplexity"]}
                      for name, scene in scene_reports.items()},
    })
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    (output / "probe-coverage.html").write_text(template.replace("__COVERAGE_DATA__", payload), encoding="utf-8")
    print(f"Coverage report written to {output}")


if __name__ == "__main__":
    main()

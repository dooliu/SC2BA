"""Run paired train and frozen evaluation cases in SC2BA."""

from collections import Counter, defaultdict
import json

import numpy as np

from .arena import Arena
from .learner import HierarchicalPolicy
from .plan import cases, plan_digest, scenario_seeds
from .records import json_line, write_json
from .upper_state.encoder import UpperEncoder


def _run_episode(arena, policy, layout, opponent, episode_seed, train):
    rng = np.random.default_rng(episode_seed)
    first = observation = arena.reset_round(layout, opponent)
    policy.begin_episode(train)
    visited, upper_unseen, lower_unseen = set(), 0, 0
    while True:
        commands, (key, upper, lower) = policy.decide(observation, rng)
        visited.add(key)
        upper_unseen += upper
        lower_unseen += lower
        observation, done, info = arena.step_macros(commands)
        if done:
            if train:
                policy.decide(observation, rng, terminal=True)
            break
    audit = arena.episode_details(first, observation, info)
    return {**audit, "visited_upper_keys": len(visited), "upper_unseen_steps": upper_unseen,
            "lower_unseen_steps": lower_unseen}


def run_policy(args, document, scenarios):
    train = args.command == "train"
    split = "train" if train else args.split
    record = None if train else json.loads(args.policy.read_text(encoding="utf-8"))
    kind = args.method if train else record["method"]
    encoder = UpperEncoder(kind, args.encoder)
    policy = HierarchicalPolicy(encoder)
    plan_sha = plan_digest(document)
    identity = {"format": "bch-sc2ba-policy-v1", "method": kind, "encoder_sha256": encoder.fingerprint,
                "plan_sha256": plan_sha, "step_mul": args.step_mul}
    if encoder.fingerprint and (encoder.fit_manifest["plan_sha256"], encoder.fit_manifest["step_mul"]) != (
        plan_sha, args.step_mul,
    ):
        raise ValueError("Encoder and experiment plan or step multiplier do not match.")
    if record is not None:
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("Evaluation policy, encoder and experiment plan must match.")
        policy.restore(record["q"])
    initial_q = policy.snapshot() if not train else None
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    counts = defaultdict(Counter)
    steps = 0
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for scenario in scenarios:
            engine_seed, order_rng, play_rng = scenario_seeds(args.seed, document, scenario.name)
            with Arena(scenario, engine_seed, step_mul=args.step_mul) as arena:
                for repeat, layout, opponent in cases(scenario, args.repeats, order_rng, split):
                    episode_seed = int(play_rng.integers(2**32))
                    audit = _run_episode(arena, policy, layout, opponent, episode_seed, train)
                    steps += audit["steps"]
                    counts[(scenario.name, opponent)][audit["outcome"]] += 1
                    stream.write(json_line({
                        "scenario": scenario.name, "layout": layout.id, "opponent": opponent, "split": split,
                        "repeat": repeat, "engine_seed": engine_seed, "episode_seed": episode_seed, **audit,
                    }))
                    stream.flush()
    if train:
        write_json(out / "policy.json", {**identity, "train_scenarios": [s.name for s in scenarios],
                                        "train_seed": args.seed, "q": policy.snapshot()})
    elif initial_q != policy.snapshot():
        raise RuntimeError("Frozen evaluation modified a Q table.")
    write_json(out / "report.json", {
        **identity, "phase": args.command, "split": split, "seed": args.seed, "total_steps": steps,
        "upper_q_rows": len(policy.upper.rows), "lower_q_rows": sum(len(table.rows) for table in policy.lower.values()),
        "opponents": [{"scenario": scenario, "opponent": opponent, "episodes": sum(result.values()),
                       "win_rate": result["win"] / sum(result.values()), "outcomes": result}
                      for (scenario, opponent), result in sorted(counts.items())],
    })
    print(f"{args.command} {kind}: {steps} decisions, {len(policy.upper.rows)} upper Q rows. {out}", flush=True)

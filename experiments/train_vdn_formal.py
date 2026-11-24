"""Train VDN in parallel SC2BA games and reserve the final test for convergence."""

import argparse
import copy
import gzip
import json
import math
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.util import Finalize
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_vdn_arena import (
    OPPONENTS,
    VDN,
    aggregate,
    append_json,
    play_episode,
    save_json,
    write_comparison,
)
from sc2ba.env import Agent, Bot, Camp, StarCraft2BAEnv, difficulties


BATCH_SIZE = 32
REPLAY_SIZE = 5_000
TARGET_UPDATE = 200
WARMUP_STEPS = 5_000
INITIAL_LR = 1e-4
FINAL_LR = 5e-6
PLATEAU_CHECKS = 5
WIN_RATE_SPREAD = 0.05
RETURN_SPREAD = 1.0
OPPONENT_SPREAD = 0.18


def create_env(seed):
    players = [Agent("T", Camp.RED.name), Bot("T", Camp.BLUE.name, difficulties["7"])]
    return StarCraft2BAEnv(
        map_name="3m", players=players, random_enemy=True, seed=seed,
        window_size_x=320, window_size_y=240,
    )


class BatchedVDN(VDN):
    def __init__(self, info, device):
        torch.backends.cudnn.allow_tf32 = False
        super().__init__(info, device=device, batch_size=BATCH_SIZE, replay_size=REPLAY_SIZE,
                         learning_rate=INITIAL_LR, target_update=TARGET_UPDATE)
        self.unrolls = {net: self.batch_gru(net.rnn) for net in (self.net, self.target)}

    @staticmethod
    def batch_gru(cell):
        gru = nn.GRU(cell.input_size, cell.hidden_size, batch_first=True).to(cell.weight_ih.device)
        # The batched torch GRU and SC2BA's GRUCell use the same gate order.
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            setattr(gru, name + "_l0", getattr(cell, name))
        gru.flatten_parameters()
        return gru

    def q_sequence(self, net, inputs):
        batch_size, length, units, _ = inputs.shape
        features = F.relu(net.fc1(inputs)).transpose(1, 2).flatten(0, 1)
        hidden, _ = self.unrolls[net](features)
        return net.fc2(hidden).reshape(batch_size, units, length, self.actions).transpose(1, 2)


def init_actor(seed):
    global actor_env, actor_policy, actor_seed, actor_cleanup
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("A sampling worker unexpectedly sees the training GPU")
    actor_seed = seed
    actor_env = create_env(seed)
    actor_policy = VDN(actor_env.get_env_info())
    actor_cleanup = Finalize(None, close_actor, exitpriority=20)


def close_actor():
    actor_env.close()


def collect_batch(weights, games, seed, epsilon, mode):
    global actor_env
    rng = np.random.default_rng(seed)
    actor_policy.net.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
    for opponent, spawn in games:
        try:
            result = play_episode(actor_env, actor_policy, actor_policy.net, opponent, spawn,
                                  epsilon, rng, capture=mode == "test")
        except RuntimeError as error:
            if "SC2BA did not report a regular environment step" not in str(error):
                raise
            print(f"Actor {os.getpid()} restarting one failed SC2 episode", flush=True)
            actor_env.close()
            actor_env = create_env(actor_seed)
            result = play_episode(actor_env, actor_policy, actor_policy.net, opponent, spawn,
                                  epsilon, rng, capture=mode == "test")
        summary, trace, experience = result
        yield summary, trace, experience if mode == "train" else None


def collect_actor_job(weights, games, seed, epsilon, mode):
    return list(collect_batch(weights, games, seed, epsilon, mode))


def weight_snapshot(model):
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def collect(pool, model, games, rng, epsilon, mode, chunk_size):
    weights = weight_snapshot(model)
    futures = [
        pool.submit(collect_actor_job, weights, games[start:start + chunk_size],
                    int(rng.integers(2 ** 31)), epsilon, mode)
        for start in range(0, len(games), chunk_size)
    ]
    for future in futures:
        yield from future.result()


def create_layouts(args):
    all_layouts = [(x / 8, y / 8) for x in range(64, 93) for y in range(100, 157)]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_layouts)
    # Reserve unseen layouts for formal evaluation. The pilot used a half-unit grid.
    new_layouts = [point for point in all_layouts if any(coord % 0.5 for coord in point)]
    validation = new_layouts[:args.val_rounds]
    test = new_layouts[args.val_rounds:args.val_rounds + args.test_rounds]
    reserved = set(validation + test)
    training = [point for point in all_layouts if point not in reserved]
    return training, validation, test


def evaluation_games(layouts, rng):
    games = [(opponent, spawn) for spawn in layouts for opponent in OPPONENTS]
    rng.shuffle(games)
    return games


def summarize_validation(rows, steps):
    return {
        "training_steps": steps, **aggregate(rows),
        "opponent_win_rates": {
            name: aggregate([row for row in rows if row["opponent"] == name])["win_rate"]
            for name, _ in OPPONENTS
        },
    }


def did_converge(checks, min_steps):
    if len(checks) < PLATEAU_CHECKS or checks[-1]["training_steps"] < min_steps:
        return False
    windows = checks[-PLATEAU_CHECKS:]
    win_rates = [row["win_rate"] for row in windows]
    rewards = [row["mean_return"] for row in windows]
    opponent_series = zip(*(row["opponent_win_rates"].values() for row in windows))
    return (
        max(win_rates) - min(win_rates) <= WIN_RATE_SPREAD
        and max(rewards) - min(rewards) <= RETURN_SPREAD
        and all(max(series) - min(series) <= OPPONENT_SPREAD for series in opponent_series)
    )


def learning_rate(steps, min_steps):
    progress = min(steps / min_steps, 1.0)
    return FINAL_LR + (INITIAL_LR - FINAL_LR) * (0.5 + 0.5 * math.cos(math.pi * progress))


def atomic_json(path, data):
    pending = path.with_suffix(".tmp")
    save_json(pending, data)
    pending.replace(path)


def save_checkpoint(path, learner, progress, rng, episode_log):
    pending = path.with_suffix(".tmp")
    episode_log.flush()
    torch.save({
        "progress": progress, "rng": rng.bit_generator.state, "log_offset": episode_log.tell(),
        "policy": learner.net.state_dict(), "target": learner.target.state_dict(),
        "optimizer": learner.optimizer.state_dict(), "replay": list(learner.replay),
        "updates": learner.updates,
    }, pending)
    pending.replace(path)


def restore_checkpoint(path, learner, rng):
    state = torch.load(path, map_location="cpu", weights_only=False)
    learner.net.load_state_dict(state["policy"])
    learner.target.load_state_dict(state["target"])
    learner.optimizer.load_state_dict(state["optimizer"])
    learner.replay = deque(state["replay"], maxlen=REPLAY_SIZE)
    learner.updates = state["updates"]
    rng.bit_generator.state = state["rng"]
    return state["progress"], state["log_offset"]


def evaluate_final(args, pool, original, learner, layouts, metrics):
    rng = np.random.default_rng(args.seed + 1)
    games = evaluation_games(layouts, rng)
    results = {}
    with gzip.open(args.output / "test_trajectories.jsonl.gz", "wt", encoding="utf-8") as traces:
        for phase, model in (("original", original), ("trained", learner.net)):
            print(f"Final test: {phase} VDN, {len(games)} games", flush=True)
            rows = []
            for summary, trace, _ in collect(pool, model, games, rng, 0.0, "test", 9):
                append_json(traces, {"phase": phase, **trace})
                rows.append(summary)
            results[phase] = rows
    comparison = write_comparison(args.output / "matchups.csv", results["original"], results["trained"])
    atomic_json(args.output / "result.json", {
        **metrics, "original": aggregate(results["original"]), "trained": aggregate(results["trained"]),
        "comparisons": comparison,
    })


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_layouts, validation_layouts, test_layouts = create_layouts(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA training device is unavailable")
    learner = BatchedVDN(create_env(args.seed).get_env_info(), device)
    original = copy.deepcopy(learner.net).requires_grad_(False)
    checkpoint = args.output / "resume.pt"
    episode_path = args.output / "training_episodes.jsonl"
    progress = {"training_steps": 0, "episodes": 0, "rounds": 0, "validation": [], "converged": False}
    if args.resume:
        progress, log_offset = restore_checkpoint(checkpoint, learner, rng)
        with episode_path.open("r+b") as file:
            file.truncate(log_offset)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save_json(args.output / "config.json", {
            **vars(args), "output": str(args.output), "opponents": OPPONENTS,
            "validation_layouts": validation_layouts, "test_layouts": test_layouts,
            "training_layout_count": len(train_layouts), "batch_size": BATCH_SIZE,
            "replay_size": REPLAY_SIZE, "target_update": TARGET_UPDATE,
            "warmup_steps": WARMUP_STEPS, "initial_lr": INITIAL_LR, "final_lr": FINAL_LR,
            "plateau": {"checks": PLATEAU_CHECKS, "win_rate_spread": WIN_RATE_SPREAD,
                        "return_spread": RETURN_SPREAD, "opponent_win_rate_spread": OPPONENT_SPREAD},
        })
    print(f"Formal VDN: {args.actors} SC2 actors, {device}, output={args.output}", flush=True)
    next_validation = (progress["training_steps"] // args.val_interval + 1) * args.val_interval
    next_save = (progress["training_steps"] // args.save_interval + 1) * args.save_interval
    started = time.monotonic()
    initial_steps = progress["training_steps"]
    workers = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.actors, mp_context=workers, initializer=init_actor,
                             initargs=(args.seed,)) as pool, episode_path.open("a", encoding="utf-8") as log:
        while progress["training_steps"] < args.max_steps and not progress["converged"]:
            steps = progress["training_steps"]
            games = [
                (opponent, train_layouts[int(rng.integers(len(train_layouts)))])
                for _ in range(args.actors * args.rounds_per_actor) for opponent in OPPONENTS
            ]
            rng.shuffle(games)
            epsilon = max(0.05, 0.25 - 0.20 * steps / 250_000)
            chunk_size = len(OPPONENTS) * args.rounds_per_actor
            for row, _, sample in collect(pool, learner.net, games, rng, epsilon, "train", chunk_size):
                learner.replay.append(sample)
                progress["training_steps"] += row["steps"]
                progress["episodes"] += 1
                log.write(json.dumps({"training_steps": progress["training_steps"], **row}) + "\n")
            progress["rounds"] += 1
            steps = progress["training_steps"]
            lr = learning_rate(steps, args.min_steps)
            learner.optimizer.param_groups[0]["lr"] = lr
            if steps >= WARMUP_STEPS and len(learner.replay) >= learner.batch_size:
                losses = [learner.update(rng) for _ in range(len(games) // 2)]
                if not np.isfinite(losses).all():
                    raise RuntimeError("Non-finite training loss")
                progress["last_td_loss"] = float(np.mean(losses))
            needs_validation = steps >= min(next_validation, args.max_steps)
            if needs_validation:
                games = evaluation_games(validation_layouts, np.random.default_rng(args.seed + 2))
                rows = [row for row, _, _ in collect(pool, learner.net, games, rng, 0.0, "validation", 18)]
                check = summarize_validation(rows, steps)
                progress["validation"].append(check)
                progress["converged"] = did_converge(progress["validation"], args.min_steps)
                atomic_json(args.output / "validation.json", progress["validation"])
                print(f"Validation {steps:,}: win={check['win_rate']:.3%}, "
                      f"return={check['mean_return']:.3f}, converged={progress['converged']}", flush=True)
                next_validation = (steps // args.val_interval + 1) * args.val_interval
            if needs_validation or steps >= next_save:
                save_checkpoint(checkpoint, learner, progress, rng, log)
                next_save = (steps // args.save_interval + 1) * args.save_interval
            throughput = (steps - initial_steps) / (time.monotonic() - started)
            atomic_json(args.output / "status.json", {
                **progress, "updates": learner.updates, "steps_per_second": throughput,
                "learning_rate": lr, "epsilon": epsilon, "state": "training",
            })
            print(f"Train {steps:,}/{args.min_steps:,} steps, {progress['episodes']:,} episodes, "
                  f"{learner.updates:,} updates, {throughput:.0f} steps/s", flush=True)
        metrics = {**progress, "optimizer_updates": learner.updates}
        if progress["converged"]:
            torch.save(learner.net.state_dict(), args.output / "agent.th")
            atomic_json(args.output / "status.json", {**metrics, "state": "testing"})
            evaluate_final(args, pool, original, learner, test_layouts, metrics)
            atomic_json(args.output / "status.json", {**metrics, "state": "complete"})
        else:
            atomic_json(args.output / "status.json", {**metrics, "state": "budget_reached"})
            print("Training budget reached without meeting the convergence criterion.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26112401)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actors", type=int, default=8)
    parser.add_argument("--rounds-per-actor", type=int, default=2)
    parser.add_argument("--min-steps", type=int, default=2_000_000)
    parser.add_argument("--max-steps", type=int, default=3_000_000)
    parser.add_argument("--val-interval", type=int, default=100_000)
    parser.add_argument("--save-interval", type=int, default=25_000)
    parser.add_argument("--val-rounds", type=int, default=72)
    parser.add_argument("--test-rounds", type=int, default=144)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.actors, args.rounds_per_actor, args.min_steps, args.val_interval,
           args.save_interval, args.val_rounds, args.test_rounds) < 1:
        parser.error("counts and intervals must be positive")
    if args.max_steps < args.min_steps:
        parser.error("max-steps must be at least min-steps")
    if args.val_rounds + args.test_rounds > 1533:
        parser.error("validation and test must fit on the reserved new layout grid")
    train(args)

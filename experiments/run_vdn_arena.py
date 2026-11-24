"""Fine-tune the bundled 3m VDN policy against the SC2BA opponent pool."""

import argparse
import copy
import csv
import gzip
import io
import json
import random
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

import sc2ba
from sc2ba.enemycontrol.enemy_control import enemy_name
from sc2ba.enemycontrol.modules.agents.rnn_agent import RNNAgent
from sc2ba.enemycontrol.modules.mixers.vdn import VDNMixer
from sc2ba.env import Agent, Bot, Camp, StarCraft2BAEnv, difficulties
from types import SimpleNamespace


OPPONENTS = tuple(enemy_name.items())
ROOT = Path(sc2ba.__file__).parent / "enemycontrol"
WARM_START = ROOT / "models" / "3m" / "vdn" / "agent.th"
BATCH_SIZE = 12
WARMUP = 18
GAMMA = 0.99
LEARNING_RATE = 1e-4
TARGET_UPDATE = 40
WIN_RATE_SPREAD = 0.12
RETURN_SPREAD = 2.0


class VDN:
    def __init__(self, info, *, device="cpu", batch_size=BATCH_SIZE, replay_size=512,
                 learning_rate=LEARNING_RATE, target_update=TARGET_UPDATE):
        with (ROOT / "config" / "default.yaml").open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.agents = info["n_agents"]
        self.actions = info["n_red_actions"]
        self.obs_size = info["obs_red_shape"]
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.target_update = target_update
        self.ids = torch.eye(self.agents)
        self.action_vectors = torch.eye(self.actions)
        self.net = RNNAgent(
            self.obs_size + self.actions + self.agents,
            SimpleNamespace(rnn_hidden_dim=config["rnn_hidden_dim"], n_actions=self.actions),
        )
        self.net.load_state_dict(torch.load(WARM_START, map_location="cpu", weights_only=True))
        self.net.to(self.device)
        self.target = copy.deepcopy(self.net).requires_grad_(False)
        self.mixer = VDNMixer()
        self.optimizer = torch.optim.RMSprop(
            self.net.parameters(), lr=learning_rate, alpha=config["optim_alpha"],
            eps=config["optim_eps"],
        )
        self.replay = deque(maxlen=replay_size)
        self.updates = 0

    @torch.no_grad()
    def choose(self, net, obs, available, last_actions, hidden, epsilon, rng):
        inputs = torch.cat((torch.from_numpy(obs), last_actions, self.ids), dim=-1).to(self.device)
        values, hidden = net(inputs, hidden)
        available_mask = torch.from_numpy(available).to(self.device)
        actions = values.masked_fill(~available_mask, -torch.inf).argmax(-1).tolist()
        if epsilon > 0:
            for unit in range(self.agents):
                if rng.random() < epsilon:
                    actions[unit] = int(rng.choice(np.flatnonzero(available[unit])))
        return actions, self.action_vectors[actions], hidden

    def q_sequence(self, net, inputs):
        batch_size, _, units, _ = inputs.shape
        hidden = net.init_hidden().expand(batch_size * units, -1)
        values = []
        for frame in inputs.unbind(dim=1):
            q, hidden = net(frame.flatten(0, 1), hidden)
            values.append(q.reshape(batch_size, units, self.actions))
        return torch.stack(values, dim=1)

    def update(self, rng):
        indices = rng.choice(len(self.replay), size=self.batch_size, replace=False)
        episodes = [self.replay[int(index)] for index in indices]
        length = max(len(episode[2]) for episode in episodes)
        obs = torch.zeros(self.batch_size, length + 1, self.agents, self.obs_size)
        available = torch.zeros(self.batch_size, length + 1, self.agents, self.actions, dtype=torch.bool)
        actions = torch.zeros(self.batch_size, length, self.agents, dtype=torch.long)
        rewards = torch.zeros(self.batch_size, length, 1)
        terminated = torch.zeros_like(rewards, dtype=torch.bool)
        filled = torch.zeros_like(rewards, dtype=torch.bool)
        for index, (ep_obs, ep_available, ep_actions, ep_rewards) in enumerate(episodes):
            steps = len(ep_actions)
            obs[index, :steps + 1] = torch.from_numpy(ep_obs)
            available[index, :steps + 1] = torch.from_numpy(ep_available)
            actions[index, :steps] = torch.from_numpy(ep_actions)
            rewards[index, :steps, 0] = torch.from_numpy(ep_rewards)
            terminated[index, steps - 1] = True
            filled[index, :steps] = True
        last_actions = torch.zeros_like(available, dtype=torch.float32)
        last_actions[:, 1:] = self.action_vectors[actions] * filled.unsqueeze(-1)
        ids = self.ids.expand(self.batch_size, length + 1, -1, -1)
        inputs = torch.cat((obs, last_actions, ids), dim=-1).to(self.device)
        available, actions, rewards, terminated, filled = (
            tensor.to(self.device) for tensor in (available, actions, rewards, terminated, filled)
        )
        online = self.q_sequence(self.net, inputs)
        chosen = online[:, :-1].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        values = self.mixer(chosen, None)
        with torch.no_grad():
            target = self.q_sequence(self.target, inputs)[:, 1:]
            greedy = online[:, 1:].masked_fill(~available[:, 1:], -torch.inf).argmax(-1, keepdim=True)
            bootstrap = self.mixer(target.gather(-1, greedy).squeeze(-1), None)
            targets = rewards + GAMMA * bootstrap * ~terminated
        loss = F.mse_loss(values[filled], targets[filled])
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())
        return float(loss)


def frame(env, camp):
    return (
        np.asarray(env.get_obs(camp), dtype=np.float32),
        np.asarray(env.get_state(camp), dtype=np.float32),
        np.asarray(env.get_avail_actions(camp), dtype=np.bool_),
    )


def serial_frame(data):
    return {name: value.tolist() for name, value in zip(("obs", "state", "avail"), data)}


def play_episode(env, learner, net, opponent, spawn, epsilon, rng, *, capture=True):
    name, opponent_id = opponent
    env.random_enemy_seq = [opponent_id]
    env.red_start_position = spawn
    # PySC2 and the bundled loader print the complete configuration at launch.
    with redirect_stdout(io.StringIO()):
        env.reset()
    assert env.enemy_code == opponent_id
    hidden = net.init_hidden().expand(learner.agents, -1)
    last_actions = torch.zeros(learner.agents, learner.actions)
    red = frame(env, Camp.RED)
    blue = frame(env, Camp.BLUE) if capture else None
    initial_units = [
        [[float(unit.pos.x), float(unit.pos.y)] for unit in side.values()]
        for side in (env.agents, env.enemies)
    ] if capture else None
    red_obs, red_avail, red_actions, red_rewards = [], [], [], []
    blue_return = 0.0
    trajectory = []
    for _ in range(env.episode_limit):
        actions, last_actions, hidden = learner.choose(
            net, red[0], red[2], last_actions, hidden, epsilon, rng,
        )
        with torch.no_grad():
            rewards, terminated, info = env.step(actions)
        if "dead_red" not in info or "dead_blue" not in info:
            raise RuntimeError("SC2BA did not report a regular environment step")
        if capture:
            blue_actions = env.blue_last_action.argmax(axis=1).tolist() if opponent_id else None
            trajectory.append({
                "red": {**serial_frame(red), "actions": actions, "reward": float(rewards[0])},
                "blue": {**serial_frame(blue), "actions": blue_actions, "reward": float(rewards[1])},
                "terminated": bool(terminated),
            })
        red_obs.append(red[0])
        red_avail.append(red[2])
        red_actions.append(actions)
        red_rewards.append(rewards[0])
        blue_return += rewards[1]
        red = frame(env, Camp.RED)
        if capture:
            blue = frame(env, Camp.BLUE)
        if terminated:
            break
    else:
        raise RuntimeError("Episode exceeded the scenario limit")
    outcome = "win" if info["red_battle_won"] else "loss" if info["blue_battle_won"] else "draw"
    summary = {
        "opponent": name, "red_spawn": spawn, "outcome": outcome, "steps": len(red_actions),
        "red_return": float(sum(red_rewards)), "blue_return": float(blue_return),
    }
    record = None
    if capture:
        record = {
            **summary, "initial_units": initial_units, "trajectory": trajectory,
            "final": {"red": serial_frame(red), "blue": serial_frame(blue)},
        }
    experience = (
        np.stack(red_obs + [red[0]]), np.stack(red_avail + [red[2]]),
        np.asarray(red_actions, dtype=np.int64), np.asarray(red_rewards, dtype=np.float32),
    )
    return summary, record, experience


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_json(file, data):
    file.write(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    file.flush()


def evaluation(env, learner, net, layouts, rng, log_episode, phase, iteration):
    rows = []
    for opponent in OPPONENTS:
        for spawn in layouts:
            summary, record, _ = play_episode(env, learner, net, opponent, spawn, 0.0, rng)
            log_episode(phase, iteration, summary, record)
            rows.append(summary)
    return rows


def aggregate(rows):
    return {
        "win_rate": sum(row["outcome"] == "win" for row in rows) / len(rows),
        "mean_return": float(np.mean([row["red_return"] for row in rows])),
        "episodes": len(rows),
    }


def stable(checks):
    return len(checks) >= 3 and all(
        max(check[key] for check in checks[-3:]) - min(check[key] for check in checks[-3:]) <= spread
        for key, spread in (("win_rate", WIN_RATE_SPREAD), ("mean_return", RETURN_SPREAD))
    )


def write_comparison(path, original, trained):
    rows = []
    for name, _ in OPPONENTS:
        row = {"opponent": name}
        for label, episodes in (("original", original), ("trained", trained)):
            matches = [match for match in episodes if match["opponent"] == name]
            row.update({f"{label}_{key}": value for key, value in aggregate(matches).items()})
            for outcome in ("win", "loss", "draw"):
                row[f"{label}_{outcome}"] = sum(match["outcome"] == outcome for match in matches)
        row["win_rate_delta_pp"] = 100 * (row["trained_win_rate"] - row["original_win_rate"])
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    layouts = [(x / 2, y / 2) for x in range(16, 24) for y in range(25, 40)]
    rng.shuffle(layouts)
    val_layouts = layouts[:args.validation_rounds]
    test_layouts = layouts[args.validation_rounds:args.validation_rounds + args.test_rounds]
    train_layouts = layouts[args.validation_rounds + args.test_rounds:]
    players = [Agent("T", Camp.RED.name), Bot("T", Camp.BLUE.name, difficulties["7"])]
    env = StarCraft2BAEnv(map_name="3m", players=players, random_enemy=True, seed=args.seed)
    learner = VDN(env.get_env_info())
    original = copy.deepcopy(learner.net).requires_grad_(False)
    save_json(args.output / "config.json", {
        **vars(args), "output": str(args.output), "algo": "VDN", "warm_start": str(WARM_START),
        "scenario": "3m mirrored generated spawns", "opponents": OPPONENTS,
        "validation_layouts": val_layouts, "test_layouts": test_layouts,
        "train_layouts": train_layouts, "lr": LEARNING_RATE, "gamma": GAMMA,
        "batch_size": BATCH_SIZE, "warmup_episodes": WARMUP, "target_update": TARGET_UPDATE,
        "stability": {"consecutive_checks": 3, "win_rate_spread": WIN_RATE_SPREAD,
                      "return_spread": RETURN_SPREAD},
    })
    print(f"Start VDN, seed={args.seed}, {len(OPPONENTS)} opponents, output={args.output}", flush=True)
    checks = []
    try:
        with (args.output / "episodes.jsonl").open("w", encoding="utf-8") as summary_file, \
                gzip.open(args.output / "trajectories.jsonl.gz", "wt", encoding="utf-8") as trace_file:
            def log_episode(phase, iteration, summary, record):
                labels = {"phase": phase, "training_episode": iteration}
                append_json(summary_file, {**labels, **summary})
                append_json(trace_file, {**labels, **record})

            opponent_order = list(OPPONENTS)
            for episode in range(1, args.train_episodes + 1):
                if (episode - 1) % len(opponent_order) == 0:
                    rng.shuffle(opponent_order)
                opponent = opponent_order[(episode - 1) % len(opponent_order)]
                spawn = train_layouts[int(rng.integers(len(train_layouts)))]
                epsilon = max(0.04, 0.15 - 0.11 * episode / 180)
                summary, record, experience = play_episode(
                    env, learner, learner.net, opponent, spawn, epsilon, rng,
                )
                learner.replay.append(experience)
                if len(learner.replay) >= WARMUP:
                    summary["loss"] = learner.update(rng)
                log_episode("train", episode, summary, record)
                if episode % len(OPPONENTS) == 0:
                    print(f"Train {episode}/{args.train_episodes}, updates={learner.updates}", flush=True)
                if episode % args.check_every == 0 or episode == args.train_episodes:
                    rows = evaluation(env, learner, learner.net, val_layouts, rng, log_episode,
                                      "validation", episode)
                    check = {"training_episode": episode, **aggregate(rows)}
                    checks.append(check)
                    print(f"Validation {check} stable={stable(checks)}", flush=True)
                    save_json(args.output / "validation.json", checks)
                    torch.save(learner.net.state_dict(), args.output / "agent.th")
                    if stable(checks):
                        break
            print("Testing original VDN", flush=True)
            before = evaluation(env, learner, original, test_layouts, rng, log_episode, "original", episode)
            print("Testing fine-tuned VDN", flush=True)
            after = evaluation(env, learner, learner.net, test_layouts, rng, log_episode, "trained", episode)
        rows = write_comparison(args.output / "matchups.csv", before, after)
        save_json(args.output / "result.json", {
            "training_episodes": episode, "optimizer_updates": learner.updates,
            "stable": stable(checks), "validation": checks, "original": aggregate(before),
            "trained": aggregate(after), "comparisons": rows, "force_restarts": env.force_restarts,
            "game_version": env._controller.ping().game_version,
            "changed_weight_tensors": sum(
                not torch.equal(tensor, original.state_dict()[name])
                for name, tensor in learner.net.state_dict().items()
            ),
        })
        print("Finished training and pairwise evaluation", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--train-episodes", type=int, default=324)
    parser.add_argument("--check-every", type=int, default=36)
    parser.add_argument("--validation-rounds", type=int, default=3)
    parser.add_argument("--test-rounds", type=int, default=12)
    options = parser.parse_args()
    if min(options.train_episodes, options.check_every, options.validation_rounds, options.test_rounds) < 1:
        parser.error("episode and round counts must be positive")
    if options.validation_rounds + options.test_rounds >= 120:
        parser.error("leave at least one generated layout for training")
    run(options)

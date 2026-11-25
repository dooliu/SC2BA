"""Fit encoders and medoids using fixed, layout-disjoint probe folds."""

from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..observation import Observation
from ..records import FOLDS, digest_file, write_json
from .codebook import fit_medoids, hash_features
from .geometry import geometry_target
from .influence import HistogramHash, influence_window
from .models import make_encoder, model_inputs


def _read_probes(path, scenario, behaviors):
    data = path / "probes.jsonl.gz"
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "bch-sc2ba-probes-v1" or manifest["probes_sha256"] != digest_file(data):
        raise ValueError("Probe manifest and data do not match.")
    available = manifest.get("behaviors", ["mixed"])
    behaviors = behaviors or available
    if len(behaviors) != len(set(behaviors)) or not set(behaviors).issubset(available):
        raise ValueError(f"Choose distinct recorded behaviors from {', '.join(available)}.")
    behaviors = set(behaviors)
    expected = {(item["name"], layout["id"]): layout["split"]
                for item in manifest["plan"]["scenarios"] for layout in item["layouts"]}
    observations, cases, folds = [], [], {name: [] for name in FOLDS}
    with gzip.open(data, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            layout = row["scenario"], row["layout"]
            if expected[layout] != row["split"]:
                raise ValueError(f"Layout {layout} is in the wrong split.")
            if (scenario and row["scenario"] != scenario) or row.get("behavior", "mixed") not in behaviors:
                continue
            folds[row["split"]].append(len(cases))
            observations.append(Observation.from_record(row["state"]))
            cases.append({"behavior": row.get("behavior", "mixed"),
                          **{key: row[key] for key in ("episode", "grouping", "macro", "opponent", "hp_change")}})
    if not all(folds.values()):
        raise ValueError("Fit requires samples in all three layout folds.")
    return manifest, observations, cases, {name: np.asarray(ids) for name, ids in folds.items()}


def _candidates(cases, ids, limit, seed):
    if len(ids) <= limit:
        return ids
    by_episode = defaultdict(list)
    for index in np.random.default_rng(seed).permutation(ids):
        by_episode[cases[index]["episode"]].append(index)
    selected = []
    remaining = list(by_episode.values())
    while len(selected) < limit:
        for indices in remaining[:limit - len(selected)]:
            selected.append(indices.pop())
        remaining = [indices for indices in remaining if indices]
    return np.asarray(selected)


def _conditions(cases):
    # The value head sees the chosen grouping/macro and opponent. The encoder sees the current state alone.
    conditions = []
    for field in ("grouping", "macro", "opponent"):
        vocabulary = {name: index for index, name in enumerate(sorted({row[field] for row in cases}))}
        indices = torch.tensor([vocabulary[row[field]] for row in cases])
        conditions.append(F.one_hot(indices, num_classes=len(vocabulary)))
    return torch.cat(conditions, dim=1).float()


def _fit_neural(args, observations, cases, folds, candidate_ids):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train, validation, test = (folds[name] for name in FOLDS)
    model = make_encoder(args.method, args.latent_dim).to(device)
    relation = nn.Linear(args.latent_dim, 66).to(device)
    conditions = _conditions(cases).to(device)
    value_head = nn.Sequential(nn.Linear(args.latent_dim + conditions.shape[1], 128), nn.ReLU(),
                               nn.Linear(128, 1)).to(device)
    optimizer = torch.optim.Adam([*model.parameters(), *relation.parameters(), *value_head.parameters()], lr=1e-3)
    geometry = torch.tensor([geometry_target(obs) for obs in observations], dtype=torch.float32, device=device)
    hp = torch.tensor([row["hp_change"] for row in cases], dtype=torch.float32, device=device)
    g_mean, g_scale = geometry[train].mean(0), geometry[train].std(0, unbiased=False).clamp_min(0.05)
    h_mean, h_scale = hp[train].mean(), hp[train].std(unbiased=False).clamp_min(1)
    geometry, hp = (geometry - g_mean) / g_scale, (hp - h_mean) / h_scale
    rng = np.random.default_rng(args.seed)

    def batches(ids):
        for start in range(0, len(ids), args.batch_size):
            yield ids[start:start + args.batch_size]

    def embed(ids):
        inputs = model_inputs(args.method, [observations[index] for index in ids])
        return model(*(tensor.to(device) for tensor in inputs))

    def predict(ids):
        z = embed(ids)
        return relation(z), value_head(torch.cat((z, conditions[ids]), dim=1)).flatten()

    def loss(ids):
        geo, h = predict(ids)
        return F.mse_loss(geo, geometry[ids]) + F.smooth_l1_loss(h, hp[ids])

    best, best_epoch = float("inf"), 0
    modules = (model, relation, value_head)
    for epoch in range(1, args.epochs + 1):
        for module in modules:
            module.train()
        for ids in batches(rng.permutation(train)):
            optimizer.zero_grad(set_to_none=True)
            loss(ids).backward()
            optimizer.step()
        for module in modules:
            module.eval()
        with torch.inference_mode():
            score = sum(loss(ids).item() * len(ids) for ids in batches(validation)) / len(validation)
        if score < best:
            best, best_epoch = score, epoch
            weights = [deepcopy(module.state_dict()) for module in modules]
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"{args.method} epoch {epoch}/{args.epochs}: validation loss {score:.5f}", flush=True)
    if best_epoch == 0:
        raise RuntimeError("Encoder fit failed to produce a finite validation loss.")
    for module, state in zip(modules, weights):
        module.load_state_dict(state)
    with torch.inference_mode():
        train_z = torch.cat([embed(ids) for ids in batches(train)])
        mean, scale = train_z.mean(0), train_z.std(0, unbiased=False).clamp_min(1e-3)
        features = torch.cat([(embed(ids) - mean) / scale for ids in batches(candidate_ids)]).cpu().numpy()
        geo_total, hp_total = 0.0, 0.0
        for ids in batches(test):
            geo, h = predict(ids)
            geo_total += ((geo - geometry[ids]).abs() * g_scale).mean(dim=1).sum().item()
            hp_total += ((h - hp[ids]).abs() * h_scale).sum().item()
    model_data = {"weights": {key: value.cpu() for key, value in model.state_dict().items()},
                  "mean": mean.cpu(), "scale": scale.cpu()}
    metrics = {"best_epoch": best_epoch, "validation_loss": best, "test_geometry_mae": geo_total / len(test),
               "test_hp_mae": hp_total / len(test)}
    return features, model_data, metrics


def fit(args):
    started = perf_counter()
    probes, observations, cases, folds = _read_probes(args.probes, args.scenario, args.behavior)
    ids = _candidates(cases, folds["train"], args.codebook_samples, args.seed)
    if args.method == "im-medoid":
        render = HistogramHash()
        features = np.stack([hash_features(render(influence_window(observations[index]),
                                                  observations[index].influence_scale)) for index in ids])
        model, metrics, metric = {}, {}, "hamming"
    else:
        features, model, metrics = _fit_neural(args, observations, cases, folds, ids)
        metric = "euclidean"
    medoids, cost = fit_medoids(features, args.codes, metric, args.seed)
    manifest = {
        "probes_sha256": probes["probes_sha256"], "plan_sha256": probes["plan_sha256"],
        "step_mul": probes["step_mul"], "horizon": probes["horizon"], "seed": args.seed, "scenario": args.scenario,
        "behaviors": dict(Counter(row["behavior"] for row in cases)),
        "method": args.method, "codes": args.codes, "latent_dim": args.latent_dim, "epochs": args.epochs,
        "sample_counts": {split: len(ids) for split, ids in folds.items()}, "codebook_samples": len(ids),
        "fold_episodes": {split: len(Counter(cases[index]["episode"] for index in ids)) for split, ids in folds.items()},
        "medoid_cost": cost, "seconds": perf_counter() - started, **metrics,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "format": "bch-sc2ba-encoder-v1", "kind": args.method, "latent_dim": args.latent_dim,
        "medoids": torch.from_numpy(medoids), "manifest": manifest, **model,
    }, args.output / "encoder.pt")
    write_json(args.output / "metrics.json", manifest)
    print(f"{args.method} encoder written to {args.output}", flush=True)

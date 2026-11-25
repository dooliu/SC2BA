"""The previous four-layer IM CNN and permutation-invariant Unit-set encoder."""

import numpy as np
import torch
from torch import nn

from .influence import GRID_SIZE, grid_input


class UnitSetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.units = nn.Sequential(nn.Conv1d(5, 48, 1), nn.ReLU(), nn.Conv1d(48, 64, 1), nn.ReLU())
        self.readout = nn.Sequential(nn.Linear(2 * 2 * 64, 256), nn.ReLU(), nn.Linear(256, latent_dim))

    def forward(self, values, mask):
        batch, sides, units, fields = values.shape
        per_unit = self.units(values.reshape(batch * sides, units, fields).transpose(1, 2))
        masked = per_unit * mask.reshape(batch * sides, 1, units)
        teams = torch.cat((masked.sum(dim=2), masked.amax(dim=2)), dim=1)
        return self.readout(teams.reshape(batch, -1))


def make_encoder(kind, latent_dim):
    if kind == "unit-set":
        return UnitSetEncoder(latent_dim)
    width = GRID_SIZE // 16
    channels = 1 if kind == "im-grid" else 2
    return nn.Sequential(
        nn.Conv2d(channels, 8, 4, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 24, 4, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(24, 32, 4, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(32 * width * width, latent_dim),
    )


def unit_arrays(observations):
    width = max(len(side) for obs in observations for side in (obs.friendly, obs.enemy))
    values = np.zeros((len(observations), 2, max(1, width), 5), dtype=np.float32)
    mask = np.zeros(values.shape[:-1], dtype=bool)
    for row, obs in enumerate(observations):
        armies = (obs.friendly, obs.enemy)
        points = np.array([unit.xy for side in armies for unit in side], dtype=np.float32)
        if not len(points):
            continue
        points = points / (GRID_SIZE / 2) - 1
        center = (points.min(0) + points.max(0)) / 2
        for side, army in enumerate(armies):
            if army:
                team = points[:len(army)] if side == 0 else points[-len(army):]
                values[row, side, :len(army), :2] = team
                values[row, side, :len(army), 3:] = (team - center) * 8
                mask[row, side, :len(army)] = True
    # The unused health channel stays zero, matching the position-only Unit-set comparison.
    return torch.from_numpy(values), torch.from_numpy(mask)


def model_inputs(kind, observations):
    if kind == "unit-set":
        return unit_arrays(observations)
    return (torch.from_numpy(np.stack([grid_input(kind, obs) for obs in observations])),)

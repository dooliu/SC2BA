"""Inference interface shared by the original hash and frozen state encoders."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from .codebook import hash_features
from .influence import KINDS, NEURAL_KINDS, HistogramHash, influence_window
from .models import make_encoder, model_inputs


class UpperEncoder:
    def __init__(self, kind, checkpoint=None):
        if kind not in KINDS or (kind == "im-hash") != (checkpoint is None):
            raise ValueError("Select im-hash alone or supply a checkpoint for a learned encoder.")
        self.kind = kind
        self.fingerprint = None
        self.hash = HistogramHash() if kind in ("im-hash", "im-medoid") else None
        if checkpoint is not None:
            content = Path(checkpoint).read_bytes()
            self.fingerprint = sha256(content).hexdigest()
            data = torch.load(BytesIO(content), weights_only=True, map_location="cpu")
            if data["format"] != "bch-sc2ba-encoder-v1" or data["kind"] != kind:
                raise ValueError("Encoder kind or preprocessing does not match the checkpoint.")
            self.medoids = data["medoids"].numpy()
            self.fit_manifest = data["manifest"]
            if kind in NEURAL_KINDS:
                self.model = make_encoder(kind, data["latent_dim"])
                self.model.load_state_dict(data["weights"])
                self.model.eval()
                self.mean, self.scale = data["mean"], data["scale"]

    @torch.inference_mode()
    def key(self, observation):
        if self.hash is not None:
            value = self.hash(influence_window(observation), observation.influence_scale)
            if self.kind == "im-hash":
                return value
            distances = (self.medoids != hash_features(value)).sum(axis=1)
        else:
            value = ((self.model(*model_inputs(self.kind, [observation]))[0] - self.mean) / self.scale).numpy()
            distances = np.square(self.medoids - value).sum(axis=1)
        return str(int(distances.argmin()))

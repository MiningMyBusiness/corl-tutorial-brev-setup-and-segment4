"""SutureBot ACT (Segment 3's checkpoint) as a plain Python policy, for precomputing comparisons.

Same model loading, image and language inputs, and decoding as segment_3's
OpenHTutorial_SutureBotACT_Inference.ipynb; this file only adds a 16-D (xyz + quat xyzw + jaw)
output so ACT's predictions can be scored like GR00T-H's. Needs the SutureBot repo, ACT branch
(https://github.com/SutureBot/SutureBot/tree/ACT), and its own env (torch, torchvision,
transformers 4.41, av, opencv, scipy); see tests/README.md.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from torchvision.transforms.functional import normalize

CHUNK_SIZE = 60
CAMERAS = ("left", "left_wrist", "right_wrist")
VIDEO_KEYS = ("observation.images.endoscope.left", "observation.images.wrist.left",
              "observation.images.wrist.right")
NORMALIZED_CHANNELS = [0, 1, 2, 9, 10, 11, 12, 19]


# ---- pose conventions (copied from the Segment 3 notebook)
def rotation_columns(matrix):
    return matrix[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)


def rot6_to_matrix(values):
    first, second = values[:, :3], values[:, 3:]
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    second = second - (first * second).sum(axis=1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), axis=2)


def actions_to_20(raw):
    raw = np.asarray(raw, dtype=np.float64)
    arms = []
    for offset in (0, 8):
        arm = raw[:, offset:offset + 8]
        rotation = Rotation.from_quat(arm[:, 3:7]).as_matrix()  # xyzw
        arms.append(np.column_stack((arm[:, :3], rotation_columns(rotation), arm[:, 7])))
    return np.column_stack(arms)


def decode_hybrid(actions, current_state, reanchor_first=False):
    decoded = np.asarray(actions, dtype=np.float64).copy()
    current_state = np.asarray(current_state)
    for output_offset, state_offset in ((0, 0), (10, 8)):
        delta_p = actions[:, output_offset:output_offset + 3]
        if reanchor_first:
            delta_p = delta_p - delta_p[0]
        decoded[:, output_offset:output_offset + 3] = current_state[state_offset:state_offset + 3] + delta_p
        current_R = Rotation.from_quat(current_state[state_offset + 3:state_offset + 7]).as_matrix()
        delta_R = rot6_to_matrix(actions[:, output_offset + 3:output_offset + 9])
        decoded[:, output_offset + 3:output_offset + 9] = rotation_columns(current_R @ delta_R)
    return decoded


def unnormalize_actions(actions, mean, std):
    decoded = np.asarray(actions, dtype=np.float64).copy()
    decoded[:, NORMALIZED_CHANNELS] = decoded[:, NORMALIZED_CHANNELS] * std[NORMALIZED_CHANNELS] + mean[NORMALIZED_CHANNELS]
    return decoded


def to16(a20):
    """(N, 20) absolute [xyz, rot6, jaw] x 2 arms -> (N, 16) [xyz, quat xyzw, jaw] x 2 arms."""
    out = []
    for o in (0, 10):
        q = Rotation.from_matrix(rot6_to_matrix(a20[:, o + 3:o + 9])).as_quat()
        out.append(np.column_stack((a20[:, o:o + 3], q, a20[:, o + 9])))
    return np.column_stack(out).astype(np.float32)


# ---- model
class ACT:
    def __init__(self, ckpt, stats_json, suturebot_dir, device="cuda"):
        self.device = device
        stats = json.loads(Path(stats_json).read_text())
        self.mean, self.std = np.asarray(stats["mean"]), np.asarray(stats["std"])
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
        state = {re.sub(r"^model\.(?:module\.)?", "", k): v for k, v in checkpoint["model_state_dict"].items()}
        layers = lambda p: len({int(k.split(".")[len(p.split("."))]) for k in state if k.startswith(p + ".")})  # noqa: E731
        config = dict(model_type="ACT", action_dim=state["action_head.weight"].shape[0],
                      num_queries=state["query_embed.weight"].shape[0], hidden_dim=state["query_embed.weight"].shape[1],
                      dim_feedforward=state["transformer.encoder.layers.0.linear1.weight"].shape[0],
                      enc_layers=layers("transformer.encoder.layers"), dec_layers=layers("transformer.decoder.layers"),
                      camera_names=list(CAMERAS), backbone="efficientnet_b3film", use_language=True, nheads=8,
                      dropout=0.1, pre_norm=False, dilation=False, masks=False, position_embedding="sine", lr_backbone=1e-5)
        config.update(checkpoint.get("model_config", {}))
        assert config["action_dim"] == 20 and config["num_queries"] == CHUNK_SIZE
        # authors' checkpoint: deployment subtracts the first predicted translation
        self.reanchor = checkpoint.get("action_representation", {}).get("reanchor_first_position", True)
        self.epoch = checkpoint.get("epoch")
        src = str(Path(suturebot_dir) / "src" / "act")
        if src not in sys.path:
            sys.path.insert(0, src)
        from detr.models import backbone
        from detr.models.detr_vae import build
        ctor = backbone.film_efficientnet_b3
        with patch.object(backbone, "film_efficientnet_b3", side_effect=lambda **kw: ctor(**{**kw, "weights": None})):
            model = build(SimpleNamespace(**config))
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()
        self._emb = {}

    def embed(self, command):
        if command not in self._emb:
            from transformers import DistilBertModel, DistilBertTokenizer
            name = "distilbert/distilbert-base-uncased"
            if not hasattr(self, "_tok"):
                self._tok = DistilBertTokenizer.from_pretrained(name)
                self._enc = DistilBertModel.from_pretrained(name).eval()
            with torch.inference_mode():
                tokens = self._tok(command, return_tensors="pt", truncation=True, padding=True, max_length=512)
                self._emb[command] = self._enc(**tokens).last_hidden_state[0, 0].clone()
        return self._emb[command]

    @staticmethod
    def image_tensor(images):
        values = [torch.from_numpy(cv2.resize(im, (480, 360))).permute(2, 0, 1).float() / 255 for im in images]
        return normalize(torch.stack(values), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def chunk(self, images, state16, command):
        """3 RGB images (endoscope left, wrist left, wrist right) + measured 16-D state -> (60, 16) absolute."""
        with torch.inference_mode():
            pred, _, _ = self.model(torch.zeros(1, 20, device=self.device), self.image_tensor(images)[None].to(self.device),
                                    None, command_embedding=self.embed(command)[None].to(self.device))
        model_space = unnormalize_actions(pred[0].cpu().numpy(), self.mean, self.std)
        return to16(decode_hybrid(model_space, state16, reanchor_first=self.reanchor))


# ---- data
class Episode:
    """One SutureBot episode from a local LeRobot root: actions, states, command and decoded frames."""

    def __init__(self, root, ep):
        self.root, self.ep = Path(root), ep
        df = pd.read_parquet(self.root / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet")
        self.actions = np.stack(df["action"].values).astype(np.float64)
        self.states = np.stack(df["observation.state"].values).astype(np.float64)
        self.frame_index = df["frame_index"].values.astype(int)
        cmd = str(df["instruction.text"].iloc[0]).strip()
        self.command = cmd[:-len(" recovery")].strip() if cmd.endswith(" recovery") else cmd

    def frames(self, rows):
        """{row: [3 full-resolution RGB images]} for the given parquet rows, decoding each video once."""
        want = {int(self.frame_index[r]): r for r in rows}
        out = {r: [None] * 3 for r in rows}
        for c, key in enumerate(VIDEO_KEYS):
            path = self.root / f"videos/chunk-{self.ep // 1000:03d}/{key}/episode_{self.ep:06d}.mp4"
            with av.open(str(path)) as f:
                for n, fr in enumerate(f.decode(video=0)):
                    if n in want:
                        out[want[n]][c] = fr.to_ndarray(format="rgb24")
                    if n >= max(want):
                        break
        missing = [r for r, ims in out.items() if any(i is None for i in ims)]
        assert not missing, f"frames missing for rows {missing}"
        return out


def receding_horizon(policy, epi, start, steps, horizon=50):
    """GR00T-H's open-loop protocol: a new chunk on the real observation every `horizon` steps,
    keep its first `horizon` rows. -> (steps, 16) absolute."""
    steps = min(steps, len(epi.actions) - start)
    rows = list(range(start, start + steps, horizon))
    ims = epi.frames(rows)
    pred = np.concatenate([policy.chunk(ims[r], epi.states[r], epi.command)[:horizon] for r in rows])
    return pred[:steps]

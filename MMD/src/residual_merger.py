"""Residual adapter artifacts and Experiment C's token-wise output branch."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

EXPERIMENT_C = "C_post_merger_residual"
ADAPTER_PREFIX = "model.visual.merger.residual_adapter."
ADAPTER_WIDTH = 128
ADAPTER_PARAMETERS = 658_048
ADAPTER_FILE = "residual_adapter.safetensors"


class ResidualMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.down = nn.Linear(width, ADAPTER_WIDTH)
        self.up = nn.Linear(ADAPTER_WIDTH, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens):
        update = self.up(nn.functional.gelu(self.down(tokens.to(self.down.weight.dtype))))
        return tokens + update.to(tokens.dtype)


def base_merger_digest(model):
    """Identify the frozen merger, including when the base was loaded from a local snapshot."""
    merger = model.model.visual.merger
    digest = hashlib.sha256()
    for name, tensor in sorted(merger.state_dict().items()):
        if name.startswith("residual_adapter."):
            continue
        digest.update(name.encode() + b"\0")
        data = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        digest.update(data.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _add_residual(merger, inputs, output):
    return merger.residual_adapter(output)


def attach_residual(model, dtype):
    """Attach only to the final vision output, leaving all DeepStack paths untouched."""
    merger = model.model.visual.merger
    if hasattr(merger, "residual_adapter"):
        raise ValueError("a residual adapter is already attached")
    weight = merger.linear_fc2.weight
    adapter = ResidualMLP(weight.shape[0]).to(device=weight.device, dtype=dtype)
    merger.add_module("residual_adapter", adapter)
    merger.register_forward_hook(_add_residual)
    return adapter


def save_residual(adapter, directory):
    tensors = {name: value.detach().cpu().contiguous() for name, value in adapter.state_dict().items()}
    save_file(tensors, str(Path(directory) / ADAPTER_FILE))


def load_residual(model, directory, model_path, model_revision):
    directory = Path(directory)
    meta = json.loads((directory / "training_run.json").read_text())
    from spatial_merger import EXPERIMENT_D, attach_spatial

    if meta.get("experiment") == EXPERIMENT_C:
        attach = attach_residual
    elif meta.get("experiment") == EXPERIMENT_D:
        attach = attach_spatial
    else:
        raise ValueError(f"not an Experiment C or D artifact: {directory}")
    if model_revision is not None and meta.get("base_revision") is not None:
        if (meta["base_model"], meta["base_revision"]) != (model_path, model_revision):
            raise ValueError("the residual adapter was trained on a different base model revision")
    if meta.get("base_merger_sha256") != base_merger_digest(model):
        raise ValueError("the residual adapter needs its original frozen merger, not a tuned merger")
    weights = load_file(str(directory / ADAPTER_FILE), device="cpu")
    adapter = attach(model, model.model.visual.merger.linear_fc2.weight.dtype)
    adapter.load_state_dict(weights, strict=True)
    adapter.requires_grad_(False).eval()

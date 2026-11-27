"""Experiment D: spatial residual mixing on ViT patches before the basic merger."""

from contextvars import ContextVar

import torch
from torch import nn

EXPERIMENT_D = "D_pre_merger_spatial"
SPATIAL_WIDTH = 320
SPATIAL_PARAMETERS = 659_904


def unpack_patches(tokens, frames, height, width, merge):
    """Convert Qwen's block-major patch order to [T, C, H, W]."""
    channels = tokens.shape[-1]
    return (tokens.reshape(frames, height // merge, width // merge, merge, merge, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(frames, channels, height, width))


def pack_patches(grid, merge):
    """Restore Qwen's block-major order so the stock merger sees the same 2×2 groups."""
    frames, channels, height, width = grid.shape
    return (grid.reshape(frames, channels, height // merge, merge, width // merge, merge)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, channels))


class SpatialMixer(nn.Module):
    def __init__(self, width, merge_size):
        super().__init__()
        self.merge_size = merge_size
        self.down = nn.Conv2d(width, SPATIAL_WIDTH, 1)
        self.spatial = nn.Conv2d(SPATIAL_WIDTH, SPATIAL_WIDTH, 3, padding=1, groups=SPATIAL_WIDTH)
        self.up = nn.Conv2d(SPATIAL_WIDTH, width, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens, grid_thw):
        grids = grid_thw.tolist()
        merge = self.merge_size
        if not grids or any(t < 1 or h < 1 or w < 1 or h % merge or w % merge for t, h, w in grids):
            raise ValueError(f"invalid patch grids for spatial merge size {merge}: {grids}")
        sizes = [t * h * w for t, h, w in grids]
        if tokens.ndim != 2 or sum(sizes) != tokens.shape[0]:
            raise ValueError("ViT token count does not match grid_thw")
        mixed = []
        for patches, (t, h, w) in zip(tokens.split(sizes), grids):
            # Each image is separate and T is a batch dimension, so convolution never
            # crosses image or frame boundaries.
            grid = unpack_patches(patches, t, h, w, merge)
            hidden = self.spatial(self.down(grid.to(self.down.weight.dtype)))
            residual = self.up(nn.functional.gelu(hidden))
            mixed.append(patches + pack_patches(residual, merge).to(patches.dtype))
        return torch.cat(mixed, dim=0)


def attach_spatial(model, dtype):
    """Capture Qwen's grid during vision forward and mix only the basic merger input."""
    vision = model.model.visual
    merger = vision.merger
    if hasattr(merger, "residual_adapter"):
        raise ValueError("a residual adapter is already attached")
    weight = merger.linear_fc2.weight
    adapter = SpatialMixer(vision.config.hidden_size, vision.spatial_merge_size).to(
        device=weight.device, dtype=dtype)
    merger.add_module("residual_adapter", adapter)

    # The final merger itself receives only the flat tokens. Keep grid metadata
    # scoped to the enclosing vision call, including when that call raises.
    grids = ContextVar("spatial_merger_grids", default=())

    def capture_grid(module, args, kwargs):
        grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        grids.set((*grids.get(), grid))

    def mix_patches(module, args):
        stack = grids.get()
        if not stack or stack[-1] is None:
            raise RuntimeError("spatial mixing requires grid_thw from the Qwen vision encoder")
        return (module.residual_adapter(args[0], stack[-1]), *args[1:])

    def clear_grid(module, args, output):
        grids.set(grids.get()[:-1])

    vision.register_forward_pre_hook(capture_grid, with_kwargs=True)
    merger.register_forward_pre_hook(mix_patches)
    vision.register_forward_hook(clear_grid, always_call=True)
    return adapter

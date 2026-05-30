from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict

_COLORMAPS: Dict[str, np.ndarray] = {}


def _build_colormap(name: str) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    if name == "viridis":
        r = np.interp(t, [0,.25,.5,.75,1], [0.267,0.230,0.128,0.370,0.993])
        g = np.interp(t, [0,.25,.5,.75,1], [0.005,0.322,0.566,0.718,0.906])
        b = np.interp(t, [0,.25,.5,.75,1], [0.329,0.546,0.551,0.389,0.144])
    elif name == "inferno":
        r = np.interp(t, [0,.25,.5,.75,1], [0.000,0.216,0.622,0.941,0.988])
        g = np.interp(t, [0,.25,.5,.75,1], [0.000,0.028,0.160,0.548,0.998])
        b = np.interp(t, [0,.25,.5,.75,1], [0.014,0.417,0.258,0.040,0.644])
    elif name == "magma":
        r = np.interp(t, [0,.25,.5,.75,1], [0.000,0.192,0.580,0.906,0.988])
        g = np.interp(t, [0,.25,.5,.75,1], [0.000,0.040,0.149,0.527,0.991])
        b = np.interp(t, [0,.25,.5,.75,1], [0.016,0.408,0.404,0.353,0.750])
    elif name == "plasma":
        r = np.interp(t, [0,.25,.5,.75,1], [0.050,0.464,0.799,0.973,0.940])
        g = np.interp(t, [0,.25,.5,.75,1], [0.030,0.045,0.320,0.641,0.975])
        b = np.interp(t, [0,.25,.5,.75,1], [0.528,0.680,0.369,0.062,0.131])
    elif name == "hot":
        r = np.clip(t * 3.0,        0.0, 1.0)
        g = np.clip(t * 3.0 - 1.0, 0.0, 1.0)
        b = np.clip(t * 3.0 - 2.0, 0.0, 1.0)
    elif name == "turbo":
        r = np.interp(t, [0,.2,.4,.6,.8,1], [0.188,0.055,0.322,0.978,0.826,0.479])
        g = np.interp(t, [0,.2,.4,.6,.8,1], [0.071,0.561,0.859,0.788,0.236,0.006])
        b = np.interp(t, [0,.2,.4,.6,.8,1], [0.231,0.890,0.396,0.058,0.047,0.013])
    elif name == "coolwarm":
        r = np.interp(t, [0,.5,1], [0.230,0.865,0.706])
        g = np.interp(t, [0,.5,1], [0.299,0.865,0.016])
        b = np.interp(t, [0,.5,1], [0.754,0.865,0.150])
    else:  # gray
        r = g = b = t
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def get_colormap(name: str) -> np.ndarray:
    if name not in _COLORMAPS:
        _COLORMAPS[name] = _build_colormap(name)
    return _COLORMAPS[name]


def apply_colormap_batch(maps: np.ndarray, name: str) -> np.ndarray:
    """[..., H, W] float32 [0,1] → [..., H, W, 3] float32 [0,1]"""
    lut = get_colormap(name)
    idx = (np.clip(maps, 0.0, 1.0) * 255).astype(np.uint8)
    return lut[idx].astype(np.float32)


def add_grid_lines(img: np.ndarray, cell_size: int,
                   n_rows: int, n_cols: int,
                   step_y: int = 8, step_x: int = 8) -> np.ndarray:
    img        = img.copy()
    line_color = np.array([0.40, 0.40, 0.40], dtype=np.float32)
    for row in range(0, n_rows, step_y):
        y = row * cell_size
        if 0 <= y < img.shape[0]:
            img[y, :] = line_color
    for col in range(0, n_cols, step_x):
        x = col * cell_size
        if 0 <= x < img.shape[1]:
            img[:, x] = line_color
    return img


def draw_line(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int,
              color, thickness: int = 1):
    H, W  = canvas.shape[:2]
    color = np.asarray(color, dtype=np.float32)
    dx    = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy    = abs(y1 - y0); sy = 1 if y0 < y1 else -1
    err   = (dx if dx > dy else -dy) // 2
    half  = thickness // 2
    while True:
        for dy_ in range(-half, half + 1):
            for dx_ in range(-half, half + 1):
                nx, ny = x0 + dx_, y0 + dy_
                if 0 <= nx < W and 0 <= ny < H:
                    canvas[ny, nx] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = err
        if e2 > -dx: err -= dy; x0 += sx
        if e2 <  dy: err += dx; y0 += sy


def render_head_grid(maps: torch.Tensor, head_list: list, H_heads: int,
                     cell_size: int, colormap: str,
                     normalize_per_head: bool) -> torch.Tensor:
    """
    maps : [n_heads, Lh, Lw] float
    → IMAGE tensor [1, H, W, 3]
    """
    n_heads, Lh, Lw = maps.shape
    cell_h = cell_size
    cell_w = max(1, int(cell_size * Lw / max(Lh, 1)))

    maps_rs = F.interpolate(
        maps.unsqueeze(1), size=(cell_h, cell_w),
        mode="bilinear", align_corners=False,
    ).squeeze(1)

    if normalize_per_head:
        mn      = maps_rs.view(n_heads, -1).min(dim=1).values.view(n_heads, 1, 1)
        mx      = maps_rs.view(n_heads, -1).max(dim=1).values.view(n_heads, 1, 1)
        maps_rs = (maps_rs - mn) / (mx - mn + 1e-8)
    else:
        maps_rs = (maps_rs - maps_rs.min()) / (maps_rs.max() - maps_rs.min() + 1e-8)

    colored = apply_colormap_batch(maps_rs.cpu().numpy(), colormap)

    n_cols  = math.ceil(math.sqrt(n_heads))
    n_rows  = math.ceil(n_heads / n_cols)
    pad_px  = 3
    label_h = 12
    lut     = get_colormap("turbo")

    grid_h  = pad_px + n_rows * (label_h + cell_h + pad_px)
    grid_w  = pad_px + n_cols * (cell_w          + pad_px)
    grid    = np.full((grid_h, grid_w, 3), 0.10, dtype=np.float32)

    for i, (h_idx, cell_img) in enumerate(zip(head_list, colored)):
        row     = i // n_cols
        col     = i %  n_cols
        y_label = pad_px + row * (label_h + cell_h + pad_px)
        y_cell  = y_label + label_h
        x0      = pad_px + col * (cell_w + pad_px)
        hue     = lut[int(h_idx / max(H_heads - 1, 1) * 255)]
        grid[y_label:y_cell,         x0:x0 + cell_w] = hue * 0.6
        grid[y_cell:y_cell + cell_h, x0:x0 + cell_w] = cell_img

    return torch.from_numpy(grid).unsqueeze(0).clamp(0.0, 1.0)
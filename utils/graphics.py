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
    elif name == "diverging":
        # Blue -> black -> red, for signed diffs: 0 reads as "no change"
        # instead of coolwarm's near-white midpoint.
        r = np.interp(t, [0,.5,1], [0.100,0.000,0.950])
        g = np.interp(t, [0,.5,1], [0.350,0.000,0.080])
        b = np.interp(t, [0,.5,1], [0.950,0.000,0.080])
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


# ── Minimal 3x5 bitmap font (digits + sign/punctuation) ─────────────────────
# No PIL dependency in this toolkit — just enough glyphs to label a colorbar.

_FONT_3x5 = {
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("###", "..#", "###", "#..", "###"),
    "3": ("###", "..#", "###", "..#", "###"),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "###", "..#", "###"),
    "6": ("###", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", "..#", "..#", "..#"),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "###"),
    "-": ("...", "...", "###", "...", "..."),
    "+": ("...", ".#.", "###", ".#.", "..."),
    ".": ("...", "...", "...", "...", ".#."),
    " ": ("...", "...", "...", "...", "..."),
}


def draw_text(canvas: np.ndarray, text: str, x: int, y: int,
              color=(1.0, 1.0, 1.0), scale: int = 2) -> None:
    """Stamp a string onto an RGB float32 canvas using the bitmap font above.
    (x, y) is the top-left corner. Unsupported characters render as blank."""
    color = np.asarray(color, dtype=np.float32)
    H, W = canvas.shape[:2]
    cx = x
    for ch in text:
        glyph = _FONT_3x5.get(ch, _FONT_3x5[" "])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit != "#":
                    continue
                py0, py1 = y + row * scale, y + (row + 1) * scale
                px0, px1 = cx + col * scale, cx + (col + 1) * scale
                py0, py1 = max(py0, 0), min(py1, H)
                px0, px1 = max(px0, 0), min(px1, W)
                if py1 > py0 and px1 > px0:
                    canvas[py0:py1, px0:px1] = color
        cx += (3 + 1) * scale


def _fmt_num(v: float) -> str:
    """Fixed-point formatting (never scientific notation — the bitmap
    font above has no 'e' glyph), precision scaled to magnitude."""
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def make_colorbar(clip_val: float, colormap: str, width: int = 240,
                  height: int = 34, scale: int = 2) -> np.ndarray:
    """Horizontal gradient strip from -clip_val (left) to +clip_val (right),
    with numeric end labels and a center tick at zero. For signed diffs."""
    lut    = get_colormap(colormap)
    bar_h  = height - 10
    t      = np.linspace(0.0, 1.0, width, dtype=np.float32)
    idx    = (t * 255).astype(np.uint8)
    strip  = lut[idx].astype(np.float32)
    canvas = np.full((height, width, 3), 0.08, dtype=np.float32)
    canvas[0:bar_h] = np.broadcast_to(strip, (bar_h, width, 3))
    canvas[bar_h:bar_h + 2, width // 2 - 1:width // 2 + 1] = 1.0  # zero tick

    lo_label = f"-{_fmt_num(clip_val)}"
    hi_label = f"+{_fmt_num(clip_val)}"
    draw_text(canvas, lo_label, 2, bar_h + 2, color=(1, 1, 1), scale=1)
    hi_x = max(width - (len(hi_label) * 4 * 1) - 2, width // 2 + 4)
    draw_text(canvas, hi_label, hi_x, bar_h + 2, color=(1, 1, 1), scale=1)
    return canvas


def make_range_colorbar(vmin: float, vmax: float, colormap: str, width: int = 240,
                        height: int = 34, scale: int = 2) -> np.ndarray:
    """Horizontal gradient strip from vmin (left) to vmax (right), with
    numeric end labels — for absolute-value heatmaps (not signed diffs),
    e.g. Metrics Heatmap where 0 isn't a meaningful center point."""
    lut    = get_colormap(colormap)
    bar_h  = height - 10
    t      = np.linspace(0.0, 1.0, width, dtype=np.float32)
    idx    = (t * 255).astype(np.uint8)
    strip  = lut[idx].astype(np.float32)
    canvas = np.full((height, width, 3), 0.08, dtype=np.float32)
    canvas[0:bar_h] = np.broadcast_to(strip, (bar_h, width, 3))

    lo_label = _fmt_num(vmin)
    hi_label = _fmt_num(vmax)
    draw_text(canvas, lo_label, 2, bar_h + 2, color=(1, 1, 1), scale=1)
    hi_x = max(width - (len(hi_label) * 4) - 2, width // 2 + 4)
    draw_text(canvas, hi_label, hi_x, bar_h + 2, color=(1, 1, 1), scale=1)
    return canvas


def vstack_padded(images: list, pad_color: float = 0.08) -> np.ndarray:
    """Stack [H, W, 3] float32 images vertically, padding narrower ones to
    the widest, instead of requiring identical widths."""
    max_w = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        if img.shape[1] < max_w:
            canvas = np.full((img.shape[0], max_w, 3), pad_color, dtype=np.float32)
            canvas[:, :img.shape[1]] = img
            padded.append(canvas)
        else:
            padded.append(img)
    return np.concatenate(padded, axis=0)


def add_grid_lines(img: np.ndarray, cell_size: int,
                   n_rows: int, n_cols: int,
                   step_y: int = 8, step_x: int = 8,
                   cell_h: int = None, cell_w: int = None,
                   y_offset: int = 0, x_offset: int = 0,
                   pad: int = 0) -> np.ndarray:
    """Draw a subtle separator line every step_y/step_x rows/cols, to
    visually group blocks/heads in batches (e.g. every 8) for easier
    reading of a large grid. cell_h/cell_w default to cell_size (square
    cells, the common case); pass them explicitly plus y_offset/x_offset
    (label margins) and pad (per-cell spacing) for grids that aren't a
    flat cell_size x cell_size tiling, e.g. Grid Viz."""
    cell_h = cell_size if cell_h is None else cell_h
    cell_w = cell_size if cell_w is None else cell_w
    img        = img.copy()
    line_color = np.array([0.40, 0.40, 0.40], dtype=np.float32)
    H, W = img.shape[:2]
    for row in range(0, n_rows, step_y):
        y = y_offset + row * (cell_h + pad)
        if 0 <= y < H:
            img[y, x_offset:] = line_color
    for col in range(0, n_cols, step_x):
        x = x_offset + col * (cell_w + pad)
        if 0 <= x < W:
            img[y_offset:, x] = line_color
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
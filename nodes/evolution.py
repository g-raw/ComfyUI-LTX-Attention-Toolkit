from __future__ import annotations
import numpy as np
import torch

from ..core.stores    import AttentionStore, get_registry
from ..utils.graphics import get_colormap, draw_line


class LTXAttentionTimestepEvolution:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "store_handle":   ("STRING", {"default": "", "placeholder": "select store..."}),
            "attn_type":    (["sa", "ca"], {"default": "sa"}),
            "block_idx":    ("INT",    {"default": 0, "min": 0, "max": 47}),
            "metric":       (["entropy","temporal","spatial","sink"],
                            {"default": "entropy"}),
            "head_indices": ("STRING", {"default": "0,4,8,16,24,31"}),
            "img_width":    ("INT",    {"default": 512, "min": 128, "max": 2048}),
            "img_height":   ("INT",    {"default": 256, "min": 64,  "max": 1024}),
            "colormap":     (["turbo","viridis","plasma"], {"default": "turbo"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("evolution_plot", "stats_text")
    FUNCTION     = "plot"
    CATEGORY     = "g_raw/LTX/Profiler"

    def plot(self, store_handle, attn_type, block_idx, metric, head_indices,
             img_width, img_height, colormap):

        if store_handle and store_handle.strip():


            get_registry().switch_attn(store_handle)


        store = AttentionStore()
        src   = store.sa if attn_type == "sa" else store.ca

        if block_idx not in src:
            raise ValueError(
                f"Block {block_idx} not found. Available: {sorted(src.keys())}"
            )

        steps_data = src[block_idx]
        step_keys  = sorted(steps_data.keys())
        n_steps    = len(step_keys)

        if n_steps < 2:
            raise ValueError(f"Only {n_steps} step(s). Need at least 2.")

        first_entry = steps_data[step_keys[0]]
        n_heads     = len(first_entry.get(metric, []))
        if n_heads == 0:
            raise ValueError(f"Metric '{metric}' not found in data.")

        if head_indices.strip().lower() == "all":
            head_list = list(range(n_heads))
        else:
            head_list = [int(x.strip()) for x in head_indices.split(",")
                         if x.strip() and 0 <= int(x.strip()) < n_heads]

        data      = np.zeros((len(head_list), n_steps), dtype=np.float32)
        ts_labels = []
        for col, sk in enumerate(step_keys):
            entry = steps_data[sk]
            vals  = entry.get(metric, torch.zeros(n_heads)).numpy()
            for row, h in enumerate(head_list):
                data[row, col] = vals[h] if h < len(vals) else 0.0
            ts_labels.append(f"{entry.get('timestep', sk):.2f}")

        canvas = np.full((img_height, img_width, 3), 0.08, dtype=np.float32)
        lut    = get_colormap(colormap)
        mx_    = 30
        my_    = 10
        plot_w = img_width  - mx_ - 8
        plot_h = img_height - my_ - 24
        v_min, v_max = data.min(), data.max()
        v_range      = max(v_max - v_min, 1e-6)

        for yi in np.linspace(0, 1, 5):
            ypx = my_ + int((1.0 - yi) * plot_h)
            if 0 <= ypx < img_height:
                canvas[ypx, mx_:mx_+plot_w] = [0.22, 0.22, 0.22]

        for col in range(n_steps):
            xpx = mx_ + int(col / max(n_steps-1, 1) * plot_w)
            if 0 <= xpx < img_width:
                canvas[my_:my_+plot_h, xpx] = [0.18, 0.18, 0.18]

        for row, h_idx in enumerate(head_list):
            color = lut[int(h_idx / max(n_heads-1, 1) * 255)]
            for col in range(n_steps - 1):
                y0  = (data[row, col]     - v_min) / v_range
                y1  = (data[row, col + 1] - v_min) / v_range
                x0  = mx_ + int(col       / max(n_steps-1, 1) * plot_w)
                x1  = mx_ + int((col + 1) / max(n_steps-1, 1) * plot_w)
                py0 = img_height - 24 - int(y0 * plot_h)
                py1 = img_height - 24 - int(y1 * plot_h)
                draw_line(canvas, x0, py0, x1, py1, color)

        for yi in range(plot_h):
            v = 1.0 - yi / plot_h
            canvas[my_ + yi, 0:mx_-3] = [v, v, v]

        out   = torch.from_numpy(canvas).unsqueeze(0).clamp(0.0, 1.0)
        stats = (
            f"Block {block_idx} | {metric} ({attn_type})\n"
            f"Steps: {n_steps}  Heads: {head_list}\n"
            f"Range: [{v_min:.4f}, {v_max:.4f}]\n"
            f"Timesteps: {ts_labels[0]} → {ts_labels[-1]}"
        )
        return (out, stats)
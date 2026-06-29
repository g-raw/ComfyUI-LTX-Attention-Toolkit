from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from ..utils.graphics import apply_colormap_batch, add_grid_lines


class LTXLatentDims:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("num_frames", "latent_height", "latent_width")
    FUNCTION     = "extract"
    CATEGORY     = "g_raw/LTX/Profiler"

    def extract(self, latent):
        s = latent["samples"]
        if s.ndim == 5:
            _, _, T, H, W = s.shape
        elif s.ndim == 4:
            _, _, H, W = s.shape
            T = 1
        else:
            raise ValueError(f"Shape inattendue: {s.shape}")
        print(f"[LTXLatentDims] T={T} H={H} W={W}")
        return (T, H, W)


class LTXAttentionCompareRuns:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "path_run_a": ("STRING", {"default": "run_a.pt"}),
            "path_run_b": ("STRING", {"default": "run_b.pt"}),
            "attn_type":  (["sa", "ca"], {"default": "sa"}),
            "metric":     (["entropy","temporal","spatial","sink"],
                          {"default": "entropy"}),
            "step_idx":   ("INT",    {"default": -1, "min": -1, "max": 255}),
            "colormap":   (["coolwarm","viridis","inferno"], {"default": "coolwarm"}),
            "cell_size":  ("INT",    {"default": 16, "min": 4, "max": 64}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("diff_heatmap", "stats_text")
    FUNCTION     = "compare"
    CATEGORY     = "g_raw/LTX/Profiler"

    @staticmethod
    def _extract(run, attn_type, metric, step_idx):
        src           = run[attn_type]
        block_indices = sorted(src.keys())
        n_heads       = 0
        for steps in src.values():
            for e in steps.values():
                if metric in e and len(e[metric]) > 0:
                    n_heads = len(e[metric]); break
            if n_heads: break
        if n_heads == 0:
            raise ValueError(f"Metric '{metric}' not found.")
        mat = np.zeros((n_heads, len(block_indices)), dtype=np.float32)
        for col, blk in enumerate(block_indices):
            steps_data = src[blk]
            if step_idx == -1:
                stack = [v[metric].numpy() for v in steps_data.values()
                         if metric in v and len(v[metric]) == n_heads]
                vals  = np.stack(stack).mean(0) if stack else np.zeros(n_heads)
            else:
                e    = steps_data.get(step_idx, {})
                vals = (e[metric].numpy()
                        if metric in e and len(e[metric]) == n_heads
                        else np.zeros(n_heads))
            mat[:, col] = vals
        return mat

    def compare(self, path_run_a, path_run_b, attn_type, metric,
                step_idx, colormap, cell_size):
        run_a = torch.load(path_run_a, map_location="cpu", weights_only=False)
        run_b = torch.load(path_run_b, map_location="cpu", weights_only=False)
        mat_a = self._extract(run_a, attn_type, metric, step_idx)
        mat_b = self._extract(run_b, attn_type, metric, step_idx)
        mh    = min(mat_a.shape[0], mat_b.shape[0])
        mb    = min(mat_a.shape[1], mat_b.shape[1])
        mat_a, mat_b = mat_a[:mh, :mb], mat_b[:mh, :mb]
        diff      = mat_a - mat_b
        abs_max   = max(abs(float(diff.min())), abs(float(diff.max())), 1e-8)
        diff_norm = (diff / abs_max) * 0.5 + 0.5
        colored   = apply_colormap_batch(diff_norm[np.newaxis], colormap)[0]
        out_h, out_w = mh * cell_size, mb * cell_size
        ct = (torch.from_numpy(colored).permute(2,0,1).unsqueeze(0).float())
        ct = F.interpolate(ct, size=(out_h, out_w), mode="nearest")
        img_np = ct.squeeze(0).permute(1,2,0).numpy()
        img_np = add_grid_lines(img_np, cell_size, mh, mb)
        out    = torch.from_numpy(img_np).unsqueeze(0).clamp(0.0, 1.0)
        stats  = (
            f"Metric: {metric} ({attn_type}) | Step: {step_idx}\n"
            f"A: mean={mat_a.mean():.4f} std={mat_a.std():.4f}\n"
            f"B: mean={mat_b.mean():.4f} std={mat_b.std():.4f}\n"
            f"Diff: mean={diff.mean():.4f} std={diff.std():.4f}\n"
            f"Max A>B: {diff.max():.4f} | Max B>A: {(-diff).max():.4f}"
        )
        return (out, stats)
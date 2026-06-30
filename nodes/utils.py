from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from ..core.stores     import get_registry
from ..utils.graphics  import apply_colormap_batch, add_grid_lines


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
    """Diff two captures (live store_handle, or a dumped .pt as fallback)
    for one metric, ranked by |A - B| per (block, head)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "store_handle_a": ("STRING", {"default": "",
                               "placeholder": "in-memory store A (overrides path_run_a)"}),
            "store_handle_b": ("STRING", {"default": "",
                               "placeholder": "in-memory store B (overrides path_run_b)"}),
            "path_run_a": ("STRING", {"default": "run_a.pt"}),
            "path_run_b": ("STRING", {"default": "run_b.pt"}),
            "attn_type":  (["sa", "ca"], {"default": "sa"}),
            "metric":     (["entropy","temporal","spatial","sink"],
                          {"default": "entropy"}),
            "step_idx":   ("INT",    {"default": -1, "min": -1, "max": 255}),
            "colormap":   (["coolwarm","viridis","inferno"], {"default": "coolwarm"}),
            "cell_size":  ("INT",    {"default": 16, "min": 4, "max": 64}),
            "top_k":      ("INT",    {"default": 15, "min": 1, "max": 1536,
                           "tooltip": "How many (block, head) pairs to list, ranked by |A - B|."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("diff_heatmap", "stats_text")
    FUNCTION     = "compare"
    CATEGORY     = "g_raw/LTX/Profiler"

    @staticmethod
    def _load_run(path: str, store_handle: str, reg):
        """Read from a live store_handle if given, else fall back to a dumped .pt."""
        if store_handle and store_handle.strip():
            inst = reg._get_attn(store_handle.strip())
            return {"sa": inst.sa, "ca": inst.ca, "cfg": inst.cfg}
        return torch.load(path, map_location="cpu", weights_only=False)

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
        return mat, block_indices

    def compare(self, store_handle_a, store_handle_b, path_run_a, path_run_b,
                attn_type, metric, step_idx, colormap, cell_size, top_k):
        reg   = get_registry()
        run_a = self._load_run(path_run_a, store_handle_a, reg)
        run_b = self._load_run(path_run_b, store_handle_b, reg)

        mat_a, blocks_a = self._extract(run_a, attn_type, metric, step_idx)
        mat_b, blocks_b = self._extract(run_b, attn_type, metric, step_idx)

        # Align by actual block index, not column position — the two runs
        # may have captured different (or differently-ordered) target_blocks.
        common_blocks = sorted(set(blocks_a) & set(blocks_b))
        if not common_blocks:
            raise ValueError(
                f"[CompareRuns] No common blocks between run A {blocks_a} "
                f"and run B {blocks_b}."
            )
        mat_a = mat_a[:, [blocks_a.index(b) for b in common_blocks]]
        mat_b = mat_b[:, [blocks_b.index(b) for b in common_blocks]]

        mh = min(mat_a.shape[0], mat_b.shape[0])
        mat_a, mat_b = mat_a[:mh], mat_b[:mh]

        diff      = mat_a - mat_b
        abs_max   = max(abs(float(diff.min())), abs(float(diff.max())), 1e-8)
        diff_norm = (diff / abs_max) * 0.5 + 0.5
        colored   = apply_colormap_batch(diff_norm[np.newaxis], colormap)[0]
        out_h, out_w = mh * cell_size, len(common_blocks) * cell_size
        ct = (torch.from_numpy(colored).permute(2,0,1).unsqueeze(0).float())
        ct = F.interpolate(ct, size=(out_h, out_w), mode="nearest")
        img_np = ct.squeeze(0).permute(1,2,0).numpy()
        img_np = add_grid_lines(img_np, cell_size, mh, len(common_blocks))
        out    = torch.from_numpy(img_np).unsqueeze(0).clamp(0.0, 1.0)

        # ── Ranked (block, head) table by |A - B| ─────────────────────────
        flat_idx = np.argsort(np.abs(diff).ravel())[::-1][:top_k]
        rank_lines = []
        for rank, fi in enumerate(flat_idx, start=1):
            h, col = np.unravel_index(fi, diff.shape)
            blk    = common_blocks[col]
            rank_lines.append(
                f"#{rank:2d}  block={blk:2d} head={h:2d}  "
                f"diff(A-B)={diff[h,col]:+.4f}  (A={mat_a[h,col]:.4f} B={mat_b[h,col]:.4f})"
            )

        stats  = (
            f"Metric: {metric} ({attn_type}) | Step: {step_idx} | "
            f"Blocks compared: {common_blocks}\n"
            f"A: mean={mat_a.mean():.4f} std={mat_a.std():.4f}\n"
            f"B: mean={mat_b.mean():.4f} std={mat_b.std():.4f}\n"
            f"Diff: mean={diff.mean():.4f} std={diff.std():.4f}\n"
            f"Max A>B: {diff.max():.4f} | Max B>A: {(-diff).max():.4f}\n\n"
            f"Top {min(top_k, diff.size)} by |A-B|:\n" + "\n".join(rank_lines)
        )
        return (out, stats)
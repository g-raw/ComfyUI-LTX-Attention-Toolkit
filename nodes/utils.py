from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from ..core.stores     import get_registry
from ..utils.graphics  import (apply_colormap_batch, add_grid_lines,
                               make_colorbar, vstack_padded)


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
    """Diff two captures (read live from the registry by store_handle)
    for one metric, ranked by |A - B| per (block, head). To compare a
    dumped .pt, load it into a handle first with Store Load."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "store_handle_a": ("STRING", {"default": "",
                               "placeholder": "store A handle"}),
            "store_handle_b": ("STRING", {"default": "",
                               "placeholder": "store B handle"}),
            "attn_type":  (["sa", "ca"], {"default": "sa"}),
            "metric":     (["entropy","temporal","spatial","sink",
                           "frame_dist_mean","frame_dist_std",
                           "frame_dist_mean_norm","frame_dist_std_norm",
                           "spatial_dist_mean","spatial_dist_std",
                           "spatial_dist_mean_norm","spatial_dist_std_norm"],
                          {"default": "entropy"}),
            "step_idx":   ("INT",    {"default": -1, "min": -1, "max": 255}),
            "colormap":   (["diverging","coolwarm","viridis","inferno"],
                          {"default": "diverging",
                           "tooltip": "diverging: 0 = black, so identical cells read as neutral "
                                      "instead of coolwarm's near-white midpoint."}),
            "cell_size":  ("INT",    {"default": 16, "min": 4, "max": 64}),
            "top_k":      ("INT",    {"default": 15, "min": 1, "max": 1536,
                           "tooltip": "How many (block, head) pairs to list, ranked by the diff_mode score."}),
            "norm_percentile": ("FLOAT", {"default": 0.98, "min": 0.5, "max": 1.0, "step": 0.01,
                           "tooltip": "Clip the diff_mode score beyond this percentile before mapping "
                                      "to color, so a few outlier cells don't wash out the rest of the "
                                      "heatmap to white. 1.0 = no clipping (use the true max)."}),
            "diff_mode":  (["absolute", "relative_pct", "zscore"], {"default": "absolute",
                           "tooltip": "absolute: A - B, in the metric's own units. Not comparable "
                                      "across metrics with different intrinsic scales (e.g. sink in "
                                      "[0,1] vs raw temporal/spatial scores).\n"
                                      "relative_pct: (A-B) / max(|A|,|B|) * 100 -- % change, scale-free.\n"
                                      "zscore: (A-B) / std(A and B combined) -- diff in units of the "
                                      "metric's own spread, the most apples-to-apples way to ask "
                                      "whether one metric moved proportionally more than another."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("diff_heatmap", "stats_text")
    FUNCTION     = "compare"
    CATEGORY     = "g_raw/LTX/Profiler"

    @staticmethod
    def _load_run(store_handle: str, reg):
        if not store_handle or not store_handle.strip():
            raise ValueError("[CompareRuns] store_handle is required.")
        inst = reg._get_attn(store_handle.strip())
        return {"sa": inst.sa, "ca": inst.ca, "cfg": inst.cfg}

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

    def compare(self, store_handle_a, store_handle_b,
                attn_type, metric, step_idx, colormap, cell_size, top_k,
                norm_percentile, diff_mode):
        reg   = get_registry()
        run_a = self._load_run(store_handle_a, reg)
        run_b = self._load_run(store_handle_b, reg)

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

        eps      = 1e-8
        raw_diff = mat_a - mat_b

        if diff_mode == "relative_pct":
            denom    = np.maximum(np.abs(mat_a), np.abs(mat_b)) + eps
            score    = raw_diff / denom * 100.0
            unit     = "%"
            score_fmt = "{:+.1f}%"
        elif diff_mode == "zscore":
            combined_std = float(np.concatenate([mat_a.ravel(), mat_b.ravel()]).std())
            denom    = combined_std + eps
            score    = raw_diff / denom
            unit     = "σ"
            score_fmt = "{:+.3f}σ"
        else:
            score     = raw_diff
            unit      = "(raw units)"
            score_fmt = "{:+.4f}"

        clip_val  = max(float(np.percentile(np.abs(score), norm_percentile * 100)), 1e-8)
        diff_clip = np.clip(score, -clip_val, clip_val)
        diff_norm = (diff_clip / clip_val) * 0.5 + 0.5
        colored   = apply_colormap_batch(diff_norm[np.newaxis], colormap)[0]
        out_h, out_w = mh * cell_size, len(common_blocks) * cell_size
        ct = (torch.from_numpy(colored).permute(2,0,1).unsqueeze(0).float())
        ct = F.interpolate(ct, size=(out_h, out_w), mode="nearest")
        img_np = ct.squeeze(0).permute(1,2,0).numpy()
        img_np = add_grid_lines(img_np, cell_size, mh, len(common_blocks))
        colorbar = make_colorbar(clip_val, colormap, width=out_w)
        img_np   = vstack_padded([img_np, colorbar])
        out      = torch.from_numpy(img_np).unsqueeze(0).clamp(0.0, 1.0)

        # ── Ranked (block, head) table by |score| ──────────────────────────
        flat_idx = np.argsort(np.abs(score).ravel())[::-1][:top_k]
        rank_lines = []
        for rank, fi in enumerate(flat_idx, start=1):
            h, col = np.unravel_index(fi, score.shape)
            blk    = common_blocks[col]
            rank_lines.append(
                f"#{rank:2d}  block={blk:2d} head={h:2d}  "
                f"score={score_fmt.format(score[h,col])}  "
                f"(raw diff={raw_diff[h,col]:+.4f}, A={mat_a[h,col]:.4f}, B={mat_b[h,col]:.4f})"
            )

        stats  = (
            f"A = '{store_handle_a}'  |  B = '{store_handle_b}'\n"
            f"diff = A - B  →  positive (red) = A > B, negative (blue) = B > A\n"
            f"diff_mode = {diff_mode}, units = {unit}\n"
            f"Metric: {metric} ({attn_type}) | Step: {step_idx} | "
            f"Blocks compared: {common_blocks}\n"
            f"A: mean={mat_a.mean():.4f} std={mat_a.std():.4f} "
            f"min={mat_a.min():.4f} max={mat_a.max():.4f}\n"
            f"B: mean={mat_b.mean():.4f} std={mat_b.std():.4f} "
            f"min={mat_b.min():.4f} max={mat_b.max():.4f}\n"
            f"Raw diff (A-B): mean={raw_diff.mean():.4f} std={raw_diff.std():.4f}\n"
            f"Max A>B: {raw_diff.max():.4f} | Max B>A: {(-raw_diff).max():.4f}\n"
            f"Heatmap color scale clipped at ±{clip_val:.4f} {unit} "
            f"({norm_percentile*100:.0f}th percentile of |{diff_mode}|), "
            f"see colorbar at the bottom of the image\n\n"
            f"Top {min(top_k, score.size)} by |{diff_mode}|:\n" + "\n".join(rank_lines)
        )
        return (out, stats)


class LTXAttentionHeadCandidates:
    """Combine the zscore diff of several metrics into one composite score
    per (block, head), to shortlist heads for a targeted intervention (e.g.
    Head Freeze) instead of eyeballing several separate heatmaps. Reuses
    LTXAttentionCompareRuns's store-loading/extraction/zscore logic."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "store_handle_a": ("STRING", {"default": "",
                               "placeholder": "store A handle"}),
            "store_handle_b": ("STRING", {"default": "",
                               "placeholder": "store B handle"}),
            "attn_type": (["sa", "ca"], {"default": "sa"}),
            "metrics":   ("STRING", {"default": "temporal,frame_dist_mean_norm,frame_dist_std_norm",
                          "tooltip": "Comma-separated metric names. Composite score = "
                                     "mean(|zscore|) across all of them, per (block, head)."}),
            "step_idx":  ("INT", {"default": -1, "min": -1, "max": 255}),
            "top_k":     ("INT", {"default": 12, "min": 1, "max": 1536,
                          "tooltip": "How many (block, head) candidates to shortlist, "
                                     "ranked by the composite score."}),
            "control_mode": (["lowest_score", "random"], {"default": "lowest_score",
                             "tooltip": "How to pick the control group: the heads least "
                                        "implicated by these metrics, or a random sample "
                                        "(excluding the candidates either way)."}),
            "control_k": ("INT", {"default": 12, "min": 0, "max": 1536,
                          "tooltip": "Control group size. 0 to skip it."}),
            "seed":      ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                          "tooltip": "Only used when control_mode=random."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("report", "candidates_csv", "control_csv")
    FUNCTION     = "rank"
    CATEGORY     = "g_raw/LTX/Profiler"

    @staticmethod
    def _zscore(run_a, run_b, attn_type, metric, step_idx):
        """Returns (zscore[n_heads, n_blocks], common_blocks)."""
        mat_a, blocks_a = LTXAttentionCompareRuns._extract(run_a, attn_type, metric, step_idx)
        mat_b, blocks_b = LTXAttentionCompareRuns._extract(run_b, attn_type, metric, step_idx)
        common = sorted(set(blocks_a) & set(blocks_b))
        if not common:
            raise ValueError(f"[HeadCandidates] No common blocks for metric '{metric}'.")
        mat_a = mat_a[:, [blocks_a.index(b) for b in common]]
        mat_b = mat_b[:, [blocks_b.index(b) for b in common]]
        mh = min(mat_a.shape[0], mat_b.shape[0])
        mat_a, mat_b = mat_a[:mh], mat_b[:mh]
        std = float(np.concatenate([mat_a.ravel(), mat_b.ravel()]).std()) + 1e-8
        z = (mat_a - mat_b) / std
        return z, common

    @staticmethod
    def _format_group(title, indices, composite, z_stack, metric_list, common_blocks):
        lines = [title]
        for rank, (h, col) in enumerate(indices, start=1):
            blk    = common_blocks[col]
            detail = ", ".join(f"{m}={z_stack[mi, h, col]:+.3f}"
                               for mi, m in enumerate(metric_list))
            lines.append(
                f"#{rank:2d}  block={blk:2d} head={h:2d}  "
                f"composite={composite[h, col]:.3f}  ({detail})"
            )
        return "\n".join(lines)

    def rank(self, store_handle_a, store_handle_b, attn_type, metrics, step_idx,
             top_k, control_mode, control_k, seed):
        reg   = get_registry()
        run_a = LTXAttentionCompareRuns._load_run(store_handle_a, reg)
        run_b = LTXAttentionCompareRuns._load_run(store_handle_b, reg)

        metric_list = list(dict.fromkeys(m.strip() for m in metrics.split(",") if m.strip()))
        if not metric_list:
            raise ValueError("[HeadCandidates] No metrics given.")

        per_metric = {m: self._zscore(run_a, run_b, attn_type, m, step_idx) for m in metric_list}

        # Intersect blocks/heads across all metrics — they should normally
        # agree (same capture config) but this stays correct if they don't.
        common_blocks = sorted(set.intersection(*(set(b) for _, b in per_metric.values())))
        if not common_blocks:
            raise ValueError("[HeadCandidates] No blocks common across all requested metrics.")
        n_heads = min(z.shape[0] for z, _ in per_metric.values())

        z_stack = np.stack([
            per_metric[m][0][:n_heads][:, [per_metric[m][1].index(b) for b in common_blocks]]
            for m in metric_list
        ], axis=0)  # [n_metrics, n_heads, n_blocks]

        composite = np.abs(z_stack).mean(axis=0)  # [n_heads, n_blocks]

        order = np.argsort(composite.ravel())[::-1]  # descending
        n_total = composite.size
        top_k   = min(top_k, n_total)
        cand_flat = order[:top_k]
        cand_idx  = [np.unravel_index(fi, composite.shape) for fi in cand_flat]

        remaining   = order[top_k:]
        control_k   = min(control_k, len(remaining))
        if control_k > 0:
            if control_mode == "random":
                rng = np.random.default_rng(seed)
                ctrl_flat = rng.choice(remaining, size=control_k, replace=False)
            else:  # lowest_score
                ctrl_flat = remaining[-control_k:][::-1]
            ctrl_idx = [np.unravel_index(fi, composite.shape) for fi in ctrl_flat]
        else:
            ctrl_idx = []

        header = (
            f"A = '{store_handle_a}'  |  B = '{store_handle_b}'  |  attn_type = {attn_type}\n"
            f"Metrics: {metric_list}\n"
            f"Composite score = mean(|zscore(A-B)|) across those metrics, per (block, head)\n"
            f"Blocks: {common_blocks}  |  Heads: 0-{n_heads - 1}\n"
        )
        candidates_txt = self._format_group(
            f"=== Candidates: top {top_k} by composite score ===",
            cand_idx, composite, z_stack, metric_list, common_blocks,
        )
        control_txt = self._format_group(
            f"=== Control group: {control_k} ({control_mode}) ===",
            ctrl_idx, composite, z_stack, metric_list, common_blocks,
        ) if ctrl_idx else "=== Control group: (skipped, control_k=0) ==="

        report = f"{header}\n{candidates_txt}\n\n{control_txt}\n"

        candidates_csv = "\n".join(f"{common_blocks[col]},{h}" for h, col in cand_idx)
        control_csv    = "\n".join(f"{common_blocks[col]},{h}" for h, col in ctrl_idx)

        print(f"[HeadCandidates]\n{report}")
        return (report, candidates_csv, control_csv)
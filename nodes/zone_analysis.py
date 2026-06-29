from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..core.stores    import AttentionStore, get_registry
from ..utils.graphics import apply_colormap_batch, add_grid_lines


class LTXAttentionZoneAnalysis:
    """
    For each (block, head), compute the ratio:
        attention_mass_vers_zone / fraction_zone_globale

    ratio > 1  → the head focuses on this zone
    ratio ≈ 1  → uniform attention (no preference)
    ratio < 1  → the head avoids this zone

    The mask uses pixel coordinates (final image) and is
    automatically resized to latent space (/32 for LTX).

    Requires store_full_maps=True in LTXAttentionCaptureSetup.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zone_mask":      ("MASK",),
                "attn_type":      (["sa", "ca"], {"default": "sa"}),
                "step_idx":       ("INT",  {"default": -1, "min": -1, "max": 255,
                                   "tooltip": "-1 = average over all steps."}),
                "num_frames":     ("INT",  {"default": 1,  "min": 1,  "max": 256}),
                "latent_height":  ("INT",  {"default": 16, "min": 1,  "max": 256}),
                "latent_width":   ("INT",  {"default": 16, "min": 1,  "max": 256}),
                "query_mode":     (["key_mass", "query_mass", "both"],
                                  {"default": "key_mass",
                                   "tooltip":
                                       "key_mass   : mass received by zone tokens "
                                                    "(which LOOK AT the zone?)\n"
                                       "query_mass : mass emitted by zone tokens "
                                                    "(from where does the zone LOOK?)\n"
                                       "both       : average of both"}),
                "aggregate_time": ("BOOLEAN", {"default": True,
                                   "tooltip": "True = aggregate over all frames.\n"
                                              "False = analyze only frame 0."}),
                "mask_threshold": ("FLOAT",  {"default": 0.5, "min": 0.0, "max": 1.0,
                                   "step": 0.05,
                                   "tooltip": "Binarization threshold for the latent mask."}),
                "colormap":       (["viridis","inferno","turbo","coolwarm"],
                                  {"default": "viridis"}),
                "cell_size":      ("INT",  {"default": 16, "min": 4, "max": 64}),
                "top_k":          ("INT",  {"default": 10, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("zone_heatmap", "ranked_heads")
    FUNCTION     = "analyze"
    CATEGORY     = "g_raw/LTX/Profiler"

    # ──────────────────────────────────────────────────────────────────────
    # Helpers statiques
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _mask_to_latent_indices(zone_mask: torch.Tensor,
                                 latent_height: int,
                                 latent_width: int,
                                 threshold: float,
                                 num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a pixel mask to latent token indices.

        Args:
            zone_mask : [H_img, W_img] float [0,1]

        Returns:
            zone_indices_spatial : [n_zone]       indices in [0, P-1]
            zone_indices_full    : [T * n_zone]   indices in [0, T*P-1]
        """
        P = latent_height * latent_width

        mask_latent = F.interpolate(
            zone_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(latent_height, latent_width),
            mode="bilinear", align_corners=False,
        ).squeeze()  # [Lh, Lw]

        zone_spatial = (mask_latent > threshold).flatten()          # [P] bool
        zone_indices_spatial = zone_spatial.nonzero(as_tuple=True)[0]  # [n_zone]

        zone_indices_full = torch.cat([
            zone_indices_spatial + f * P for f in range(num_frames)
        ])  # [T * n_zone]

        return zone_indices_spatial, zone_indices_full

    @staticmethod
    def _compute_zone_mass(W: torch.Tensor,
                           head_idx: int,
                           zone_indices: torch.Tensor,
                           query_mode: str) -> float:
        """
        Calculate attention mass related to the zone for a head.

        W : [H, Sq, Sk]

        key_mass   : mean of W[h, :, zone_idx]  → how much queries look at zone
        query_mass : mean of W[h, zone_idx, :]  → where do zone tokens look
        both       : average of the two
        """
        h_map = W[head_idx]   # [Sq, Sk]
        Sq, Sk = h_map.shape

        key_idx   = zone_indices[zone_indices < Sk]
        query_idx = zone_indices[zone_indices < Sq]

        if len(key_idx) == 0 and len(query_idx) == 0:
            return 0.0

        if query_mode == "key_mass":
            if len(key_idx) == 0:
                return 0.0
            return h_map[:, key_idx].mean().item()

        elif query_mode == "query_mass":
            if len(query_idx) == 0:
                return 0.0
            return h_map[query_idx, :].mean().item()

        else:  # both
            km = h_map[:, key_idx].mean().item()   if len(key_idx)   > 0 else 0.0
            qm = h_map[query_idx, :].mean().item() if len(query_idx) > 0 else 0.0
            return (km + qm) / 2.0

    # ──────────────────────────────────────────────────────────────────────
    # Main analysis
    # ──────────────────────────────────────────────────────────────────────

    def analyze(self, zone_mask, attn_type, step_idx, num_frames,
                latent_height, latent_width, query_mode,
                aggregate_time, mask_threshold,
                colormap, cell_size, top_k):

        reg = get_registry()


        h = reg._cur_attn or reg.create('default')


        reg.switch_attn(h)


        store = AttentionStore()
        src   = store.sa if attn_type == "sa" else store.ca

        if not src:
            raise ValueError(f"[ZoneAnalysis] Store {attn_type} is empty.")

        # ── Check that maps exist ────────────────────────────────
        has_maps = any(
            e.get("map") is not None
            for steps in src.values()
            for e in steps.values()
        )
        if not has_maps:
            raise ValueError(
                "[ZoneAnalysis] No complete maps found.\n"
                "Re-run capture with store_full_maps=True."
            )

        P = latent_height * latent_width
        T = num_frames

        # ── Normalize mask input ───────────────────────────────────
        if zone_mask.dim() == 3:
            mask_2d = zone_mask[0]        # [H_img, W_img]
        else:
            mask_2d = zone_mask

        # ── Token indices within the zone ────────────────────────────────
        zone_spatial, zone_full = self._mask_to_latent_indices(
            mask_2d, latent_height, latent_width, mask_threshold, T
        )

        if len(zone_spatial) == 0:
            raise ValueError(
                "[ZoneAnalysis] The mask covers no latent tokens.\n"
                f"Latent resolution: {latent_height}x{latent_width} = {P} tokens.\n"
                f"Lower mask_threshold (currently {mask_threshold})."
            )

        zone_indices = zone_full if aggregate_time else zone_spatial
        n_zone       = len(zone_indices)
        n_total      = T * P if aggregate_time else P
        # Fraction of tokens in the zone (reference "random")
        zone_frac    = n_zone / n_total

        # ── Detect n_heads ──────────────────────────────────────────────
        n_heads = 0
        for steps in src.values():
            for e in steps.values():
                n_heads = len(e.get("entropy", []))
                if n_heads:
                    break
            if n_heads:
                break

        block_indices = sorted(src.keys())
        n_blocks      = len(block_indices)

        # ── Ratio matrix [n_heads, n_blocks] ────────────────────────
        ratio_mat = np.zeros((n_heads, n_blocks), dtype=np.float32)
        count_mat = np.zeros((n_heads, n_blocks), dtype=np.int32)

        for col, blk in enumerate(block_indices):
            steps_data = src[blk]

            target_steps = (sorted(steps_data.keys())
                            if step_idx == -1
                            else ([step_idx] if step_idx in steps_data else []))

            for sk in target_steps:
                entry = steps_data[sk]
                if entry.get("map") is None:
                    continue

                W = entry["map"].float()      # [H, Sq, Sk] fp32 CPU
                H_h, Sq, Sk = W.shape

                # Check geometric compatibility
                expected = T * P if aggregate_time else P
                if Sk < expected or Sq < expected:
                    continue

                for h in range(min(H_h, n_heads)):
                    mass  = self._compute_zone_mass(W, h, zone_indices, query_mode)
                    ratio = mass / (zone_frac + 1e-8)
                    ratio_mat[h, col] += ratio
                    count_mat[h, col] += 1

        # Normalize by number of effective steps
        safe_count = np.maximum(count_mat, 1)
        ratio_mat  = ratio_mat / safe_count

        # ── Heatmap ───────────────────────────────────────────────────────
        display    = ratio_mat.copy()
        mn_d, mx_d = display.min(), display.max()
        if mx_d > mn_d:
            display = (display - mn_d) / (mx_d - mn_d)

        colored   = apply_colormap_batch(display[np.newaxis], colormap)[0]
        out_h     = n_heads  * cell_size
        out_w     = n_blocks * cell_size
        colored_t = (torch.from_numpy(colored)
                     .permute(2, 0, 1).unsqueeze(0).float())
        colored_t = F.interpolate(colored_t, (out_h, out_w), mode="nearest")
        img_np    = colored_t.squeeze(0).permute(1, 2, 0).numpy()
        img_np    = add_grid_lines(img_np, cell_size, n_heads, n_blocks)
        out       = torch.from_numpy(img_np).unsqueeze(0).clamp(0.0, 1.0)

        # ── Ranking ────────────────────────────────────────────────────────
        actual_k = min(top_k, n_heads * n_blocks)
        flat_idx = np.argsort(ratio_mat.ravel())[::-1][:actual_k]

        rank_lines = []
        for i, fi in enumerate(flat_idx):
            b_pos = fi % n_blocks
            h_pos = fi // n_blocks
            rank_lines.append(
                f"  #{i+1:02d} | Block {block_indices[b_pos]:2d} "
                f"Head {h_pos:2d} | ratio={ratio_mat.ravel()[fi]:.3f}"
            )

        # Stats on mask coverage
        mask_pct    = 100.0 * zone_frac
        n_zero_maps = int((count_mat == 0).sum())

        stats = (
            f"Zone: {len(zone_spatial)} tokens / {P} per frame "
            f"({mask_pct:.1f}% of latent space)\n"
            f"Mode  : {query_mode} | "
            f"Temporal aggregation: {aggregate_time} ({T} frames)\n"
            f"Ratio = 1.0 → uniform attention (random baseline)\n"
            f"Ratio > 1.0 → head focused on zone\n"
            f"Blocks without map: {n_zero_maps} / {n_heads * n_blocks}\n"
            f"\nTop-{actual_k} heads focused on zone:\n"
            + "\n".join(rank_lines)
        )

        return (out, stats)
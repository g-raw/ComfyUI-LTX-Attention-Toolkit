from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from ..core.stores    import AttentionStore
from ..utils.graphics import (get_colormap, apply_colormap_batch,
                               add_grid_lines, render_head_grid)
from ..utils.helpers  import resolve_entry, parse_heads, log_node


class LTXAttentionQueryMap:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "attn_type":          (["sa", "ca"], {"default": "sa"}),
            "block_idx":          ("INT",    {"default": 0,  "min": 0,  "max": 47}),
            "step_idx":           ("INT",    {"default": -1, "min": -1, "max": 255}),
            "head_indices":       ("STRING", {"default": "all"}),
            "num_frames":         ("INT",    {"default": 1,  "min": 1,  "max": 256}),
            "latent_height":      ("INT",    {"default": 32, "min": 1,  "max": 256}),
            "latent_width":       ("INT",    {"default": 32, "min": 1,  "max": 256}),
            "key_token_idx":      ("INT",    {"default": -1, "min": -1, "max": 65535}),
            "aggregate_frames":   (["mean", "max", "first"], {"default": "mean"}),
            "colormap":           (["inferno","viridis","magma","plasma",
                                    "hot","turbo","gray"], {"default": "inferno"}),
            "normalize_per_head": ("BOOLEAN", {"default": True}),
            "cell_size":          ("INT",    {"default": 96, "min": 16, "max": 512}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("query_map",)
    FUNCTION     = "visualize"
    CATEGORY     = "g_raw/LTX/Profiler"

    def visualize(self, attn_type, block_idx, step_idx, head_indices,
                  num_frames, latent_height, latent_width,
                  key_token_idx, aggregate_frames, colormap,
                  normalize_per_head, cell_size):

        store = AttentionStore.get()
        src   = store.sa if attn_type == "sa" else store.ca
        entry = resolve_entry(src, block_idx, step_idx, attn_type)

        W               = entry["map"].float()
        H_heads, Sq, Sk = W.shape
        head_list       = parse_heads(head_indices, H_heads)
        n_heads         = len(head_list)
        T, Lh, Lw       = num_frames, latent_height, latent_width

        if Sq != T * Lh * Lw:
            raise ValueError(
                f"[QueryMap] Sq={Sq} ≠ T×Lh×Lw={T*Lh*Lw}. "
                "Vérifie num_frames/latent_height/latent_width."
            )

        if key_token_idx >= 0:
            raw_maps = W[head_list, :, min(key_token_idx, Sk-1)]
        else:
            raw_maps = W[head_list].mean(dim=2)

        maps_4d = raw_maps.view(n_heads, T, Lh, Lw)
        if   aggregate_frames == "mean":  maps = maps_4d.mean(dim=1)
        elif aggregate_frames == "max":   maps = maps_4d.max(dim=1).values
        else:                             maps = maps_4d[:, 0]

        out = render_head_grid(maps, head_list, H_heads, cell_size,
                               colormap, normalize_per_head)
        log_node("QueryMap", attn_type, block_idx, step_idx, entry,
                 n_heads, T, Lh, Lw)
        return (out,)


class LTXAttentionKeyMap:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "block_idx":          ("INT",    {"default": 0,  "min": 0,  "max": 47}),
            "step_idx":           ("INT",    {"default": -1, "min": -1, "max": 255}),
            "head_indices":       ("STRING", {"default": "all"}),
            "num_frames":         ("INT",    {"default": 1,  "min": 1,  "max": 256}),
            "latent_height":      ("INT",    {"default": 32, "min": 1,  "max": 256}),
            "latent_width":       ("INT",    {"default": 32, "min": 1,  "max": 256}),
            "query_token_idx":    ("INT",    {"default": -1, "min": -1, "max": 65535}),
            "aggregate_frames":   (["mean", "max", "first"], {"default": "mean"}),
            "colormap":           (["inferno","viridis","magma","plasma",
                                    "hot","turbo","gray"], {"default": "inferno"}),
            "normalize_per_head": ("BOOLEAN", {"default": True}),
            "cell_size":          ("INT",    {"default": 96, "min": 16, "max": 512}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("key_map",)
    FUNCTION     = "visualize"
    CATEGORY     = "g_raw/LTX/Profiler"

    def visualize(self, block_idx, step_idx, head_indices,
                  num_frames, latent_height, latent_width,
                  query_token_idx, aggregate_frames, colormap,
                  normalize_per_head, cell_size):

        store = AttentionStore.get()
        entry = resolve_entry(store.sa, block_idx, step_idx, "sa")

        W               = entry["map"].float()
        H_heads, Sq, Sk = W.shape
        head_list       = parse_heads(head_indices, H_heads)
        n_heads         = len(head_list)
        T, Lh, Lw       = num_frames, latent_height, latent_width

        if Sk != T * Lh * Lw:
            raise ValueError(
                f"[KeyMap] Sk={Sk} ≠ T×Lh×Lw={T*Lh*Lw}."
            )

        if query_token_idx >= 0:
            raw_maps = W[head_list, min(query_token_idx, Sq-1), :]
        else:
            raw_maps = W[head_list].mean(dim=1)

        maps_4d = raw_maps.view(n_heads, T, Lh, Lw)
        if   aggregate_frames == "mean":  maps = maps_4d.mean(dim=1)
        elif aggregate_frames == "max":   maps = maps_4d.max(dim=1).values
        else:                             maps = maps_4d[:, 0]

        out = render_head_grid(maps, head_list, H_heads, cell_size,
                               colormap, normalize_per_head)
        log_node("KeyMap", "sa", block_idx, step_idx, entry,
                 n_heads, T, Lh, Lw)
        return (out,)


class LTXAttentionMetricsViz:

    METRICS = ["entropy", "temporal", "spatial", "sink"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "attn_type": (["sa", "ca"], {"default": "sa"}),
            "metric":    (cls.METRICS,  {"default": "entropy"}),
            "step_idx":  ("INT", {"default": -1, "min": -1, "max": 255}),
            "colormap":  (["viridis","inferno","magma","plasma",
                           "hot","turbo","coolwarm"], {"default": "viridis"}),
            "cell_size": ("INT", {"default": 16, "min": 4, "max": 64}),
            "normalize": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("heatmap_image", "stats_text")
    FUNCTION     = "visualize"
    CATEGORY     = "g_raw/LTX/Profiler"

    def visualize(self, attn_type, metric, step_idx, colormap, cell_size, normalize):
        store = AttentionStore.get()
        src   = store.sa if attn_type == "sa" else store.ca
        if not src:
            raise ValueError(f"Aucune donnée {attn_type}.")

        block_indices = sorted(src.keys())
        n_blocks      = len(block_indices)
        n_heads       = 0
        for steps in src.values():
            for e in steps.values():
                if metric in e and len(e[metric]) > 0:
                    n_heads = len(e[metric]); break
            if n_heads: break
        if n_heads == 0:
            raise ValueError(f"Métrique '{metric}' introuvable.")

        mat = np.zeros((n_heads, n_blocks), dtype=np.float32)
        for col, blk in enumerate(block_indices):
            steps_data = src[blk]
            if step_idx == -1:
                stack = [v[metric].float() for v in steps_data.values()
                         if metric in v and len(v[metric]) == n_heads]
                vals  = torch.stack(stack).mean(0) if stack else torch.zeros(n_heads)
            else:
                vals = (steps_data[step_idx][metric].float()
                        if step_idx in steps_data and metric in steps_data[step_idx]
                        else torch.zeros(n_heads))
            mat[:, col] = vals.numpy()

        mat_disp = mat.copy()
        if normalize:
            mn, mx = mat_disp.min(), mat_disp.max()
            if mx > mn:
                mat_disp = (mat_disp - mn) / (mx - mn)

        colored   = apply_colormap_batch(mat_disp[np.newaxis], colormap)[0]
        out_h, out_w = n_heads * cell_size, n_blocks * cell_size
        colored_t = (torch.from_numpy(colored).permute(2,0,1).unsqueeze(0).float())
        colored_t = F.interpolate(colored_t, size=(out_h, out_w), mode="nearest")
        img_np    = colored_t.squeeze(0).permute(1,2,0).numpy()
        img_np    = add_grid_lines(img_np, cell_size, n_heads, n_blocks)
        out       = torch.from_numpy(img_np).unsqueeze(0).clamp(0.0, 1.0)

        top_k    = 5
        flat_idx = np.argsort(mat.ravel())[::-1][:top_k]
        top_str  = ", ".join(
            f"b{block_indices[fi % n_blocks]}_h{fi // n_blocks}={mat.ravel()[fi]:.3f}"
            for fi in flat_idx
        )
        stats = (
            f"Métrique: {metric} | Type: {attn_type} | Step: {step_idx}\n"
            f"Moy: {mat.mean():.4f}  Std: {mat.std():.4f}\n"
            f"Top-{top_k}: {top_str}\n"
            f"Blocs: {block_indices}"
        )
        return (out, stats)


class LTXAttentionGridViz:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "map_store":     ("ATTN_MAP_STORE",),
            "view":          (["key_map","query_map","diff"], {"default": "key_map"}),
            "target_blocks": ("STRING", {"default": "all"}),
            "target_heads":  ("STRING", {"default": "all"}),
            "step_idx":      ("INT",    {"default": -1, "min": -1, "max": 255}),
            "frame_mode":    ("STRING", {"default": "avg",
                              "tooltip": "avg / all / sequence / 0,1,2..."}),
            "colormap":      (["inferno","viridis","magma","hot","turbo","gray"],
                             {"default": "inferno"}),
            "upsample":      ("INT",    {"default": 4, "min": 1, "max": 32}),
            "cell_padding":  ("INT",    {"default": 2, "min": 0, "max": 16}),
            "normalize":     (["global","per_cell","per_block","per_head"],
                             {"default": "per_cell"}),
            "draw_labels":   ("BOOLEAN", {"default": True}),
            "latent_frames": ("INT",    {"default": 10, "min": 1, "max": 256}),
            "latent_height": ("INT",    {"default": 11, "min": 1, "max": 256}),
            "latent_width":  ("INT",    {"default": 20, "min": 1, "max": 256}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("grid_image", "size_info")
    FUNCTION     = "visualize"
    CATEGORY     = "g_raw/LTX/Profiler"

    @staticmethod
    def _parse_frame_mode(s: str, n: int):
        s = s.strip().lower()
        if s == "avg":      return "avg",      list(range(n))
        if s == "all":      return "stack",    list(range(n))
        if s == "sequence": return "sequence", list(range(n))
        try:
            idxs = [int(x.strip()) for x in s.split(",") if x.strip()]
            idxs = [i for i in idxs if 0 <= i < n]
            if not idxs:
                raise ValueError(f"Aucun index valide (max={n-1})")
            return ("single" if len(idxs) == 1 else "stack"), idxs
        except ValueError as e:
            raise ValueError(f"frame_mode '{s}' invalide: {e}")

    def visualize(self, map_store, view, target_blocks, target_heads,
                  step_idx, frame_mode, colormap, upsample, cell_padding,
                  normalize, draw_labels, latent_frames, latent_height, latent_width):

        available_blocks = sorted(map_store.keys())
        if not available_blocks:
            raise ValueError("[GridViz] map_store vide.")

        first_steps     = map_store[available_blocks[0]]
        first_step_key  = sorted(first_steps.keys())[0]
        available_heads = sorted(first_steps[first_step_key].keys())

        block_list = (available_blocks if target_blocks.strip().lower() == "all"
                      else [int(x.strip()) for x in target_blocks.split(",")
                            if x.strip() and int(x.strip()) in map_store])
        head_list  = (available_heads if target_heads.strip().lower() == "all"
                      else [int(x.strip()) for x in target_heads.split(",")
                            if x.strip() and int(x.strip()) in available_heads])

        if not block_list: raise ValueError("[GridViz] Aucun bloc valide.")
        if not head_list:  raise ValueError("[GridViz] Aucune tête valide.")

        n_blocks, n_heads = len(block_list), len(head_list)
        H_lat_ref = latent_height
        W_lat_ref = latent_width
        F_ref     = latent_frames

        # ── Extraction ────────────────────────────────────────────────────
        raw = {}
        for h_pos, h_idx in enumerate(head_list):
            raw[h_pos] = {}
            for b_pos, b_idx in enumerate(block_list):
                if b_idx not in map_store:
                    raw[h_pos][b_pos] = np.zeros((1, H_lat_ref, W_lat_ref),
                                                  dtype=np.float32)
                    continue
                steps     = map_store[b_idx]
                step_keys = sorted(steps.keys())
                if step_idx == -1:
                    frames_list = [
                        steps[sk][h_idx][view].float().numpy()
                        for sk in step_keys
                        if h_idx in steps[sk] and view in steps[sk][h_idx]
                    ]
                    data = (np.stack(frames_list).mean(0) if frames_list
                            else np.zeros((1, H_lat_ref, W_lat_ref), dtype=np.float32))
                else:
                    sk   = step_idx if step_idx in steps else step_keys[-1]
                    data = (steps[sk][h_idx][view].float().numpy()
                            if h_idx in steps[sk] and view in steps[sk][h_idx]
                            else np.zeros((1, H_lat_ref, W_lat_ref), dtype=np.float32))

                H_lat_ref = data.shape[1]
                W_lat_ref = data.shape[2]
                F_ref     = max(F_ref, data.shape[0])
                raw[h_pos][b_pos] = data

        render_mode, frame_indices = self._parse_frame_mode(frame_mode, F_ref)

        # ── Normalisation ─────────────────────────────────────────────────
        if normalize == "global":
            all_vals = np.concatenate([raw[h][b].ravel()
                                       for h in range(n_heads)
                                       for b in range(n_blocks)])
            g_min, g_max = float(all_vals.min()), float(all_vals.max())
            def norm_fn(arr, h, b):
                return (arr - g_min) / (g_max - g_min + 1e-8)

        elif normalize == "per_block":
            block_stats = {
                b: (float(np.concatenate([raw[h][b].ravel()
                                          for h in range(n_heads)]).min()),
                    float(np.concatenate([raw[h][b].ravel()
                                          for h in range(n_heads)]).max()))
                for b in range(n_blocks)
            }
            def norm_fn(arr, h, b):
                mn, mx = block_stats[b]
                return (arr - mn) / (mx - mn + 1e-8)

        elif normalize == "per_head":
            head_stats = {
                h: (float(np.concatenate([raw[h][b].ravel()
                                          for b in range(n_blocks)]).min()),
                    float(np.concatenate([raw[h][b].ravel()
                                          for b in range(n_blocks)]).max()))
                for h in range(n_heads)
            }
            def norm_fn(arr, h, b):
                mn, mx = head_stats[h]
                return (arr - mn) / (mx - mn + 1e-8)

        else:  # per_cell
            def norm_fn(arr, h, b):
                mn, mx = float(arr.min()), float(arr.max())
                return (arr - mn) / (mx - mn + 1e-8)

        # ── Rendu ─────────────────────────────────────────────────────────
        lut       = get_colormap(colormap)
        lut_turbo = get_colormap("turbo")
        pad       = cell_padding
        lbl_h     = 20 if draw_labels else 0
        lbl_w     = 24 if draw_labels else 0

        def render_cell(frame_arr: np.ndarray) -> np.ndarray:
            """[H, W] float32 [0,1] → [H*up, W*up, 3] float32"""
            up  = upsample
            img = np.repeat(np.repeat(frame_arr, up, axis=0), up, axis=1)
            return lut[(img * 255).astype(np.uint8)].astype(np.float32)

        def build_grid(frame_sel) -> np.ndarray:
            sample    = frame_sel(raw[0][0])
            stack_h   = 1 if sample.ndim == 2 else sample.shape[0]
            cell_h    = H_lat_ref * upsample * stack_h
            cell_w    = W_lat_ref * upsample
            gh        = lbl_h + n_heads  * (cell_h + pad) + pad
            gw        = lbl_w + n_blocks * (cell_w + pad) + pad
            grid      = np.full((gh, gw, 3), 0.08, dtype=np.float32)

            for h_pos in range(n_heads):
                for b_pos in range(n_blocks):
                    data     = np.clip(norm_fn(raw[h_pos][b_pos].copy(),
                                               h_pos, b_pos), 0.0, 1.0)
                    selected = frame_sel(data)
                    if selected.ndim == 2:
                        cell_img = render_cell(selected)
                    else:
                        cell_img = np.concatenate(
                            [render_cell(selected[i])
                             for i in range(selected.shape[0])], axis=0
                        )
                    y0 = lbl_h + pad + h_pos * (cell_h + pad)
                    x0 = lbl_w + pad + b_pos * (cell_w + pad)
                    grid[y0:y0+cell_h, x0:x0+cell_w] = cell_img

            if draw_labels:
                for b_pos, b_idx in enumerate(block_list):
                    x0 = lbl_w + pad + b_pos * (cell_w + pad)
                    grid[2:lbl_h-2, x0:x0+min(cell_w,20)] = (
                        lut_turbo[int(b_idx/47*255)])
                    grid[lbl_h:, x0:x0+1] = 0.3
                for h_pos, h_idx in enumerate(head_list):
                    y0 = lbl_h + pad + h_pos * (cell_h + pad)
                    grid[y0:y0+min(cell_h,12), 2:lbl_w-2] = (
                        lut_turbo[int(h_idx/31*255)])
                    grid[y0:y0+1, lbl_w:] = 0.3
            return grid

        # ── Sélecteurs ───────────────────────────────────────────────────
        grids = []
        if render_mode == "avg":
            grids.append(build_grid(lambda d: d.mean(axis=0)))
        elif render_mode == "single":
            fi = frame_indices[0]
            grids.append(build_grid(lambda d, i=fi: d[i] if i < d.shape[0] else d[-1]))
        elif render_mode == "stack":
            fi = frame_indices
            grids.append(build_grid(
                lambda d, idx=fi: np.stack(
                    [d[i] if i < d.shape[0] else d[-1] for i in idx]
                )
            ))
        elif render_mode == "sequence":
            for fi in frame_indices:
                grids.append(build_grid(
                    lambda d, i=fi: d[i] if i < d.shape[0] else d[-1]
                ))

        max_h = max(g.shape[0] for g in grids)
        max_w = max(g.shape[1] for g in grids)
        padded = []
        for g in grids:
            if g.shape[0] < max_h or g.shape[1] < max_w:
                p = np.full((max_h, max_w, 3), 0.08, dtype=np.float32)
                p[:g.shape[0], :g.shape[1]] = g
                padded.append(p)
            else:
                padded.append(g)

        out  = torch.from_numpy(np.stack(padded)).clamp(0.0, 1.0)
        info = (
            f"Grille: {n_blocks} blocs × {n_heads} têtes | "
            f"{len(grids)} image(s) | {max_w}×{max_h}px"
        )
        return (out, info)
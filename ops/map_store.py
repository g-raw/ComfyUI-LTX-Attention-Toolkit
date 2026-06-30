from __future__ import annotations
import torch

from ..core.stores   import _AttnInst
from ..utils.helpers import reset_call_count


# ── Entry format helper ──────────────────────────────────────────────────────

def _ms_entry_to_dict(key_map, query_map, timestep, full=None):
    """Convert a single head's reduced maps into the same entry shape used by SetupCapture."""
    entry = {
        "key_map":   key_map.float(),
        "query_map": query_map.float(),
        "timestep":  float(timestep),
    }
    if full is not None:
        entry["full"] = full.float()
    return entry


# ── Callback factory (writes to map_data + optionally to registry store) ─────

def make_map_store_callback(
    map_data: dict,
    parsed_heads,
    store_mode: str,
    full_block_set: set,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    target_inst: "_AttnInst | None" = None,   # ← write to registry store too
):
    """
    Returns the callback store_map(attn_map, block_idx, head_idx_arg,
                                    step_idx, timestep, num_frames)
    which writes into *map_data* and, when *target_inst* is provided,
    also records an entry in target_inst.sa/attn_type matching SetupCapture's format.
    """
    P = latent_height * latent_width

    def store_map(attn_map: torch.Tensor, block_idx: int,
                  head_idx_arg: int, step_idx: int,
                  timestep: float, num_frames: int):

        W_all = attn_map.detach().contiguous().cpu()
        H_heads = W_all.shape[0]

        # ── Sanity check: skip invalid maps silently ────────────────────────
        if W_all.numel() == 0 or not torch.isfinite(W_all).all():
            return

        head_range = (
            range(H_heads) if parsed_heads is None
            else [h for h in sorted(parsed_heads) if h < H_heads]
        ) if head_idx_arg == -1 else (
            [head_idx_arg] if head_idx_arg < H_heads else []
        )

        n_frames_actual = num_frames if num_frames > 0 else latent_frames

        def to_spatial(vec):
            total = vec.shape[0]
            if total == n_frames_actual * latent_height * latent_width:
                return vec.view(n_frames_actual, latent_height, latent_width)
            if total % P == 0:
                return vec.view(total // P, latent_height, latent_width)
            return vec.view(1, 1, -1)

        use_full = (
            store_mode == "full_fp16" or
            (store_mode == "hybrid" and block_idx in full_block_set)
        )

        for h_idx in head_range:
            W         = W_all[h_idx]
            key_map   = to_spatial(W.mean(dim=0).float())
            query_map = to_spatial(W.mean(dim=1).float())
            entry     = _ms_entry_to_dict(key_map, query_map, timestep)
            if use_full:
                entry["full"] = W.float()

            # 1 ── Write into caller's local map_data (backward compat) ───────
            map_data.setdefault(block_idx, {}) \
                    .setdefault(step_idx, {})[h_idx] = entry

            # 2 ── Also write to registry store so visualizers read it via handle ──
            if target_inst is not None:
                sa_key   = f"mapstore_{block_idx}"
                ms_step  = target_inst.get_ms_step_counter(sa_key)

                store_dict = target_inst.sa     # MapStore uses sa for both types
                if block_idx not in store_dict:
                    store_dict[block_idx] = {}
                entry_reg = {
                    "map":       W.half(),       # full attention map (same as SetupCapture)
                    "key_map":   key_map,
                    "query_map": query_map,
                    "entropy":   torch.zeros(H_heads),  # dummy — entropy not computed by MapStore
                    "timestep":  float(timestep),
                    "step_idx":  ms_step,
                }
                store_dict[block_idx][ms_step] = entry_reg

    return store_map


# ── Selective forward builder (used by map_store_node.py) ─────────────────────

def make_map_store_patched_forward(
    original_forward,
    parsed_blocks: set[int],
    parsed_steps,
    target_call_n: int,
    store_map_cb,
    parsed_heads,
    step_counters: dict[int, int],
    num_frames_ref: list[float],
    target_inst: "_AttnInst | None",         # ← same instance used by callback
):
    """Build a patched _forward that injects MapStore metadata per block."""

    def patched_forward(self_dm, x, timestep, context, attention_mask,
                        frame_rate=25, transformer_options={},
                        keyframe_idxs=None, **kwargs):

        vx = x[0] if isinstance(x, (list, tuple)) else x
        if vx.dim() == 5:
            _, _, F_lat, H_lat, W_lat = vx.shape
            num_frames_ref[0] = F_lat

        ts_val = float(
            (timestep[0] if isinstance(timestep, (list, tuple))
             else timestep).mean().item()
        )

        patches_replace = dict(transformer_options.get("patches_replace", {}))
        dit_replace     = dict(patches_replace.get("dit", {}))

        for blk_idx in parsed_blocks:
            if blk_idx not in step_counters:
                step_counters[blk_idx] = 0
            step_idx               = step_counters[blk_idx]
            step_counters[blk_idx] += 1

            if parsed_steps is not None and step_idx not in parsed_steps:
                continue

            existing = dit_replace.get(("double_block", blk_idx))

            def make_hook(bidx, sidx, ts, n_frames, existing_h):
                def hook(args, orig):
                    to = dict(args.get("transformer_options", {}))
                    to["_ms_block_idx"]    = bidx
                    to["_ms_step_idx"]     = sidx
                    to["_ms_timestep"]     = ts
                    to["_ms_n_frames"]     = n_frames
                    to["_ms_target_call"]  = target_call_n
                    to["_ms_parsed_heads"] = parsed_heads
                    to["_ms_store_cb"]     = store_map_cb
                    reset_call_count(bidx)
                    new_args = {**args, "transformer_options": to}
                    if existing_h and not getattr(existing_h, "_is_ms_hook", False):
                        return existing_h(new_args, orig)
                    return orig["original_block"](new_args)
                hook._is_ms_hook   = True
                hook._is_profiler_hook = False       # non-profiler patch — preserved on unwrap
                return hook

            dit_replace[("double_block", blk_idx)] = make_hook(
                blk_idx, step_idx, ts_val, num_frames_ref[0], existing
            )

        patches_replace["dit"] = dit_replace
        transformer_options    = {**transformer_options,
                                  "patches_replace": patches_replace}
        return original_forward(
            x, timestep, context, attention_mask,
            frame_rate, transformer_options, keyframe_idxs, **kwargs,
        )

    return patched_forward

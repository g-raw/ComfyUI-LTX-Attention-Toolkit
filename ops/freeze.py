from __future__ import annotations
import warnings

import torch
import torch.nn.functional as F

from ..utils.helpers import reset_call_count


def make_freeze_hook_factory(block_configs: dict, freeze_from_step: int,
                             blend_weight: float):
    """Returns a make_hook(block_idx, existing_hook) -> hook callable, for
    register_layer(). block_configs: {block_idx: [{"head_idx": int,
    "frozen_map": Tensor}, ...]} -- one block can freeze several heads in
    a single hook/attention call. Tracks its own per-block step counter
    (fresh each time this factory is built, i.e. fresh each node run) so
    the freeze only activates once current_step >= freeze_from_step;
    before that, returns None so the call falls through to whatever
    earlier layers/the original block already contribute."""
    step_counters: dict = {}

    def make_hook(block_idx, existing_hook):
        freeze_configs = block_configs.get(block_idx)
        if freeze_configs is None:
            return None

        current_step             = step_counters.get(block_idx, 0)
        step_counters[block_idx] = current_step + 1
        if current_step < freeze_from_step:
            return None

        def hook(args: dict, orig: dict):
            to = dict(args.get("transformer_options", {}))
            to["_freeze_block_idx"] = block_idx
            to["_freeze_configs"]   = freeze_configs
            to["_freeze_blend"]     = blend_weight
            reset_call_count(block_idx)
            new_args = {**args, "transformer_options": to}
            if existing_hook is not None:
                return existing_hook(new_args, orig)
            return orig["original_block"](new_args)
        return hook

    return make_hook


def apply_head_freeze(q, k, v, heads, freeze_configs,
                      blend_weight, original_fn, extra_args, extra_kwargs,
                      attn_precision, transformer_options):
    B, Sq, HD = q.shape
    if HD % heads != 0:
        return original_fn(q, k, v, heads, *extra_args,
                           attn_precision=attn_precision,
                           transformer_options=transformer_options,
                           **extra_kwargs)

    D_head     = HD // heads
    out_normal = original_fn(q, k, v, heads, *extra_args,
                             attn_precision=attn_precision,
                             transformer_options=transformer_options,
                             **extra_kwargs)
    if blend_weight == 0.0:
        return out_normal

    out_mod = out_normal.clone()

    with torch.no_grad():
        v_ = v.view(B, -1, heads, D_head).permute(0, 2, 1, 3).float()

        for cfg in freeze_configs:
            head_idx   = cfg["head_idx"]
            frozen_map = cfg["frozen_map"]
            v_h        = v_[:, head_idx]                     # [B, Sk, D]
            fm         = frozen_map.to(device=v.device, dtype=torch.float32)

            if fm.shape[0] != Sq or fm.shape[1] != v_.shape[2]:
                orig_h, orig_w = fm.shape[0], fm.shape[1]
                if abs(Sq / max(1, orig_h)) > 2.0 or abs(v_.shape[2] / max(1, orig_w)) > 2.0:
                    warnings.warn(f"Frozen attention map resized from ({orig_h}, {orig_w}) "
                                  f"to ({Sq}, {v_.shape[2]}). "
                                  f"Ratio: {Sq/max(1,orig_h):.1f}x — may produce artifacts.")
                fm = F.interpolate(
                    fm.unsqueeze(0).unsqueeze(0),
                    size=(Sq, v_.shape[2]),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)
                fm = fm / (fm.sum(dim=-1, keepdim=True) + 1e-8)

            frozen_out = torch.bmm(
                fm.unsqueeze(0).expand(B, -1, -1), v_h
            ).to(dtype=out_normal.dtype)                     # [B, Sq, D]

            h_start = head_idx * D_head
            h_end   = h_start + D_head

            if blend_weight == 1.0:
                out_mod[:, :, h_start:h_end] = frozen_out
            else:
                out_mod[:, :, h_start:h_end] = (
                    blend_weight         * frozen_out +
                    (1.0 - blend_weight) * out_normal[:, :, h_start:h_end]
                )

    return out_mod
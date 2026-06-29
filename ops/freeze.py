from __future__ import annotations
import warnings

import torch
import torch.nn.functional as F

from ..utils.helpers import reset_call_count


def make_freeze_hook(block_idx: int, head_idx: int,
                     frozen_map: torch.Tensor, blend_weight: float,
                     existing_hook=None):
    def hook(args: dict, orig: dict):
        to = dict(args.get("transformer_options", {}))
        to["_freeze_block_idx"] = block_idx
        to["_freeze_head_idx"]  = head_idx
        to["_freeze_map"]       = frozen_map
        to["_freeze_blend"]     = blend_weight
        reset_call_count(block_idx)
        new_args = {**args, "transformer_options": to}
        if existing_hook is not None:
            return existing_hook(new_args, orig)
        return orig["original_block"](new_args)
    hook._is_freeze_hook = True
    return hook


def apply_head_freeze(q, k, v, heads, freeze_head_idx,
                      frozen_map, blend_weight,
                      original_fn, extra_args, extra_kwargs,
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

    with torch.no_grad():
        v_   = v.view(B, -1, heads, D_head).permute(0, 2, 1, 3).float()
        v_h  = v_[:, freeze_head_idx]                        # [B, Sk, D]
        fm   = frozen_map.to(device=v.device, dtype=torch.float32)

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
        ).to(dtype=out_normal.dtype)                         # [B, Sq, D]

    out_mod = out_normal.clone()
    h_start = freeze_head_idx * D_head
    h_end   = h_start + D_head

    if blend_weight == 1.0:
        out_mod[:, :, h_start:h_end] = frozen_out
    else:
        out_mod[:, :, h_start:h_end] = (
            blend_weight         * frozen_out +
            (1.0 - blend_weight) * out_normal[:, :, h_start:h_end]
        )
    return out_mod
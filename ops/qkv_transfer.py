from __future__ import annotations
import math
import warnings

import torch
import torch.nn.functional as F

from ..utils.helpers import get_call_counts


def _resample_kv(k_src, v_src, target_len):
    if k_src.shape[0] == target_len:
        return k_src, v_src
    ratio = target_len / k_src.shape[0]
    if abs(ratio - 1.0) > 1.0:  # ratio > 2x or < 0.5x
        warnings.warn(f"QKV resampling ratio {ratio:.2f}x is large — results may be inaccurate.")
    def _resamp(t):
        return F.interpolate(
            t.T.unsqueeze(0), size=target_len,
            mode="linear", align_corners=False,
        ).squeeze(0).T
    return _resamp(k_src), _resamp(v_src)


def apply_qkv_transfer(q, k, v, heads, cfg,
                        original_fn, extra_args, extra_kwargs,
                        attn_precision, transformer_options):

    head_configs  = cfg.get("head_configs", [])
    qkv_store     = cfg.get("qkv_store")
    attn_type     = cfg.get("attn_type", "sa")
    target_call_n = cfg.get("target_call_n", 0)
    block_idx     = cfg.get("block_idx")
    step_range    = cfg.get("step_range", (0, 9999))
    current_step  = cfg.get("current_step", 0)

    call_n = get_call_counts().get(f"block_{block_idx}", 1) - 1
    if call_n != target_call_n:
        return original_fn(q, k, v, heads, *extra_args,
                           attn_precision=attn_precision,
                           transformer_options=transformer_options,
                           **extra_kwargs)

    if not (step_range[0] <= current_step <= step_range[1]):
        return original_fn(q, k, v, heads, *extra_args,
                           attn_precision=attn_precision,
                           transformer_options=transformer_options,
                           **extra_kwargs)

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
    out_mod    = out_normal.clone()

    def to_hsd(t):
        b, s, hd = t.shape
        return t[0].view(s, heads, hd // heads).permute(1, 0, 2).float()

    q_hsd = to_hsd(q)
    k_hsd = to_hsd(k)
    v_hsd = to_hsd(v)

    with torch.no_grad():
        for hcfg in head_configs:
            h_idx      = hcfg["head_idx"]
            src_step   = hcfg["source_step"]
            blend      = hcfg["blend"]
            use_map    = hcfg["use_map"]
            use_q      = hcfg["use_q"]
            use_k      = hcfg["use_k"]
            use_v      = hcfg["use_v"]
            sim_filter = hcfg["sim_filter"]
            sim_thresh = hcfg["sim_threshold"]

            if blend == 0.0:
                continue

            h_start = h_idx * D_head
            h_end   = h_start + D_head

            if use_map:
                from ..core.stores import AttentionStore, get_registry
                reg = get_registry()
                h = reg._cur_attn or reg.create('default')
                reg.switch_attn(h)
                attn_store = AttentionStore()
                try:
                    src = attn_store.sa if attn_type == "sa" else attn_store.ca
                    fm  = src[block_idx][src_step]["map"][h_idx].float()
                    fm  = fm.to(device=q.device)
                    if fm.shape[0] != Sq or fm.shape[1] != Sq:
                        fm = F.interpolate(
                            fm.unsqueeze(0).unsqueeze(0),
                            size=(Sq, Sq),
                            mode="bilinear", align_corners=False,
                        ).squeeze()
                        fm = fm / (fm.sum(-1, keepdim=True) + 1e-8)
                    result = fm @ v_hsd[h_idx]
                except (KeyError, TypeError):
                    continue
            else:
                qkv_src = qkv_store.get_qkv(attn_type, block_idx, src_step, h_idx)
                if qkv_src is None:
                    continue
                q_src, k_src, v_src = [t.to(device=q.device) for t in qkv_src]
                k_src, v_src = _resample_kv(k_src, v_src, Sq)
                if use_q and q_src.shape[0] != Sq:
                    q_ratio = Sq / q_src.shape[0]
                    if abs(q_ratio - 1.0) > 1.0:
                        warnings.warn(f"Q interpolation ratio {q_ratio:.2f}x is large — results may be inaccurate.")
                    q_src = F.interpolate(
                        q_src.T.unsqueeze(0), size=Sq,
                        mode="linear", align_corners=False,
                    ).squeeze(0).T

                scale  = 1.0 / math.sqrt(D_head)
                q_eff  = q_src if use_q else q_hsd[h_idx]
                k_eff  = k_src if use_k else k_hsd[h_idx]
                v_eff  = v_src if use_v else v_hsd[h_idx]
                scores = (q_eff @ k_eff.T) * scale
                result = F.softmax(scores, dim=-1) @ v_eff

            result   = result.to(dtype=out_normal.dtype)
            normal_h = out_normal[:, :, h_start:h_end]
            expanded = result.unsqueeze(0).expand(B, -1, -1)

            if sim_filter and not use_map and q_src.shape[0] == Sq:
                cos_sim  = F.cosine_similarity(
                    q_hsd[h_idx], q_src.to(device=q.device), dim=-1
                )
                sim_mask = (cos_sim > sim_thresh).float().view(1, Sq, 1)
                sim_mask = sim_mask.to(dtype=out_normal.dtype)
                out_mod[:, :, h_start:h_end] = (
                    blend * sim_mask * expanded +
                    (1.0 - blend * sim_mask) * normal_h
                )
            else:
                out_mod[:, :, h_start:h_end] = (
                    blend * expanded + (1.0 - blend) * normal_h
                )

    return out_mod
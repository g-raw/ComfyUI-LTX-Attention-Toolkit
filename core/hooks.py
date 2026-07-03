from __future__ import annotations
import math
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("ltx_profiler")

from ..core.stores   import (
    AttentionStore, QKVStore, get_current_attn, get_current_qkv, get_registry)
from ..utils.helpers import increment_call_count, get_call_counts

_ORIGINAL_OPT_ATTN        = None
_ORIGINAL_OPT_ATTN_MASKED = None
_HOOK_ACTIVE              = False


def _compute_attn_map(q: torch.Tensor, k: torch.Tensor,
                      heads: int) -> torch.Tensor:
    """
    Computes softmax(QK^T/sqrt(d)) chunked over heads.
    q, k : [B, S, H*D_head]
    Returns [H, Sq, Sk] fp16.
    """
    B, Sq, HD = q.shape
    D_head    = HD // heads
    H_heads   = heads

    def to_bhsd(t):
        b, s, hd = t.shape
        return t.view(b, s, heads, hd // heads).permute(0, 2, 1, 3).float()

    with torch.no_grad():
        q_ = to_bhsd(q.detach())
        k_ = to_bhsd(k.detach())
        Sk   = k_.shape[2]
        scale = 1.0 / math.sqrt(D_head)
        chunk = 4
        result = torch.empty(H_heads, Sq, Sk, dtype=torch.float16, device=q.device)
        for h0 in range(0, heads, chunk):
            h1   = min(h0 + chunk, heads)
            qc   = q_[:1, h0:h1]
            kc   = k_[:1, h0:h1]
            sc   = torch.einsum("bhsd,bhkd->bhsk", qc, kc) * scale
            wc   = F.softmax(sc, dim=-1).half()
            result[h0:h1] = wc[0]
            del qc, kc, sc, wc
        return result  # [H, Sq, Sk]


def _make_full_hook(original_fn):
    """
    Universal hook — attached to optimized_attention and optimized_attention_masked.
    Manages in order:
      1. Profiling      → AttentionStore
      2. QKV Capture    → QKVStore
      3. QKV Transfer   → substitution Q/K/V
      4. QKV Multiplier → per-head Q/K/V/O scaling
      5. Head Freeze    → frozen map
      6. Normal         → delegation
    """

    def hooked(q, k, v, heads, *args,
               attn_precision=None, transformer_options=None, **kwargs):

        from ..ops.freeze        import apply_head_freeze
        from ..ops.qkv_transfer  import apply_qkv_transfer
        from ..ops.qkv_multiply  import apply_qkv_multiply

        to = transformer_options or {}

        block_idx = to.get("_profiler_block_idx",
                    to.get("_freeze_block_idx",
                    to.get("_qkv_block_idx", None)))

        if block_idx is None:
            return original_fn(q, k, v, heads, *args,
                               attn_precision=attn_precision,
                               transformer_options=transformer_options,
                               **kwargs)

        call_n = increment_call_count(block_idx)
        is_sa  = (call_n == 0)
        is_ca  = (call_n == 1)

        # ── Pre-compute attention map once if any op might need it ─────────
        needs_attn_map = to.get("_profiler_block_idx") is not None
        computed_attn_map = None

        if needs_attn_map:
            try:
                B, Sq, HD = q.shape
                if HD % heads == 0:
                    computed_attn_map = _compute_attn_map(q, k, heads)
                    to["_computed_attn_map"] = computed_attn_map
            except Exception as e:
                logger.error("[AttnMap] pre-compute error b=%d: %s", block_idx, e, exc_info=True)

        # ── 1. Profiling ────────────────────────────────────────────────────
        if to.get("_profiler_block_idx") is not None:
            store = AttentionStore()
            if store.cfg:
                capture_sa = to.get("_profiler_capture_sa", True)
                capture_ca = to.get("_profiler_capture_ca", True)
                if (is_sa and capture_sa) or (is_ca and capture_ca):
                    try:
                        attn_map = computed_attn_map
                        if attn_map is None:
                            B, Sq, HD = q.shape
                            if HD % heads == 0:
                                attn_map = _compute_attn_map(q, k, heads)
                        store.record(
                            attn_type         = "sa" if is_sa else "ca",
                            block_idx         = block_idx,
                            timestep          = to.get("_profiler_timestep", -1.0),
                            attn_weights      = attn_map,
                            num_frames        = to.get("_profiler_num_frames", 1),
                            patches_per_frame = to.get("_profiler_patches_per_frame", 1),
                            latent_h          = to.get("_profiler_latent_h", 1),
                            latent_w          = to.get("_profiler_latent_w", 1),
                        )
                    except Exception as e:
                        logger.error("[Profiling] block b=%d: %s", block_idx, e, exc_info=True)

        # Cleanup pre-computed attn map reference
        if computed_attn_map is not None:
            del computed_attn_map
            to.pop("_computed_attn_map", None)

        # ── 2. QKV Capture ───────────────────────────────────────────────────
        if to.get("_qkv_capture_active"):
            capture_sa = to.get("_qkv_capture_sa", True)
            capture_ca = to.get("_qkv_capture_ca", False)
            if (is_sa and capture_sa) or (is_ca and capture_ca):
                qkv_store = QKVStore()
                if qkv_store.cfg:
                    try:
                        qkv_store.record(
                            "sa" if is_sa else "ca",
                            block_idx,
                            to.get("_qkv_timestep", -1.0),
                            q, k, v, heads,
                        )
                    except Exception as e:
                        logger.error("[QKV] block b=%d: %s", block_idx, e, exc_info=True)

        # ── 3. QKV Transfer ──────────────────────────────────────────────────
        if to.get("_qkv_transfer_active") and is_sa:
            return apply_qkv_transfer(
                q, k, v, heads,
                to.get("_qkv_transfer_cfg", {}),
                original_fn, args, kwargs,
                attn_precision, transformer_options,
            )

        # ── 4. QKV Multiplier ───────────────────────────────────────────────
        if to.get("_qkvmul_active"):
            mcfg      = to.get("_qkvmul_cfg", {})
            apply_sa  = mcfg.get("apply_sa", True)
            apply_ca  = mcfg.get("apply_ca", True)
            if (is_sa and apply_sa) or (is_ca and apply_ca):
                return apply_qkv_multiply(
                    q, k, v, heads, mcfg,
                    original_fn, args, kwargs,
                    attn_precision, transformer_options,
                )

        # ── 5. Head Freeze ───────────────────────────────────────────────────
        if to.get("_freeze_configs") and is_sa:
            return apply_head_freeze(
                q, k, v, heads,
                to["_freeze_configs"],
                to.get("_freeze_blend", 1.0),
                original_fn, args, kwargs,
                attn_precision, transformer_options,
            )

        # ── 6. Normal ────────────────────────────────────────────────────────
        return original_fn(q, k, v, heads, *args,
                           attn_precision=attn_precision,
                           transformer_options=transformer_options,
                           **kwargs)

    return hooked


def install_hook():
    global _ORIGINAL_OPT_ATTN, _ORIGINAL_OPT_ATTN_MASKED, _HOOK_ACTIVE
    if _HOOK_ACTIVE:
        return
    import comfy.ldm.modules.attention as _ca
    _ORIGINAL_OPT_ATTN        = _ca.optimized_attention
    _ORIGINAL_OPT_ATTN_MASKED = _ca.optimized_attention_masked
    _ca.optimized_attention        = _make_full_hook(_ORIGINAL_OPT_ATTN)
    _ca.optimized_attention_masked = _make_full_hook(_ORIGINAL_OPT_ATTN_MASKED)
    _HOOK_ACTIVE = True
    print("[LTXProfiler] Hooks installed.")


def uninstall_hook():
    global _ORIGINAL_OPT_ATTN, _ORIGINAL_OPT_ATTN_MASKED, _HOOK_ACTIVE
    if not _HOOK_ACTIVE:
        return
    import comfy.ldm.modules.attention as _ca
    _ca.optimized_attention        = _ORIGINAL_OPT_ATTN
    _ca.optimized_attention_masked = _ORIGINAL_OPT_ATTN_MASKED
    _HOOK_ACTIVE = False
    print("[LTXProfiler] Hooks removed.")
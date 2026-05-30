from __future__ import annotations
import math

import torch
import torch.nn.functional as F

from ..core.stores   import AttentionStore, QKVStore
from ..utils.helpers import increment_call_count, get_call_counts

_ORIGINAL_OPT_ATTN        = None
_ORIGINAL_OPT_ATTN_MASKED = None
_HOOK_ACTIVE              = False


def _compute_attn_map(q: torch.Tensor, k: torch.Tensor,
                      heads: int) -> torch.Tensor:
    """
    Calcule softmax(QK^T/√d) chunked sur les têtes.
    q, k : [B, S, H*D_head]
    Retourne [H, Sq, Sk] fp16.
    """
    B, Sq, HD = q.shape
    D_head    = HD // heads

    def to_bhsd(t):
        b, s, hd = t.shape
        return t.view(b, s, heads, hd // heads).permute(0, 2, 1, 3).float()

    with torch.no_grad():
        q_ = to_bhsd(q.detach())
        k_ = to_bhsd(k.detach())
        scale = 1.0 / math.sqrt(D_head)
        chunk = 4
        maps  = []
        for h0 in range(0, heads, chunk):
            h1  = min(h0 + chunk, heads)
            qc  = q_[:1, h0:h1]
            kc  = k_[:1, h0:h1]
            sc  = torch.einsum("bhsd,bhkd->bhsk", qc, kc) * scale
            wc  = F.softmax(sc, dim=-1).half()
            maps.append(wc[0])
            del qc, kc, sc, wc
        return torch.cat(maps, dim=0)  # [H, Sq, Sk]


def _make_full_hook(original_fn):
    """
    Hook universel — branché sur optimized_attention et optimized_attention_masked.
    Gère dans l'ordre :
      1. Profiling classique     → AttentionStore
      2. MapStore callback       → map_data via _ms_store_cb
      3. QKV Capture             → QKVStore
      4. QKV Transfer            → substitution Q/K/V
      5. Head Freeze             → map gelée
      6. Normal                  → délégation
    """

    def hooked(q, k, v, heads, *args,
               attn_precision=None, transformer_options=None, **kwargs):

        from ..ops.freeze       import apply_head_freeze
        from ..ops.qkv_transfer import apply_qkv_transfer

        to = transformer_options or {}

        block_idx = to.get("_profiler_block_idx",
                    to.get("_freeze_block_idx",
                    to.get("_qkv_block_idx",
                    to.get("_ms_block_idx", None))))

        if block_idx is None:
            return original_fn(q, k, v, heads, *args,
                               attn_precision=attn_precision,
                               transformer_options=transformer_options,
                               **kwargs)

        call_n = increment_call_count(block_idx)
        is_sa  = (call_n == 0)
        is_ca  = (call_n == 1)

        # ── 1. Profiling ────────────────────────────────────────────────────
        if to.get("_profiler_block_idx") is not None:
            store = AttentionStore.get()
            if store.cfg:
                capture_sa = to.get("_profiler_capture_sa", True)
                capture_ca = to.get("_profiler_capture_ca", True)
                if (is_sa and capture_sa) or (is_ca and capture_ca):
                    try:
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
                            )
                            del attn_map
                    except Exception as e:
                        print(f"[LTXProfiler] profiling error b={block_idx}: {e}")

        # ── 2. MapStore callback ─────────────────────────────────────────────
        ms_cb          = to.get("_ms_store_cb")
        target_call_ms = to.get("_ms_target_call", 0)

        if ms_cb is not None and call_n == target_call_ms:
            try:
                B, Sq, HD = q.shape
                if HD % heads == 0:
                    attn_map = _compute_attn_map(q, k, heads)
                    ms_cb(
                        attn_map,
                        block_idx,
                        -1,                           # head_idx=-1 = tous
                        to.get("_ms_step_idx",   0),
                        to.get("_ms_timestep",   0.0),
                        to.get("_ms_n_frames",   1),
                    )
                    del attn_map
            except Exception as e:
                import traceback
                print(f"[LTXProfiler/MapStore] error b={block_idx}: {e}")
                traceback.print_exc()

        # ── 3. QKV Capture ───────────────────────────────────────────────────
        if to.get("_qkv_capture_active"):
            capture_sa = to.get("_qkv_capture_sa", True)
            capture_ca = to.get("_qkv_capture_ca", False)
            if (is_sa and capture_sa) or (is_ca and capture_ca):
                qkv_store = QKVStore.get()
                if qkv_store.cfg:
                    try:
                        qkv_store.record(
                            "sa" if is_sa else "ca",
                            block_idx,
                            to.get("_qkv_timestep", -1.0),
                            q, k, v, heads,
                        )
                    except Exception as e:
                        print(f"[LTXProfiler/QKV] capture error: {e}")

        # ── 4. QKV Transfer ──────────────────────────────────────────────────
        if to.get("_qkv_transfer_active") and is_sa:
            return apply_qkv_transfer(
                q, k, v, heads,
                to.get("_qkv_transfer_cfg", {}),
                original_fn, args, kwargs,
                attn_precision, transformer_options,
            )

        # ── 5. Head Freeze ───────────────────────────────────────────────────
        if to.get("_freeze_head_idx") is not None and is_sa:
            return apply_head_freeze(
                q, k, v, heads,
                to["_freeze_head_idx"],
                to["_freeze_map"],
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
    print("[LTXProfiler] Hooks installés.")


def uninstall_hook():
    global _ORIGINAL_OPT_ATTN, _ORIGINAL_OPT_ATTN_MASKED, _HOOK_ACTIVE
    if not _HOOK_ACTIVE:
        return
    import comfy.ldm.modules.attention as _ca
    _ca.optimized_attention        = _ORIGINAL_OPT_ATTN
    _ca.optimized_attention_masked = _ORIGINAL_OPT_ATTN_MASKED
    _HOOK_ACTIVE = False
    print("[LTXProfiler] Hooks retirés.")
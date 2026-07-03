from __future__ import annotations


def apply_qkv_multiply(q, k, v, heads, cfg,
                        original_fn, extra_args, extra_kwargs,
                        attn_precision, transformer_options):
    """Scale Q/K/V per head before the attention call, and/or scale the
    per-head slice of the (pre-output-projection) attention output
    afterward. Q/K scaling changes attention sharpness (it rescales the
    softmax logits), not magnitude -- a head with q_mult=k_mult=0 still
    contributes via uniform-attention-weighted V, it isn't ablated. V or
    O scaling directly controls how much the head contributes to the
    residual stream, so o_mult/v_mult=0 is what actually "kills" a head.
    """
    head_configs = cfg.get("head_configs", [])
    B, Sq, HD = q.shape
    if HD % heads != 0 or not head_configs:
        return original_fn(q, k, v, heads, *extra_args,
                           attn_precision=attn_precision,
                           transformer_options=transformer_options,
                           **extra_kwargs)

    D_head = HD // heads

    need_qkv = any(c["q_mult"] != 1.0 or c["k_mult"] != 1.0 or c["v_mult"] != 1.0
                   for c in head_configs)
    need_o   = any(c["o_mult"] != 1.0 for c in head_configs)

    q_mod, k_mod, v_mod = q, k, v
    if need_qkv:
        q_mod, k_mod, v_mod = q.clone(), k.clone(), v.clone()
        for c in head_configs:
            h0 = c["head_idx"] * D_head
            h1 = h0 + D_head
            if c["q_mult"] != 1.0:
                q_mod[:, :, h0:h1] *= c["q_mult"]
            if c["k_mult"] != 1.0:
                k_mod[:, :, h0:h1] *= c["k_mult"]
            if c["v_mult"] != 1.0:
                v_mod[:, :, h0:h1] *= c["v_mult"]

    out = original_fn(q_mod, k_mod, v_mod, heads, *extra_args,
                      attn_precision=attn_precision,
                      transformer_options=transformer_options,
                      **extra_kwargs)

    if need_o:
        out = out.clone()
        for c in head_configs:
            if c["o_mult"] != 1.0:
                h0 = c["head_idx"] * D_head
                h1 = h0 + D_head
                out[:, :, h0:h1] *= c["o_mult"]

    return out

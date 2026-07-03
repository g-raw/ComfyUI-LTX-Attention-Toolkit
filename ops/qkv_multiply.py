from __future__ import annotations


def apply_qkv_multiply(q, k, v, heads, cfg,
                        original_fn, extra_args, extra_kwargs,
                        attn_precision, transformer_options):
    """Scale Q per head before the attention call (qk_mult), and/or scale
    V per head before it too (vo_mult). Only two independent knobs:
    scaling Q and K separately would just multiply through as
    (q_mult * k_mult) on the softmax logits since both are uniform
    per-head scalars, so a single qk_mult applied to Q alone has the
    identical effect. Same reasoning for V/O -- attn_weights @ V is
    linear in V, so scaling V before the matmul or scaling the output
    after gives the identical result; scaling V here (like qk_mult
    scales Q) avoids cloning the output tensor afterward.

    qk_mult changes attention sharpness (rescales the softmax logits),
    not magnitude -- a head with qk_mult=0 still contributes via
    uniform-attention-weighted V, it isn't ablated. vo_mult directly
    controls how much the head contributes to the residual stream, so
    vo_mult=0 is what actually "kills" a head.
    """
    head_configs = cfg.get("head_configs", [])
    B, Sq, HD = q.shape
    if HD % heads != 0 or not head_configs:
        return original_fn(q, k, v, heads, *extra_args,
                           attn_precision=attn_precision,
                           transformer_options=transformer_options,
                           **extra_kwargs)

    D_head = HD // heads

    need_q = any(c["qk_mult"] != 1.0 for c in head_configs)
    need_v = any(c["vo_mult"] != 1.0 for c in head_configs)

    q_mod, v_mod = q, v
    if need_q:
        q_mod = q.clone()
        for c in head_configs:
            if c["qk_mult"] != 1.0:
                h0 = c["head_idx"] * D_head
                h1 = h0 + D_head
                q_mod[:, :, h0:h1] *= c["qk_mult"]
    if need_v:
        v_mod = v.clone()
        for c in head_configs:
            if c["vo_mult"] != 1.0:
                h0 = c["head_idx"] * D_head
                h1 = h0 + D_head
                v_mod[:, :, h0:h1] *= c["vo_mult"]

    return original_fn(q_mod, k, v_mod, heads, *extra_args,
                       attn_precision=attn_precision,
                       transformer_options=transformer_options,
                       **extra_kwargs)

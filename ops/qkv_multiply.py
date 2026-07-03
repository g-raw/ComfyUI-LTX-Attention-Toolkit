from __future__ import annotations


def apply_qkv_multiply(q, k, v, heads, cfg,
                        original_fn, extra_args, extra_kwargs,
                        attn_precision, transformer_options):
    """Scale Q per head before the attention call (qk_mult), and/or scale
    the per-head slice of the (pre-output-projection) attention output
    afterward (vo_mult). Only two independent knobs: scaling Q and K
    separately would just multiply through as (q_mult * k_mult) on the
    softmax logits since both are uniform per-head scalars, so a single
    qk_mult applied to Q alone has the identical effect. Same reasoning
    for V/O -- V is scaled before the linear attn_weights @ V matmul, O
    after, so a single vo_mult applied to the output alone is equivalent
    to any v_mult * o_mult split.

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
    need_o = any(c["vo_mult"] != 1.0 for c in head_configs)

    q_mod = q
    if need_q:
        q_mod = q.clone()
        for c in head_configs:
            if c["qk_mult"] != 1.0:
                h0 = c["head_idx"] * D_head
                h1 = h0 + D_head
                q_mod[:, :, h0:h1] *= c["qk_mult"]

    out = original_fn(q_mod, k, v, heads, *extra_args,
                      attn_precision=attn_precision,
                      transformer_options=transformer_options,
                      **extra_kwargs)

    if need_o:
        out = out.clone()
        for c in head_configs:
            if c["vo_mult"] != 1.0:
                h0 = c["head_idx"] * D_head
                h1 = h0 + D_head
                out[:, :, h0:h1] *= c["vo_mult"]

    return out

from __future__ import annotations
import types

from ..utils.helpers import reset_call_count


def make_profiler_hook_factory(timestep_ref: list, num_frames_ref: list,
                               ppf_ref: list, h_lat_ref: list, w_lat_ref: list,
                               cfg: dict, attn_blocks: set, qkv_blocks: set):
    """Returns a make_hook(block_idx, existing_hook) -> hook callable, for
    register_layer(). Injects profiling and/or QKV-capture metadata into
    transformer_options then delegates to existing_hook (if any earlier
    layer already claimed this block) or the original block.
    is_profiler_block/is_qkv_block gate which metadata gets injected -- a
    block can be QKV-only (no attention-metrics hook needed, avoids a
    wasted attn-map computation in core/hooks.py) or profiler-only,
    independently."""
    def make_hook(block_idx, existing_hook):
        is_profiler_block = block_idx in attn_blocks
        is_qkv_block      = block_idx in qkv_blocks

        def block_hook(args: dict, orig: dict):
            to = dict(args.get("transformer_options", {}))
            if is_profiler_block:
                to["_profiler_block_idx"]         = block_idx
                to["_profiler_timestep"]          = timestep_ref[0]
                to["_profiler_num_frames"]        = num_frames_ref[0]
                to["_profiler_patches_per_frame"] = ppf_ref[0]
                to["_profiler_latent_h"]          = h_lat_ref[0]
                to["_profiler_latent_w"]          = w_lat_ref[0]
                to["_profiler_capture_sa"]        = cfg.get("capture_sa", True)
                to["_profiler_capture_ca"]        = cfg.get("capture_ca", True)
            if is_qkv_block:
                to["_qkv_block_idx"]      = block_idx
                to["_qkv_capture_active"] = True
                to["_qkv_capture_sa"]     = cfg.get("capture_sa", True)
                to["_qkv_capture_ca"]     = cfg.get("capture_ca", True)
                to["_qkv_timestep"]       = timestep_ref[0]
            reset_call_count(block_idx)
            new_args = {**args, "transformer_options": to}
            if existing_hook is not None:
                return existing_hook(new_args, orig)
            return orig["original_block"](new_args)
        return block_hook

    return make_hook


def register_layer(dm, node_id, blocks: set, make_hook, on_call=None):
    """Register (or replace) a single node's contribution to the shared
    per-block hook chain on this diffusion_model.

    blocks: set of block indices this layer wants a hook on.
    make_hook(block_idx, existing_hook) -> hook or None. Returning None
    means "don't touch this call" (e.g. step out of range) -- falls
    through to whatever earlier layers/the original block already
    contribute for that call.
    on_call(x, timestep, transformer_options): optional, invoked once per
    _forward call before the block loop (Setup Capture uses this to
    refresh the geometry/timestep ref boxes _make_block_hook reads).

    Re-registering the same node_id replaces its entry in place (Python
    dict key reassignment keeps the original insertion position) -- the
    old closure (and any tensors it held) is simply dropped for GC. This
    is what avoids the old leak (wrap-on-top-of-wrap across repeated
    runs of the same node) while still letting *different* node
    instances coexist and chain within one run.
    """
    dm._ltx_layers = getattr(dm, "_ltx_layers", {})
    dm._ltx_layers[node_id] = {"blocks": blocks, "make_hook": make_hook, "on_call": on_call}
    _install_layer_runner(dm)


def unregister_layer(dm, node_id):
    """Remove only this node's own contribution. Used by every node's
    blank/disabled path instead of wiping the whole diffusion_model --
    leaves any other node's layers (and the pristine forward) intact."""
    getattr(dm, "_ltx_layers", {}).pop(node_id, None)


def reset_all_layers(dm):
    """Clear every registered layer and restore the pristine forward.
    Manual escape hatch for orphaned layers left behind by a node that
    was deleted/rewired out of the graph (nothing calls unregister_layer
    for an id that stops executing)."""
    if getattr(dm, "_ltx_runner_installed", False):
        dm._forward = dm._ltx_pristine_forward
        dm._ltx_runner_installed = False
    dm._ltx_layers = {}


def _install_layer_runner(dm):
    if getattr(dm, "_ltx_runner_installed", False):
        return

    pristine = dm._forward

    def patched_forward(self_dm, x, timestep, context, attention_mask,
                        frame_rate=25, transformer_options={},
                        keyframe_idxs=None, **kwargs):

        layers = getattr(self_dm, "_ltx_layers", {})

        for spec in layers.values():
            if spec["on_call"]:
                spec["on_call"](x, timestep, transformer_options)

        patches_replace = dict(transformer_options.get("patches_replace", {}))
        dit_replace     = dict(patches_replace.get("dit", {}))

        all_blocks = set()
        for spec in layers.values():
            all_blocks |= spec["blocks"]

        for block_idx in all_blocks:
            existing = dit_replace.get(("double_block", block_idx))
            for spec in layers.values():
                if block_idx in spec["blocks"]:
                    new_hook = spec["make_hook"](block_idx, existing)
                    if new_hook is not None:
                        existing = new_hook
            if existing is not None:
                dit_replace[("double_block", block_idx)] = existing
            else:
                dit_replace.pop(("double_block", block_idx), None)

        patches_replace["dit"] = dit_replace
        transformer_options    = {**transformer_options,
                                  "patches_replace": patches_replace}
        return pristine(
            x, timestep, context, attention_mask,
            frame_rate, transformer_options, keyframe_idxs, **kwargs,
        )

    dm._forward               = types.MethodType(patched_forward, dm)
    dm._ltx_pristine_forward  = pristine
    dm._ltx_runner_installed  = True

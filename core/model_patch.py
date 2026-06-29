from __future__ import annotations
import types

from ..utils.helpers import reset_call_count


def _make_block_hook(block_idx: int, timestep_ref: list,
                     num_frames_ref: list, ppf_ref: list,
                     cfg: dict, existing_hook=None):
    """
    Hook injected via patches_replace["dit"].
    Injects profiling metadata into transformer_options
    then delegates to the original block.
    """
    def block_hook(args: dict, orig: dict):
        to = dict(args.get("transformer_options", {}))
        to["_profiler_block_idx"]         = block_idx
        to["_profiler_timestep"]          = timestep_ref[0]
        to["_profiler_num_frames"]        = num_frames_ref[0]
        to["_profiler_patches_per_frame"] = ppf_ref[0]
        to["_profiler_capture_sa"]        = cfg.get("capture_sa", True)
        to["_profiler_capture_ca"]        = cfg.get("capture_ca", True)
        reset_call_count(block_idx)
        new_args = {**args, "transformer_options": to}
        if existing_hook is not None:
            return existing_hook(new_args, orig)
        return orig["original_block"](new_args)

    return block_hook


def wrap_diffusion_model(diffusion_model, cfg: dict):
    """
    Wraps _forward to inject geometry and block hooks
    at each denoising step.
    """
    original_forward  = diffusion_model._forward
    timestep_ref      = [0.0]
    num_frames_ref    = [1]
    ppf_ref           = [1]

    def patched_forward(self_dm, x, timestep, context, attention_mask,
                        frame_rate=25, transformer_options={},
                        keyframe_idxs=None, **kwargs):

        # Geometry
        vx = x[0] if isinstance(x, (list, tuple)) else x
        if vx.dim() == 5:
            _, _, F_lat, H_lat, W_lat = vx.shape
            num_frames_ref[0] = F_lat
            ppf_ref[0]        = H_lat * W_lat

        # Scalar timestep
        ts = timestep[0] if isinstance(timestep, (list, tuple)) else timestep
        timestep_ref[0] = float(ts.mean().item())

        # Inject block hooks
        patches_replace = dict(transformer_options.get("patches_replace", {}))
        dit_replace     = dict(patches_replace.get("dit", {}))

        for blk_idx in cfg.get("target_blocks", set()):
            existing = dit_replace.get(("double_block", blk_idx))
            if getattr(existing, "_is_profiler_hook", False):
                existing = getattr(existing, "_wrapped_original", None)

            hook = _make_block_hook(
                blk_idx, timestep_ref, num_frames_ref, ppf_ref,
                cfg, existing_hook=existing,
            )
            hook._is_profiler_hook   = True
            hook._wrapped_original   = existing
            dit_replace[("double_block", blk_idx)] = hook

        patches_replace["dit"] = dit_replace
        transformer_options    = {**transformer_options,
                                  "patches_replace": patches_replace}
        return original_forward(
            x, timestep, context, attention_mask,
            frame_rate, transformer_options, keyframe_idxs, **kwargs,
        )

    diffusion_model._forward = types.MethodType(patched_forward, diffusion_model)
    diffusion_model._profiler_original_forward = original_forward
    diffusion_model._profiler_patched          = True


def unwrap_diffusion_model(diffusion_model):
    if getattr(diffusion_model, "_profiler_patched", False):
        diffusion_model._forward  = diffusion_model._profiler_original_forward
        diffusion_model._profiler_patched = False


def make_simple_patched_forward(original_forward, block_configs: dict,
                                 step_counters: dict):
    """
    Fabrics a generic patched_forward for nodes that don't need
    full geometry (Transfer, Freeze, MapStore).

    block_configs : {block_idx: callable(dit_replace, current_step, existing)}
      The callable modifies dit_replace in-place.
    """
    def patched_forward(self_dm, x, timestep, context, attention_mask,
                        frame_rate=25, transformer_options={},
                        keyframe_idxs=None, **kwargs):

        patches_replace = dict(transformer_options.get("patches_replace", {}))
        dit_replace     = dict(patches_replace.get("dit", {}))

        for blk_idx, configure_hook in block_configs.items():
            if blk_idx not in step_counters:
                step_counters[blk_idx] = 0
            current_step             = step_counters[blk_idx]
            step_counters[blk_idx]  += 1
            existing                 = dit_replace.get(("double_block", blk_idx))
            configure_hook(dit_replace, current_step, existing, blk_idx)

        patches_replace["dit"] = dit_replace
        transformer_options    = {**transformer_options,
                                  "patches_replace": patches_replace}
        return original_forward(
            x, timestep, context, attention_mask,
            frame_rate, transformer_options, keyframe_idxs, **kwargs,
        )

    return patched_forward
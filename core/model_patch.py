from __future__ import annotations
import types

from ..utils.helpers import reset_call_count


def _make_block_hook(block_idx: int, timestep_ref: list,
                     num_frames_ref: list, ppf_ref: list,
                     h_lat_ref: list, w_lat_ref: list,
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
        to["_profiler_latent_h"]          = h_lat_ref[0]
        to["_profiler_latent_w"]          = w_lat_ref[0]
        to["_profiler_capture_sa"]        = cfg.get("capture_sa", True)
        to["_profiler_capture_ca"]        = cfg.get("capture_ca", True)
        if cfg.get("capture_qkv"):
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


def wrap_diffusion_model(diffusion_model, cfg: dict):
    """
    Wraps _forward to inject geometry and block hooks
    at each denoising step.
    """
    original_forward  = diffusion_model._forward
    timestep_ref      = [0.0]
    num_frames_ref    = [1]
    ppf_ref           = [1]
    h_lat_ref         = [1]
    w_lat_ref         = [1]

    def patched_forward(self_dm, x, timestep, context, attention_mask,
                        frame_rate=25, transformer_options={},
                        keyframe_idxs=None, **kwargs):

        # Geometry
        vx = x[0] if isinstance(x, (list, tuple)) else x
        if vx.dim() == 5:
            _, _, F_lat, H_lat, W_lat = vx.shape
            num_frames_ref[0] = F_lat
            ppf_ref[0]        = H_lat * W_lat
            h_lat_ref[0]      = H_lat
            w_lat_ref[0]      = W_lat

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
                h_lat_ref, w_lat_ref, cfg, existing_hook=existing,
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

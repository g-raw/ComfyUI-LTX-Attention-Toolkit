from __future__ import annotations
import types

from ..core.stores      import AttentionStore, QKVStore
from ..core.hooks       import install_hook
from ..core.model_patch import wrap_diffusion_model, unwrap_diffusion_model
from ..utils.helpers    import parse_int_set, reset_call_count


class LTXAttentionCaptureSetup:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":           ("MODEL",),
            "capture_sa":      ("BOOLEAN", {"default": True}),
            "capture_ca":      ("BOOLEAN", {"default": True}),
            "store_full_maps": ("BOOLEAN", {"default": False}),
            "map_downsample":  ("INT",     {"default": 1, "min": 1, "max": 16}),
            "target_blocks":   ("STRING",  {"default": "0,8,16,24,32,40,47"}),
            "capture_steps":   ("STRING",  {"default": "all"}),
            "reset_store":     ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION     = "setup"
    CATEGORY     = "g_raw/LTX/Profiler"

    def setup(self, model, capture_sa, capture_ca, store_full_maps,
              map_downsample, target_blocks, capture_steps, reset_store):

        parsed_blocks = parse_int_set(target_blocks, range(48)) or set(range(48))
        parsed_steps  = parse_int_set(capture_steps)

        store = AttentionStore.get()
        if reset_store:
            store.reset()

        store.cfg = {
            "capture_sa":      capture_sa,
            "capture_ca":      capture_ca,
            "store_full_maps": store_full_maps,
            "map_downsample":  map_downsample,
            "target_blocks":   parsed_blocks,
            "capture_steps":   parsed_steps,
        }

        install_hook()
        patched = model.clone()
        dm      = patched.model.diffusion_model
        unwrap_diffusion_model(dm)
        wrap_diffusion_model(dm, store.cfg)

        print(
            f"[LTXProfiler] CaptureSetup\n"
            f"  SA={capture_sa} CA={capture_ca} full_maps={store_full_maps}\n"
            f"  Blocs : {sorted(parsed_blocks)}\n"
            f"  Steps : {'tous' if parsed_steps is None else sorted(parsed_steps)}"
        )
        return (patched,)


class LTXQKVCapture:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":         ("MODEL",),
            "target_blocks": ("STRING", {"default": "24"}),
            "target_heads":  ("STRING", {"default": "8,12,16"}),
            "capture_steps": ("STRING", {"default": "all"}),
            "capture_sa":    ("BOOLEAN", {"default": True}),
            "capture_ca":    ("BOOLEAN", {"default": False}),
            "reset_store":   ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("capture_model",)
    FUNCTION     = "setup"
    CATEGORY     = "g_raw/LTX/Profiler"

    def setup(self, model, target_blocks, target_heads,
              capture_steps, capture_sa, capture_ca, reset_store):

        parsed_blocks = parse_int_set(target_blocks, range(48)) or set(range(48))
        parsed_heads  = parse_int_set(target_heads)
        parsed_steps  = parse_int_set(capture_steps)

        qkv_store = QKVStore.get()
        if reset_store:
            qkv_store.reset()

        qkv_store.cfg = {
            "target_blocks": parsed_blocks,
            "target_heads":  parsed_heads,
            "capture_steps": parsed_steps,
            "capture_sa":    capture_sa,
            "capture_ca":    capture_ca,
        }

        install_hook()
        patched      = model.clone()
        dm           = patched.model.diffusion_model
        unwrap_diffusion_model(dm)

        original_forward = dm._forward
        step_counters    = {}
        timestep_ref     = [0.0]

        def patched_forward(self_dm, x, timestep, context, attention_mask,
                            frame_rate=25, transformer_options={},
                            keyframe_idxs=None, **kwargs):

            ts_val = float(
                (timestep[0] if isinstance(timestep, (list, tuple))
                 else timestep).mean().item()
            )
            timestep_ref[0] = ts_val

            patches_replace = dict(transformer_options.get("patches_replace", {}))
            dit_replace     = dict(patches_replace.get("dit", {}))

            for blk_idx in parsed_blocks:
                if blk_idx not in step_counters:
                    step_counters[blk_idx] = 0
                current_step           = step_counters[blk_idx]
                step_counters[blk_idx] += 1

                if parsed_steps is not None and current_step not in parsed_steps:
                    continue

                existing = dit_replace.get(("double_block", blk_idx))

                def make_hook(bidx, sidx, existing_h):
                    def hook(args, orig):
                        to = dict(args.get("transformer_options", {}))
                        to["_qkv_block_idx"]      = bidx
                        to["_qkv_step_idx"]       = sidx
                        to["_qkv_capture_active"] = True
                        to["_qkv_capture_sa"]     = capture_sa
                        to["_qkv_capture_ca"]     = capture_ca
                        to["_qkv_timestep"]       = timestep_ref[0]
                        reset_call_count(bidx)
                        new_args = {**args, "transformer_options": to}
                        if existing_h and not getattr(existing_h, "_is_capture_hook", False):
                            return existing_h(new_args, orig)
                        return orig["original_block"](new_args)
                    hook._is_capture_hook = True
                    return hook

                dit_replace[("double_block", blk_idx)] = make_hook(
                    blk_idx, current_step, existing
                )

            patches_replace["dit"] = dit_replace
            transformer_options    = {**transformer_options,
                                      "patches_replace": patches_replace}
            return original_forward(
                x, timestep, context, attention_mask,
                frame_rate, transformer_options, keyframe_idxs, **kwargs,
            )

        dm._forward                   = types.MethodType(patched_forward, dm)
        dm._profiler_patched          = True
        dm._profiler_original_forward = original_forward

        print(
            f"[LTXProfiler] QKVCapture\n"
            f"  Blocs : {sorted(parsed_blocks)}\n"
            f"  Têtes : {'toutes' if parsed_heads is None else sorted(parsed_heads)}\n"
            f"  Steps : {'tous' if parsed_steps is None else sorted(parsed_steps)}"
        )
        return (patched,)
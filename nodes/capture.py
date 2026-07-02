from __future__ import annotations
import types
import warnings

from ..core.stores      import get_registry, AttentionStore, QKVStore
from ..core.hooks       import install_hook
from ..core.model_patch import wrap_diffusion_model, unwrap_diffusion_model
from ..utils.helpers    import parse_int_set, reset_call_count, parse_block_head_pairs


class LTXAttentionCaptureSetup:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":           ("MODEL",),
            "capture_sa":      ("BOOLEAN", {"default": True}),
            "capture_ca":      ("BOOLEAN", {"default": True}),
            "target_blocks":   ("STRING",  {"default": "0,8,16,24,32,40,47"}),
            "target_heads":    ("STRING",  {"default": "all"}),
            "capture_steps":   ("STRING",  {"default": "all"}),
            "store_mode":      (["reduced", "full_fp16", "hybrid"],
                               {"default": "reduced",
                                "tooltip": "reduced: metrics + key/query maps only\n"
                                           "full_fp16: + full attention map for every target block\n"
                                           "hybrid: full map only for full_blocks, reduced elsewhere"}),
            "full_blocks":     ("STRING",  {"default": "8,16,24,32,40",
                               "tooltip": "Blocks stored at full resolution (every head) in "
                                          "hybrid mode. A block also listed in full_targets uses "
                                          "full_targets' per-head selection instead (full_targets "
                                          "wins for that one block); full_blocks still applies "
                                          "normally to any block it doesn't cover -- the two can "
                                          "be combined: coarse whole-block capture for exploration "
                                          "(Query Map/Key Map/Zone Analysis) alongside fine "
                                          "per-head capture for Head Freeze targets."}),
            "full_targets":    ("STRING",  {"default": "", "multiline": True,
                               "tooltip": "Optional, hybrid mode only: for the specific blocks "
                                          "listed here, restrict full-map storage to specific "
                                          "(block, head) pairs instead of every head -- saves RAM "
                                          "when you already know which heads you'll feed into Head "
                                          "Freeze. Paste Head Candidates' candidates_csv directly "
                                          "(one 'block,head' per line), or type manually as "
                                          "'block:head | block:head | ...'. Heads not listed for a "
                                          "block covered here won't have a full map available, so "
                                          "Query Map/Key Map/Zone Analysis on them will error or "
                                          "skip -- use Head Freeze/QKV Transfer for those, or add "
                                          "that block to full_blocks instead (not here) for full "
                                          "multi-head coverage."}),
            "map_downsample":  ("INT",     {"default": 1, "min": 1, "max": 64}),
            "reset_store":     ("BOOLEAN", {"default": True}),
            "store_name":      ("STRING",  {"default": ""}),
        }}

    RETURN_TYPES = ("MODEL", "STORE_HANDLE")
    RETURN_NAMES = ("patched_model", "store_handle")
    FUNCTION     = "setup"
    CATEGORY     = "g_raw/LTX/Profiler"

    def setup(self, model, capture_sa, capture_ca, target_blocks, target_heads,
              capture_steps, store_mode, full_blocks, full_targets, map_downsample,
              reset_store, store_name):

        # Validate reset_store is a boolean
        if not isinstance(reset_store, (bool, int)):
            raise ValueError(f"reset_store must be a boolean, got {type(reset_store).__name__}")

        # Validate map_downsample in [1, 64]
        if not (1 <= map_downsample <= 64):
            raise ValueError(f"map_downsample must be in [1, 64], got {map_downsample}")

        # Validate capture_steps format
        cs = capture_steps.strip().lower()
        if cs != "all":
            parsed_steps_test = parse_int_set(capture_steps)
            if parsed_steps_test is not None and len(parsed_steps_test) == 0:
                raise ValueError(f"capture_steps '{capture_steps}' does not contain valid integers.")

        valid_modes = ("reduced", "full_fp16", "hybrid")
        if store_mode not in valid_modes:
            raise ValueError(f"store_mode must be one of {valid_modes}, got '{store_mode}'")

        parsed_blocks  = parse_int_set(target_blocks, range(48)) or set(range(48))
        parsed_heads   = parse_int_set(target_heads)
        parsed_steps   = parse_int_set(capture_steps)
        full_block_set = {int(x.strip()) for x in full_blocks.split(",") if x.strip()}

        full_target_map = None
        if full_targets.strip():
            full_target_map = {}
            for blk, hd in parse_block_head_pairs(full_targets):
                full_target_map.setdefault(blk, set()).add(hd)

        # Create/use named store via StoreRegistry (atomic to avoid race)
        reg = get_registry()
        inst = reg.create_and_get_attn(store_name if store_name else None)
        if reset_store:
            inst.reset_data()

        handle = inst.name

        inst.cfg = {
            "capture_sa":    capture_sa,
            "capture_ca":    capture_ca,
            "target_blocks": parsed_blocks,
            "target_heads":  parsed_heads,
            "capture_steps": parsed_steps,
            "store_mode":      store_mode,
            "full_blocks":     full_block_set,
            "full_target_map": full_target_map,
            "map_downsample":  map_downsample,
        }

        install_hook()
        patched = model.clone()
        dm      = patched.model.diffusion_model
        unwrap_diffusion_model(dm)
        wrap_diffusion_model(dm, inst.cfg)

        full_summary = ""
        if store_mode == "hybrid":
            if full_target_map is not None:
                full_summary = " (full targets: " + ", ".join(
                    f"b{b}h{sorted(hs)}" for b, hs in sorted(full_target_map.items())
                ) + ")"
            else:
                full_summary = f" (full: {sorted(full_block_set)})"

        print(
            f"[LTXProfiler] CaptureSetup\n"
            f"  SA={capture_sa} CA={capture_ca} store_mode={store_mode}\n"
            f"  Blocks : {sorted(parsed_blocks)}{full_summary}\n"
            f"  Heads : {'all' if parsed_heads is None else sorted(parsed_heads)}\n"
            f"  Steps : {'all' if parsed_steps is None else sorted(parsed_steps)}\n"
            f"  Store : {handle}"
        )
        return (patched, handle)


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
            "store_name":    ("STRING",  {"default": ""}),
        }}

    RETURN_TYPES = ("MODEL", "QKV_STORE_HANDLE")
    RETURN_NAMES = ("capture_model", "qkv_handle")
    FUNCTION     = "setup"
    CATEGORY     = "g_raw/LTX/Profiler"

    def setup(self, model, target_blocks, target_heads,
              capture_steps, capture_sa, capture_ca, reset_store, store_name):

        parsed_blocks = parse_int_set(target_blocks, range(48)) or set(range(48))
        parsed_heads  = parse_int_set(target_heads)
        parsed_steps  = parse_int_set(capture_steps)

        # Create/use named QKV store via StoreRegistry (atomic to avoid race)
        reg = get_registry()
        qkv_inst = reg.create_and_get_qkv(store_name if store_name else None)
        qkv_handle = qkv_inst.name
        if reset_store:
            qkv_inst.reset_data()

        qkv_inst.cfg = {
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
            f"  Blocks : {sorted(parsed_blocks)}\n"
            f"  Heads : {'all' if parsed_heads is None else sorted(parsed_heads)}\n"
            f"  Steps : {'all' if parsed_steps is None else sorted(parsed_steps)}\n"
            f"  Store : {qkv_handle}"
        )
        return (patched, qkv_handle)
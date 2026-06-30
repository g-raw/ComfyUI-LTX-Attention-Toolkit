from __future__ import annotations
import types

from ..core.stores         import get_registry, AttentionStore
from ..core.hooks          import install_hook
from ..core.model_patch    import unwrap_diffusion_model
from ..ops.map_store       import make_map_store_callback, make_map_store_patched_forward
from ..utils.helpers       import parse_int_set


class LTXAttentionMapStore:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":         ("MODEL",),
            "attn_type":     (["sa", "ca"], {"default": "sa"}),
            "target_blocks": ("STRING", {"default": "all"}),
            "target_heads":  ("STRING", {"default": "all"}),
            "target_steps":  ("STRING", {"default": "all"}),
            "store_name":    ("STRING", {"default": "", "placeholder": "optional name",
                               "tooltip": "Named store; empty = auto-generated."}),
            "store_mode":    (["reduced", "full_fp16", "hybrid"],
                             {"default": "reduced"}),
            "full_blocks":   ("STRING", {"default": "8,16,24,32,40",
                               "tooltip": "Blocks for hybrid mode storage."}),
            "latent_frames": ("INT", {"default": 10, "min": 1, "max": 256}),
            "latent_height": ("INT", {"default": 11, "min": 1, "max": 256}),
            "latent_width":  ("INT", {"default": 20, "min": 1, "max": 256}),
            "reset_store":   ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES  = ("MODEL", "ATTN_MAP_STORE", "MAP_STORE_HANDLE")
    RETURN_NAMES  = ("patched_model", "map_store", "store_handle")
    FUNCTION      = "setup"
    CATEGORY      = "g_raw/LTX/Profiler"

    def setup(self, model, attn_type, target_blocks, target_heads,
              target_steps, store_name, store_mode, full_blocks,
              latent_frames, latent_height, latent_width, reset_store):

        parsed_blocks  = parse_int_set(target_blocks, range(48)) or set(range(48))
        parsed_heads   = parse_int_set(target_heads)
        parsed_steps   = parse_int_set(target_steps)
        full_block_set = {int(x.strip()) for x in full_blocks.split(",") if x.strip()}

        valid_modes = ("reduced", "full_fp16", "hybrid")
        if store_mode not in valid_modes:
            raise ValueError(f"store_mode must be one of {valid_modes}, got '{store_mode}'")

        target_call_n  = 0 if attn_type == "sa" else 1

        # ── Store instance (unique auto-name per call via id-based counter) ──
        reg = get_registry()
        store       = reg.create_and_get_attn(store_name or None)
        handle      = store.name
        if reset_store:
            store.reset_data()
        store.cfg = {
            "capture_sa":      attn_type == "sa",
            "capture_ca":      attn_type == "ca",
            "store_full_maps": False,
            "target_blocks":   parsed_blocks,
            "capture_steps":   parsed_steps,
        }

        # ── Callback (writes to map_data + store.sa/ca simultaneously) ──────
        map_data      = {}
        step_counters: dict[int, int] = {}
        num_frames_ref = [latent_frames]

        store_map_cb = make_map_store_callback(
            map_data, parsed_heads, store_mode, full_block_set,
            latent_frames, latent_height, latent_width,
            target_inst=store,   # ← also write to registry (key fix for Bug 1)
        )
        store._save_callback   = store_map_cb
        store._parsed_heads    = parsed_heads

        # ── Patch model ─────────────────────────────────────────────────────
        patched       = model.clone()
        dm            = patched.model.diffusion_model
        unwrap_diffusion_model(dm)      # clean slate (no cfg collision now)
        original_fwd  = dm._forward

        install_hook()                    # ensures _make_full_hook is active

        pf = make_map_store_patched_forward(
            original_fwd,
            parsed_blocks, parsed_steps,
            target_call_n, store_map_cb, parsed_heads,
            step_counters, num_frames_ref, store,
        )
        dm._forward                      = types.MethodType(pf, dm)
        dm._profiler_patched             = True
        dm._profiler_original_forward    = original_fwd

        # ── RAM estimation ─────────────────────────────────────────────────
        P         = latent_height * latent_width
        n_full    = len(full_block_set if store_mode == "hybrid"
                        else parsed_blocks if store_mode == "full_fp16"
                        else set())
        n_red     = len(parsed_blocks) - n_full
        n_heads_  = 32 if parsed_heads is None else len(parsed_heads)
        n_steps_  = 4  if parsed_steps is None   else len(parsed_steps)
        ram_full  = n_full * n_heads_ * n_steps_ * P * P * 2 / 1e9
        ram_red   = (n_red + n_full) * n_heads_ * n_steps_ * P * 2 * 4 / 1e9

        print(
            f"[LTXProfiler/MapStore] mode={store_mode}  store={handle}\n"
            f"  blocks={len(parsed_blocks)} "
            f"({'of '+str(n_full)+' full' if store_mode=='hybrid' else ''})\n"
            f"  heads={'32' if parsed_heads is None else len(parsed_heads)}\n"
            f"  RAM full~{ram_full:.1f}GB  red~{ram_red:.2f}GB  "
            f"tot~{ram_full+ram_red:.1f}GB"
        )
        return (patched, map_data, handle)

from __future__ import annotations

from ..core.stores      import get_registry
from ..core.hooks       import install_hook
from ..core.model_patch import wrap_diffusion_model, unwrap_diffusion_model
from ..utils.helpers    import parse_int_set, parse_block_head_pairs


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
                                          "'block:head | block:head | ...' (also accepts "
                                          "'block:h1,h2,...' and 'block:all'). Heads not listed for "
                                          "a block covered here won't have a full map available, so "
                                          "Query Map/Key Map/Zone Analysis on them will error or "
                                          "skip -- use Head Freeze/QKV Transfer for those, or add "
                                          "that block to full_blocks instead (not here) for full "
                                          "multi-head coverage."}),
            "map_downsample":  ("INT",     {"default": 1, "min": 1, "max": 64}),

            # ── QKV capture (separate from the attention-map settings above) ──
            "capture_qkv":     ("BOOLEAN", {"default": False,
                               "tooltip": "Also capture raw Q/K/V per head into a separate QKV "
                                          "store, for QKV Transfer. Not redundant with store_mode's "
                                          "full attention map: the map only replays the exact "
                                          "historical pattern, raw Q/K/V lets QKV Transfer "
                                          "recombine components from two different generations "
                                          "into a new pattern."}),
            "qkv_targets":     ("STRING",  {"default": "", "multiline": True,
                               "tooltip": "Only used when capture_qkv is on. Same format as "
                                          "full_targets/Head Freeze's targets: paste Head "
                                          "Candidates' candidates_csv directly (one 'block,head' "
                                          "per line), or type manually as 'block:head | "
                                          "block:h1,h2,... | block:all | ...'. A whole-string "
                                          "'all' (or 'all:all') captures every block and head. "
                                          "Independent from target_blocks/target_heads above -- "
                                          "raw Q/K/V is much more expensive than the metrics/key/"
                                          "query maps, so this needs its own explicit list."}),

            "store_name":      ("STRING",  {"default": ""}),
            "reset_store":     ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL", "STORE_HANDLE", "QKV_STORE_HANDLE")
    RETURN_NAMES = ("patched_model", "store_handle", "qkv_handle")
    FUNCTION     = "setup"
    CATEGORY     = "g_raw/LTX/Profiler"

    def setup(self, model, capture_sa, capture_ca, target_blocks, target_heads,
              capture_steps, store_mode, full_blocks, full_targets, map_downsample,
              capture_qkv, qkv_targets, store_name, reset_store):

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
                # "all" here just duplicates full_blocks for that one block
                # (no store exists yet to resolve "all" against captured
                # heads, unlike Head Freeze) -- supported for consistency
                # with Head Freeze's syntax, but full_blocks is simpler.
                heads = range(32) if hd == "all" else (hd,)
                full_target_map.setdefault(blk, set()).update(heads)

        qkv_target_map = None
        if capture_qkv:
            if not qkv_targets.strip():
                raise ValueError("[CaptureSetup] capture_qkv is on but qkv_targets is blank.")
            if qkv_targets.strip().lower() in ("all", "all:all"):
                qkv_target_map = {b: set(range(32)) for b in range(48)}
            else:
                qkv_target_map = {}
                for blk, hd in parse_block_head_pairs(qkv_targets):
                    heads = set(range(32)) if hd == "all" else {hd}
                    qkv_target_map.setdefault(blk, set()).update(heads)

        # Create/use named store(s) via StoreRegistry (atomic to avoid race)
        reg  = get_registry()
        inst = reg.create_and_get_attn(store_name if store_name else None)
        if reset_store:
            inst.reset_data()

        handle = inst.name

        inst.cfg = {
            "capture_sa":       capture_sa,
            "capture_ca":       capture_ca,
            "target_blocks":    parsed_blocks,
            "target_heads":     parsed_heads,
            "capture_steps":    parsed_steps,
            "store_mode":       store_mode,
            "full_blocks":      full_block_set,
            "full_target_map":  full_target_map,
            "map_downsample":   map_downsample,
            # Cross-store bookkeeping consumed only by wrap_diffusion_model's
            # block-hook-installation loop (core/model_patch.py), not by
            # _attn_record -- QKV capture's own filtering reads the same
            # map from qkv_inst.cfg below.
            "target_block_map": qkv_target_map,
        }

        qkv_handle = ""
        if capture_qkv:
            qkv_inst = reg.create_and_get_qkv(store_name if store_name else None)
            if reset_store:
                qkv_inst.reset_data()
            qkv_inst.cfg = {
                "target_block_map": qkv_target_map,
                "capture_steps":    parsed_steps,
                "capture_sa":       capture_sa,
                "capture_ca":       capture_ca,
            }
            qkv_handle = qkv_inst.name

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

        qkv_line = "(off)"
        if capture_qkv:
            targets_str = ", ".join(
                f"b{b}h{sorted(hs)}" for b, hs in sorted(qkv_target_map.items())
            )
            qkv_line = f"{qkv_handle} ({targets_str})"

        print(
            f"[LTXProfiler] CaptureSetup\n"
            f"  SA={capture_sa} CA={capture_ca} store_mode={store_mode}\n"
            f"  Blocks : {sorted(parsed_blocks)}{full_summary}\n"
            f"  Heads : {'all' if parsed_heads is None else sorted(parsed_heads)}\n"
            f"  Steps : {'all' if parsed_steps is None else sorted(parsed_steps)}\n"
            f"  Store : {handle}\n"
            f"  QKV   : {qkv_line}"
        )
        return (patched, handle, qkv_handle)

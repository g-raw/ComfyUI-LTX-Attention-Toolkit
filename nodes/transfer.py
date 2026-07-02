from __future__ import annotations
import types

from ..core.stores      import AttentionStore, QKVStore, get_registry
from ..core.hooks       import install_hook
from ..core.model_patch import unwrap_diffusion_model
from ..ops.freeze       import make_freeze_hook
from ..utils.helpers    import parse_int_set, reset_call_count, parse_block_head_pairs


class LTXAttentionHeadFreeze:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":              ("MODEL",),
            "targets":            ("STRING", {"default": "24:8", "multiline": True,
                                   "tooltip": "One or more (block, head) pairs to freeze in "
                                              "the same run. Paste Head Candidates' "
                                              "candidates_csv directly (one 'block,head' per "
                                              "line), or type manually as "
                                              "'block:head | block:head | ...'. Use 'block:all' "
                                              "to freeze every captured head of that block in "
                                              "one entry. Leave blank to disable the freeze "
                                              "entirely -- this is the reliable way to turn it "
                                              "off; ComfyUI's node bypass/mute skips this node's "
                                              "cleanup, so the diffusion_model (shared across "
                                              "runs) can be left patched from a previous run."}),
            "freeze_from_step":   ("INT",   {"default": 3,  "min": 0, "max": 255}),
            "freeze_step_source": ("INT",   {"default": 3,  "min": 0, "max": 255}),
            "attn_type":          (["sa"],  {"default": "sa"}),
            "blend_weight":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                   "step": 0.05,
                                   "tooltip": "Shared across all targets — no per-head override yet."}),
            "store_handle":       ("STRING", {"default": "", "placeholder": "select store..."}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("steered_model",)
    FUNCTION     = "apply_freeze"
    CATEGORY     = "g_raw/LTX/Profiler"

    def apply_freeze(self, model, targets, freeze_from_step,
                     freeze_step_source, attn_type, blend_weight, store_handle):

        pairs = parse_block_head_pairs(targets) if targets.strip() else []
        if not pairs:
            # Nothing to freeze — unwrap so a previous run's patch doesn't
            # linger on this shared diffusion_model, then pass the model
            # through untouched. This is the reliable way to disable Head
            # Freeze: ComfyUI's node bypass/mute skips apply_freeze()
            # entirely, so it can't clean up a stale patch from an earlier
            # run — clearing `targets` (not bypassing the node) is what
            # actually turns the effect off.
            patched = model.clone()
            unwrap_diffusion_model(patched.model.diffusion_model)
            print("[LTXProfiler] HeadFreeze: no targets, passing model through unmodified.")
            return (patched,)

        reg = get_registry()
        if store_handle and store_handle.strip():
            reg.switch_attn(store_handle)
        elif not reg._cur_attn:
            reg.switch_attn(reg.create("default"))

        store = AttentionStore()
        src   = store.sa if attn_type == "sa" else store.ca

        # Resolve each target's frozen map up front, grouped by block so a
        # single hook can freeze several heads of the same block at once.
        block_configs: dict[int, list] = {}
        for block_idx, head_sel in pairs:
            if block_idx not in src:
                raise ValueError(f"[Freeze] Block {block_idx} not found in store.")
            if freeze_step_source not in src[block_idx]:
                raise ValueError(
                    f"[Freeze] Step {freeze_step_source} not found for block {block_idx}. "
                    f"Available: {sorted(src[block_idx].keys())}"
                )
            entry = src[block_idx][freeze_step_source]
            entry_map = entry.get("map")
            if entry_map is None:
                raise ValueError("[Freeze] No map. Re-run with store_mode=full_fp16 (or hybrid).")

            # "all" -> every head actually captured for this block (the
            # sparse dict's keys if hybrid+full_targets, else every row
            # of the dense tensor).
            if head_sel == "all":
                head_list = (sorted(entry_map.keys()) if isinstance(entry_map, dict)
                            else list(range(entry_map.shape[0])))
            else:
                if isinstance(entry_map, dict) and head_sel not in entry_map:
                    raise ValueError(
                        f"[Freeze] Block {block_idx} was captured with full_targets "
                        f"(specific heads only) and head {head_sel} wasn't one of them. "
                        f"Heads available for this block: {sorted(entry_map.keys())}."
                    )
                head_list = [head_sel]

            for head_idx in head_list:
                block_configs.setdefault(block_idx, []).append({
                    "head_idx":   head_idx,
                    "frozen_map": entry_map[head_idx].float(),
                })

        patched       = model.clone()
        dm            = patched.model.diffusion_model
        unwrap_diffusion_model(dm)      # clean slate — see QKVTransfer for why
        original_fwd  = dm._forward
        step_counters = {}

        def patched_forward(self_dm, x, timestep, context, attention_mask,
                            frame_rate=25, transformer_options={},
                            keyframe_idxs=None, **kwargs):

            patches_replace = dict(transformer_options.get("patches_replace", {}))
            dit_replace     = dict(patches_replace.get("dit", {}))

            for block_idx, freeze_configs in block_configs.items():
                existing = dit_replace.get(("double_block", block_idx))

                if block_idx not in step_counters:
                    step_counters[block_idx] = 0
                current_step             = step_counters[block_idx]
                step_counters[block_idx] += 1

                if current_step >= freeze_from_step:
                    dit_replace[("double_block", block_idx)] = make_freeze_hook(
                        block_idx, freeze_configs, blend_weight,
                        existing_hook=existing,
                    )
                else:
                    if existing is not None:
                        dit_replace[("double_block", block_idx)] = existing
                    else:
                        dit_replace.pop(("double_block", block_idx), None)

            patches_replace["dit"] = dit_replace
            transformer_options    = {**transformer_options,
                                      "patches_replace": patches_replace}
            return original_fwd(
                x, timestep, context, attention_mask,
                frame_rate, transformer_options, keyframe_idxs, **kwargs,
            )

        dm._forward                   = types.MethodType(patched_forward, dm)
        dm._profiler_patched          = True
        dm._profiler_original_forward = original_fwd
        summary = ", ".join(
            f"b{b}h{cfg['head_idx']}"
            for b, cfgs in block_configs.items() for cfg in cfgs
        )
        print(
            f"[LTXProfiler] HeadFreeze targets=[{summary}] "
            f"from_step={freeze_from_step} src_step={freeze_step_source} "
            f"blend={blend_weight}"
        )
        return (patched,)


class LTXQKVTransfer:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":              ("MODEL",),
            "attn_type":          (["sa", "ca"], {"default": "sa"}),
            "target_blocks":      ("STRING", {"default": "24",
                                   "tooltip": "Simple: '24,32' ou 'all'\n"
                                              "Extended: '24:8,12 | 32:all'"}),
            "head_indices":       ("STRING", {"default": "8,12,16"}),
            "source_step":        ("INT",    {"default": 0, "min": 0, "max": 255}),
            "transfer_from_step": ("INT",    {"default": 0, "min": 0, "max": 999}),
            "transfer_to_step":   ("INT",    {"default": 999, "min": 0, "max": 999}),
            "blend":              ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 1.0,
                                   "step": 0.05}),
            "use_map":            ("BOOLEAN", {"default": False}),
            "use_q":              ("BOOLEAN", {"default": False}),
            "use_k":              ("BOOLEAN", {"default": True}),
            "use_v":              ("BOOLEAN", {"default": True}),
            "sim_filter":         ("BOOLEAN", {"default": False}),
            "sim_threshold":      ("FLOAT",   {"default": 0.3,
                                   "min": -1.0, "max": 1.0, "step": 0.05}),
            "qkv_handle":         ("STRING",  {"default": "", "placeholder": "select QKV store..."}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("transfer_model",)
    FUNCTION     = "apply_transfer"
    CATEGORY     = "g_raw/LTX/Profiler"

    @staticmethod
    def _parse_block_head_mapping(target_blocks, head_indices,
                                   qkv_store, attn_type, source_step):
        store_src = qkv_store.data.get(attn_type, {})

        def resolve_heads(blk, heads_str):
            if heads_str.strip().lower() == "all":
                return sorted(store_src.get(blk, {}).get(source_step, {}).keys())
            return [int(x.strip()) for x in heads_str.split(",") if x.strip()]

        if ":" in target_blocks:
            mapping = {}
            for seg in target_blocks.split("|"):
                seg = seg.strip()
                if not seg:
                    continue
                if ":" in seg:
                    blk_str, h_str = seg.split(":", 1)
                    blk = int(blk_str.strip())
                    mapping[blk] = resolve_heads(blk, h_str)
                else:
                    blk = int(seg)
                    mapping[blk] = resolve_heads(blk, head_indices)
            return mapping

        if target_blocks.strip().lower() == "all":
            block_list = sorted(store_src.keys())
        else:
            block_list = [int(x.strip()) for x in target_blocks.split(",") if x.strip()]
        return {blk: resolve_heads(blk, head_indices) for blk in block_list}

    def apply_transfer(self, model, attn_type, target_blocks, head_indices,
                       source_step, transfer_from_step, transfer_to_step,
                       blend, use_map, use_q, use_k, use_v,
                       sim_filter, sim_threshold, qkv_handle):

        if use_map:
            use_q = use_k = use_v = False
        if not any([use_map, use_q, use_k, use_v]):
            print("[Transfer] No component selected.")
            return (model,)

        reg = get_registry()
        if qkv_handle and qkv_handle.strip():
            reg.switch_qkv(qkv_handle)
        elif not reg._cur_qkv:
            reg.switch_qkv(reg.create_qkv("default"))

        qkv_store = QKVStore()
        if not any(qkv_store.data[t] for t in qkv_store.data):
            raise ValueError("[Transfer] QKVStore is empty.")

        block_head_map = self._parse_block_head_mapping(
            target_blocks, head_indices, qkv_store, attn_type, source_step
        )
        if not block_head_map:
            raise ValueError(f"[Transfer] No blocks resolved from '{target_blocks}'.")

        # Validation
        store_src = qkv_store.data.get(attn_type, {})
        for blk_idx, heads in block_head_map.items():
            if blk_idx not in store_src:
                raise ValueError(f"[Transfer] Block {blk_idx} not found.")
            if source_step not in store_src[blk_idx]:
                raise ValueError(f"[Transfer] Step {source_step} not found for block {blk_idx}.")
            captured = sorted(store_src[blk_idx][source_step].keys())
            missing  = [h for h in heads if h not in captured]
            if missing:
                raise ValueError(f"[Transfer] Heads {missing} not found in store. Captured: {captured}")

        target_call_n    = 0 if attn_type == "sa" else 1
        per_block_configs = {
            blk: [{"head_idx": h, "source_step": source_step, "blend": blend,
                   "use_map": use_map, "use_q": use_q, "use_k": use_k, "use_v": use_v,
                   "sim_filter": sim_filter, "sim_threshold": sim_threshold}
                  for h in heads]
            for blk, heads in block_head_map.items()
        }

        patched       = model.clone()
        dm            = patched.model.diffusion_model
        unwrap_diffusion_model(dm)
        original_fwd  = dm._forward
        step_counters = {}

        def patched_forward(self_dm, x, timestep, context, attention_mask,
                            frame_rate=25, transformer_options={},
                            keyframe_idxs=None, **kwargs):

            from ..ops.qkv_transfer import apply_qkv_transfer

            patches_replace = dict(transformer_options.get("patches_replace", {}))
            dit_replace     = dict(patches_replace.get("dit", {}))

            for blk_idx, head_configs in per_block_configs.items():
                if blk_idx not in step_counters:
                    step_counters[blk_idx] = 0
                current_step             = step_counters[blk_idx]
                step_counters[blk_idx]  += 1
                in_range                 = (transfer_from_step
                                            <= current_step
                                            <= transfer_to_step)
                existing = dit_replace.get(("double_block", blk_idx))

                if in_range:
                    tcfg = {
                        "head_configs":  head_configs,
                        "qkv_store":     qkv_store,
                        "attn_type":     attn_type,
                        "target_call_n": target_call_n,
                        "block_idx":     blk_idx,
                        "current_step":  current_step,
                        "step_range":    (transfer_from_step, transfer_to_step),
                    }

                    def make_hook(cfg, existing_h, bidx):
                        def hook(args, orig):
                            to = dict(args.get("transformer_options", {}))
                            to["_qkv_block_idx"]       = bidx
                            to["_qkv_transfer_active"] = True
                            to["_qkv_transfer_cfg"]    = cfg
                            reset_call_count(bidx)
                            new_args = {**args, "transformer_options": to}
                            if existing_h and not getattr(existing_h,
                                                          "_is_transfer_hook", False):
                                return existing_h(new_args, orig)
                            return orig["original_block"](new_args)
                        hook._is_transfer_hook = True
                        return hook

                    dit_replace[("double_block", blk_idx)] = make_hook(
                        tcfg, existing, blk_idx
                    )
                else:
                    cur = dit_replace.get(("double_block", blk_idx))
                    if getattr(cur, "_is_transfer_hook", False):
                        if existing and not getattr(existing, "_is_transfer_hook", False):
                            dit_replace[("double_block", blk_idx)] = existing
                        else:
                            dit_replace.pop(("double_block", blk_idx), None)

            patches_replace["dit"] = dit_replace
            transformer_options    = {**transformer_options,
                                      "patches_replace": patches_replace}
            return original_fwd(
                x, timestep, context, attention_mask,
                frame_rate, transformer_options, keyframe_idxs, **kwargs,
            )

        dm._forward                   = types.MethodType(patched_forward, dm)
        dm._profiler_patched          = True
        dm._profiler_original_forward = original_fwd
        return (patched,)
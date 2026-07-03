from __future__ import annotations

from ..core.stores      import AttentionStore, QKVStore, get_registry
from ..core.hooks       import install_hook
from ..core.model_patch import register_layer, unregister_layer
from ..ops.freeze       import make_freeze_hook_factory
from ..utils.helpers    import reset_call_count, parse_block_head_pairs


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
                                              "'block:head | block:head | ...' (or "
                                              "'block:h1,h2,h3' for several heads of one block "
                                              "in a single entry). Use 'block:all' to freeze "
                                              "every captured head of that block. Leave blank "
                                              "to disable the freeze "
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
        },
        "hidden": {"node_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("steered_model",)
    FUNCTION     = "apply_freeze"
    CATEGORY     = "g_raw/LTX/Profiler"

    def apply_freeze(self, model, targets, freeze_from_step,
                     freeze_step_source, attn_type, blend_weight, store_handle,
                     node_id=None):

        pairs = parse_block_head_pairs(targets) if targets.strip() else []
        if not pairs:
            # Nothing to freeze — unregister just this node's own layer so a
            # previous run's patch doesn't linger, then pass the model
            # through untouched. This is the reliable way to disable Head
            # Freeze: ComfyUI's node bypass/mute skips apply_freeze()
            # entirely, so it can't clean up a stale patch from an earlier
            # run — clearing `targets` (not bypassing the node) is what
            # actually turns the effect off.
            patched = model.clone()
            unregister_layer(patched.model.diffusion_model, node_id)
            print("[LTXProfiler] HeadFreeze: no targets, passing model through unmodified.")
            return (patched,)

        install_hook()  # no-op if already installed (e.g. by Setup Capture) --
        # needed here too since this node can run without a prior Setup
        # Capture in the same session, and _freeze_* keys do nothing unless
        # optimized_attention/optimized_attention_masked are wrapped.

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

        patched = model.clone()
        dm      = patched.model.diffusion_model
        register_layer(
            dm, node_id, set(block_configs.keys()),
            make_freeze_hook_factory(block_configs, freeze_from_step, blend_weight),
        )
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
            "targets":            ("STRING", {"default": "24:8,12,16", "multiline": True,
                                   "tooltip": "Same format as Head Freeze's targets: "
                                              "'block,head' one per line (paste Head "
                                              "Candidates' candidates_csv directly), or "
                                              "'block:head1,head2,... | block:all | ...'. "
                                              "'all' alone (nothing else) means every "
                                              "block/head captured in the QKV store. Blank "
                                              "disables (see below)."}),
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
            "handle":             ("STRING",  {"default": "", "placeholder": "select store..."}),
        },
        "hidden": {"node_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("transfer_model",)
    FUNCTION     = "apply_transfer"
    CATEGORY     = "g_raw/LTX/Profiler"

    @staticmethod
    def _resolve_targets(targets, store_src, source_step):
        """targets -> {block_idx: [head_idx, ...]}, resolving 'all' (either
        a whole-string 'all' meaning every block/head captured, or a
        per-block 'block:all') against what's actually in store_src for
        source_step."""
        if targets.strip().lower() == "all":
            return {
                blk: sorted(store_src[blk].get(source_step, {}).keys())
                for blk in sorted(store_src.keys())
            }

        block_head_map: dict[int, list] = {}
        for blk, head_sel in parse_block_head_pairs(targets):
            if head_sel == "all":
                heads = sorted(store_src.get(blk, {}).get(source_step, {}).keys())
            else:
                heads = [head_sel]
            block_head_map.setdefault(blk, []).extend(heads)
        return block_head_map

    @staticmethod
    def _make_transfer_hook_factory(per_block_configs, qkv_store, attn_type,
                                    target_call_n, transfer_from_step, transfer_to_step):
        """Returns a make_hook(block_idx, existing_hook) -> hook|None for
        register_layer(). Owns a fresh per-block step counter (one per
        node run, dropped along with the old layer on re-registration --
        same leak-safety property the old unwrap-before-wrap had, without
        wiping other nodes' layers)."""
        step_counters: dict = {}

        def make_hook(block_idx, existing_hook):
            head_configs = per_block_configs.get(block_idx)
            if head_configs is None:
                return None

            current_step             = step_counters.get(block_idx, 0)
            step_counters[block_idx] = current_step + 1
            if not (transfer_from_step <= current_step <= transfer_to_step):
                return None

            cfg = {
                "head_configs":  head_configs,
                "qkv_store":     qkv_store,
                "attn_type":     attn_type,
                "target_call_n": target_call_n,
                "block_idx":     block_idx,
                "current_step":  current_step,
                "step_range":    (transfer_from_step, transfer_to_step),
            }

            def hook(args, orig):
                to = dict(args.get("transformer_options", {}))
                to["_qkv_block_idx"]       = block_idx
                to["_qkv_transfer_active"] = True
                to["_qkv_transfer_cfg"]    = cfg
                reset_call_count(block_idx)
                new_args = {**args, "transformer_options": to}
                if existing_hook is not None:
                    return existing_hook(new_args, orig)
                return orig["original_block"](new_args)
            return hook

        return make_hook

    def apply_transfer(self, model, attn_type, targets,
                       source_step, transfer_from_step, transfer_to_step,
                       blend, use_map, use_q, use_k, use_v,
                       sim_filter, sim_threshold, handle, node_id=None):

        if use_map:
            use_q = use_k = use_v = False
        if not any([use_map, use_q, use_k, use_v]):
            return self._disable(model, node_id, "no component selected")
        if not targets.strip():
            return self._disable(model, node_id, "targets is blank")

        install_hook()  # no-op if already installed -- see HeadFreeze's note

        reg = get_registry()
        if handle and handle.strip():
            reg.switch_qkv(handle)
        elif not reg._cur_qkv:
            reg.switch_qkv(reg.create_qkv("default"))

        qkv_store = QKVStore()
        if not any(qkv_store.data[t] for t in qkv_store.data):
            raise ValueError("[Transfer] QKVStore is empty.")

        store_src = qkv_store.data.get(attn_type, {})
        block_head_map = self._resolve_targets(targets, store_src, source_step)
        if not block_head_map:
            raise ValueError(f"[Transfer] No blocks/heads resolved from '{targets}'.")

        # Validation
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

        patched = model.clone()
        dm      = patched.model.diffusion_model
        register_layer(
            dm, node_id, set(per_block_configs.keys()),
            self._make_transfer_hook_factory(
                per_block_configs, qkv_store, attn_type, target_call_n,
                transfer_from_step, transfer_to_step,
            ),
        )
        return (patched,)

    @staticmethod
    def _disable(model, node_id, reason: str):
        """Clone + unregister-own-layer + pass through untouched.
        diffusion_model is shared across every model.clone() in the
        session, so silently returning the input `model` here (as this
        used to do) can leave a previous run's transfer patch in effect.
        Reached only when this node's own code actually runs (unlike
        ComfyUI's bypass/mute, which skips it entirely and can't clean up
        a stale patch either)."""
        print(f"[LTXProfiler] QKVTransfer: {reason}, passing model through unmodified.")
        patched = model.clone()
        unregister_layer(patched.model.diffusion_model, node_id)
        return (patched,)


class LTXQKVMultiplier:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":       ("MODEL",),
            "targets":     ("STRING", {"default": "24:8", "multiline": True,
                            "tooltip": "Same format as Head Freeze/QKV Transfer's targets: "
                                       "paste Head Candidates' candidates_csv directly (one "
                                       "'block,head' per line), or type manually as "
                                       "'block:head | block:h1,h2,... | block:all | ...'. A "
                                       "whole-string 'all' (or 'all:all') targets every block "
                                       "(0-47) and head (0-31). No prior capture needed -- this "
                                       "is a live multiply, not a replay. Leave blank to disable "
                                       "entirely -- this is the reliable way to turn it off; "
                                       "ComfyUI's node bypass/mute skips this node's cleanup, so "
                                       "the diffusion_model (shared across runs) can be left "
                                       "patched from a previous run."}),
            "apply_sa":    ("BOOLEAN", {"default": True}),
            "apply_ca":    ("BOOLEAN", {"default": True}),
            "qk_mult":     ("FLOAT",   {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                            "tooltip": "Scales Q for the targeted heads before the attention "
                                       "dot product -- changes attention sharpness (softmax "
                                       "logit scale), not the head's output magnitude. 0 makes "
                                       "the head attend uniformly over all keys, it does NOT "
                                       "ablate it (still contributes via V). Scaling Q and K "
                                       "separately would be redundant -- both are uniform "
                                       "per-head scalars, so only their product changes the "
                                       "logits, hence a single knob."}),
            "vo_mult":     ("FLOAT",   {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                            "tooltip": "Scales the targeted head's output (V before the "
                                       "attn_weights@V matmul, equivalently the head's slice of "
                                       "the output after -- same net effect either way since the "
                                       "matmul is linear in V, hence a single knob). Directly "
                                       "scales the head's contribution to the residual stream. "
                                       "0 zeroes it out (true ablation)."}),
            "from_step":   ("INT",     {"default": 0,   "min": 0, "max": 999,
                            "tooltip": "Denoising step (per targeted block) to start applying "
                                       "from. Defaults to the full range."}),
            "to_step":     ("INT",     {"default": 999, "min": 0, "max": 999,
                            "tooltip": "Denoising step (per targeted block) to stop applying "
                                       "at, inclusive. Defaults to the full range."}),
        },
        "hidden": {"node_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("steered_model",)
    FUNCTION     = "apply_multiply"
    CATEGORY     = "g_raw/LTX/Profiler"

    @staticmethod
    def _parse_targets(targets: str) -> dict:
        """Same whole-string 'all'/'all:all' convention as Setup Capture's
        qkv_targets -- no store to resolve 'all' against here, so it just
        means every block (0-47) and every head (0-31)."""
        t = targets.strip()
        if not t:
            return {}
        if t.lower() in ("all", "all:all"):
            return {b: set(range(32)) for b in range(48)}
        block_head_map: dict[int, set] = {}
        for blk, head_sel in parse_block_head_pairs(targets):
            heads = set(range(32)) if head_sel == "all" else {head_sel}
            block_head_map.setdefault(blk, set()).update(heads)
        return block_head_map

    @staticmethod
    def _make_multiply_hook_factory(per_block_configs, apply_sa, apply_ca,
                                    from_step, to_step):
        """Returns a make_hook(block_idx, existing_hook) -> hook|None for
        register_layer(). Owns a fresh per-block step counter (one per
        node run, dropped along with the old layer on re-registration)."""
        step_counters: dict = {}

        def make_hook(block_idx, existing_hook):
            head_configs = per_block_configs.get(block_idx)
            if head_configs is None:
                return None

            current_step             = step_counters.get(block_idx, 0)
            step_counters[block_idx] = current_step + 1
            if not (from_step <= current_step <= to_step):
                return None

            cfg = {
                "head_configs": head_configs,
                "apply_sa":     apply_sa,
                "apply_ca":     apply_ca,
            }

            def hook(args, orig):
                to = dict(args.get("transformer_options", {}))
                to["_qkvmul_block_idx"] = block_idx
                to["_qkvmul_active"]    = True
                to["_qkvmul_cfg"]       = cfg
                reset_call_count(block_idx)
                new_args = {**args, "transformer_options": to}
                if existing_hook is not None:
                    return existing_hook(new_args, orig)
                return orig["original_block"](new_args)
            return hook

        return make_hook

    def apply_multiply(self, model, targets, apply_sa, apply_ca,
                       qk_mult, vo_mult, from_step, to_step,
                       node_id=None):

        block_head_map = self._parse_targets(targets)
        if not block_head_map:
            # Same reliable-disable convention as Head Freeze/QKV Transfer --
            # clearing `targets` (not bypassing the node) is what actually
            # turns the effect off, since bypass/mute skips this code and
            # can't clean up a stale patch on the shared diffusion_model.
            patched = model.clone()
            unregister_layer(patched.model.diffusion_model, node_id)
            print("[LTXProfiler] QKVMultiplier: no targets, passing model through unmodified.")
            return (patched,)

        install_hook()  # no-op if already installed -- see HeadFreeze's note

        per_block_configs = {
            blk: [{"head_idx": h, "qk_mult": qk_mult, "vo_mult": vo_mult}
                  for h in sorted(heads)]
            for blk, heads in block_head_map.items()
        }

        patched = model.clone()
        dm      = patched.model.diffusion_model
        register_layer(
            dm, node_id, set(per_block_configs.keys()),
            self._make_multiply_hook_factory(
                per_block_configs, apply_sa, apply_ca, from_step, to_step,
            ),
        )
        summary = ", ".join(
            f"b{b}h{sorted(h['head_idx'] for h in cfgs)}"
            for b, cfgs in per_block_configs.items()
        )
        print(
            f"[LTXProfiler] QKVMultiplier targets=[{summary}] "
            f"qk={qk_mult} vo={vo_mult} "
            f"sa={apply_sa} ca={apply_ca} steps=[{from_step},{to_step}]"
        )
        return (patched,)

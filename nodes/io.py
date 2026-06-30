from __future__ import annotations
import os
import torch

from ..core.stores import AttentionStore, get_registry, QKVStore


class LTXAttentionStoreDump:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "output_path":       ("STRING",  {"default": "ltx_attn_profile.pt"}),
            "include_full_maps": ("BOOLEAN", {"default": True}),
            "store_handle":      ("STRING",  {"default": "", "placeholder": "select store..."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION     = "dump"
    CATEGORY     = "g_raw/LTX/Profiler"
    OUTPUT_NODE  = True

    def dump(self, output_path, include_full_maps, store_handle):
        reg = get_registry()
        if store_handle and store_handle.strip():
            reg.switch_attn(store_handle)
        elif not reg._cur_attn:
            reg.switch_attn(reg.create("default"))
        store = AttentionStore()

        def _serialize(src):
            return {
                blk: {
                    step: {k: v for k, v in entry.items()
                           if k != "map" or include_full_maps}
                    for step, entry in steps.items()
                }
                for blk, steps in src.items()
            }

        torch.save({"sa": _serialize(store.sa),
                    "ca": _serialize(store.ca),
                    "cfg": store.cfg}, output_path)
        n_sa = sum(len(s) for s in store.sa.values())
        n_ca = sum(len(s) for s in store.ca.values())
        print(f"[LTXProfiler] Dump → {output_path} | SA:{n_sa} CA:{n_ca}")
        return (output_path,)


class LTXAttentionStoreLoad:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_path":   ("STRING",  {"default": "ltx_attn_profile.pt"}),
            "merge":        ("BOOLEAN", {"default": False}),
            "store_handle": ("STRING",  {"default": "", "placeholder": "select/name store..."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION     = "load"
    CATEGORY     = "g_raw/LTX/Profiler"
    OUTPUT_NODE  = True

    def load(self, input_path, merge, store_handle):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"[LTXProfiler/Load] File not found: {input_path}")
        payload = torch.load(input_path, map_location="cpu", weights_only=False)
        reg = get_registry()
        if store_handle and store_handle.strip():
            if store_handle not in reg.list_names():
                reg.switch_attn(reg.create(store_handle))
            else:
                reg.switch_attn(store_handle)
        elif not reg._cur_attn:
            reg.switch_attn(reg.create('default'))
        store = AttentionStore()
        if not merge:
            store.sa  = payload.get("sa",  {})
            store.ca  = payload.get("ca",  {})
            store.cfg = payload.get("cfg", {})
        else:
            for blk, steps in payload.get("sa", {}).items():
                store.sa.setdefault(blk, {}).update(steps)
            for blk, steps in payload.get("ca", {}).items():
                store.ca.setdefault(blk, {}).update(steps)

        n_sa    = sum(len(s) for s in store.sa.values())
        n_ca    = sum(len(s) for s in store.ca.values())
        summary = (f"Loaded: {input_path}\n"
                   f"SA: {len(store.sa)} blocks {n_sa} entries | "
                   f"CA: {len(store.ca)} blocks {n_ca} entries")
        print(f"[LTXProfiler] {summary}")
        return (summary,)


class LTXQKVDump:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "output_path": ("STRING", {"default": "ltx_qkv_source.pt"}),
            "qkv_handle":  ("STRING", {"default": "", "placeholder": "select QKV store..."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION     = "dump"
    CATEGORY     = "g_raw/LTX/Profiler"
    OUTPUT_NODE  = True

    def dump(self, output_path, qkv_handle):
        reg = get_registry()
        if qkv_handle and qkv_handle.strip():
            reg.switch_qkv(qkv_handle)
        elif not reg._cur_qkv:
            reg.switch_qkv(reg.create_qkv('default'))
        store = QKVStore()
        torch.save({"data": store.data, "cfg": store.cfg}, output_path)
        print(f"[LTXProfiler/QKV] Dump → {output_path}")
        return (output_path,)


class LTXQKVLoad:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_path": ("STRING",  {"default": "ltx_qkv_source.pt"}),
            "merge":      ("BOOLEAN", {"default": False}),
            "qkv_handle": ("STRING",  {"default": "", "placeholder": "select/name QKV store..."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("load_summary",)
    FUNCTION     = "load"
    CATEGORY     = "g_raw/LTX/Profiler"
    OUTPUT_NODE  = True

    def load(self, input_path, merge, qkv_handle):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"[LTXProfiler/QKVLoad] File not found: {input_path}")
        payload = torch.load(input_path, map_location="cpu", weights_only=False)
        reg = get_registry()
        if qkv_handle and qkv_handle.strip():
            if qkv_handle not in reg.list_qkv_names():
                reg.switch_qkv(reg.create_qkv(qkv_handle))
            else:
                reg.switch_qkv(qkv_handle)
        elif not reg._cur_qkv:
            reg.switch_qkv(reg.create_qkv('default'))
        store = QKVStore()
        if not merge:
            store.reset()
            store.data = payload["data"]
            store.cfg  = payload.get("cfg", {})
        else:
            for blk, steps in payload["data"].items():
                store.data.setdefault(blk, {}).update(steps)
        summary = f"QKV loaded: {input_path}"
        print(f"[LTXProfiler] {summary}")
        return (summary,)
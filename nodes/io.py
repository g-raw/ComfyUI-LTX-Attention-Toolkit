from __future__ import annotations
import os
import torch

from ..core.stores import AttentionStore, get_registry, QKVStore


def _serialize_attn(src, include_full_maps):
    return {
        blk: {
            step: {k: v for k, v in entry.items()
                   if k != "map" or include_full_maps}
            for step, entry in steps.items()
        }
        for blk, steps in src.items()
    }


class LTXStoreDump:
    """Save the current AttentionStore and/or QKVStore to a single .pt file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "output_path":       ("STRING",  {"default": "ltx_store.pt"}),
            "include_full_maps": ("BOOLEAN", {"default": True}),
            "handle":            ("STRING",  {"default": "",
                                  "placeholder": "store name (blank = current, if any)"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION     = "dump"
    CATEGORY     = "g_raw/LTX/Profiler"
    OUTPUT_NODE  = True

    def dump(self, output_path, include_full_maps, handle):
        reg     = get_registry()
        payload = {}
        summary = []

        h = handle.strip()
        if h:
            inst = reg._get_attn(h)  # explicit handle: raise clearly if it's a typo
        elif reg._cur_attn:
            inst = reg._get_attn(reg._cur_attn)
        else:
            inst = None
        if inst is not None:
            payload["attn"] = {
                "sa":  _serialize_attn(inst.sa, include_full_maps),
                "ca":  _serialize_attn(inst.ca, include_full_maps),
                "cfg": inst.cfg,
            }
            n_sa = sum(len(s) for s in inst.sa.values())
            n_ca = sum(len(s) for s in inst.ca.values())
            summary.append(f"attn '{inst.name}' SA:{n_sa} CA:{n_ca}")

        # Same name as the attn store, since Setup Capture's single
        # `handle` output covers both -- but QKV is optional (only
        # exists if capture_qkv was on), so a miss here isn't a typo,
        # just silently skip instead of raising like the attn lookup.
        if h:
            qinst = reg._qkv.get(h)
        elif reg._cur_qkv:
            qinst = reg._get_qkv(reg._cur_qkv)
        else:
            qinst = None
        if qinst is not None:
            payload["qkv"] = {"data": qinst.data, "cfg": qinst.cfg}
            summary.append(f"qkv '{qinst.name}'")

        if not payload:
            raise ValueError(
                f"[Dump] No attn or QKV store found under handle '{h}' — "
                "capture something first."
            )

        torch.save(payload, output_path)
        print(f"[LTXProfiler] Dump → {output_path} | " + " | ".join(summary))
        return (output_path,)


class LTXStoreLoad:
    """Load an AttentionStore and/or QKVStore section from a .pt file saved by Store Dump."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_path": ("STRING",  {"default": "ltx_store.pt"}),
            "merge":      ("BOOLEAN", {"default": False}),
            "handle":     ("STRING",  {"default": "",
                           "placeholder": "name for the store (blank = 'default')"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "handle")
    FUNCTION     = "load"
    CATEGORY     = "g_raw/LTX/Profiler"
    OUTPUT_NODE  = True

    def load(self, input_path, merge, handle):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"[LTXProfiler/Load] File not found: {input_path}")
        payload = torch.load(input_path, map_location="cpu", weights_only=False)
        reg = get_registry()

        name          = handle.strip() or "default"
        summary_parts = []

        if "attn" in payload:
            reg.create(name)
            store  = AttentionStore()
            section = payload["attn"]
            if not merge:
                store.sa  = section.get("sa",  {})
                store.ca  = section.get("ca",  {})
                store.cfg = section.get("cfg", {})
            else:
                for blk, steps in section.get("sa", {}).items():
                    store.sa.setdefault(blk, {}).update(steps)
                for blk, steps in section.get("ca", {}).items():
                    store.ca.setdefault(blk, {}).update(steps)
            n_sa = sum(len(s) for s in store.sa.values())
            n_ca = sum(len(s) for s in store.ca.values())
            summary_parts.append(
                f"attn '{name}': SA {len(store.sa)} blocks {n_sa} entries | "
                f"CA {len(store.ca)} blocks {n_ca} entries"
            )

        if "qkv" in payload:
            reg.create_qkv(name)
            qstore  = QKVStore()
            section = payload["qkv"]
            if not merge:
                qstore.reset()
                qstore.data = section["data"]
                qstore.cfg  = section.get("cfg", {})
            else:
                for blk, steps in section["data"].items():
                    qstore.data.setdefault(blk, {}).update(steps)
            summary_parts.append(f"qkv '{name}' loaded")

        if not summary_parts:
            raise ValueError(f"[Load] '{input_path}' contains neither an attn nor a QKV section.")

        summary = f"Loaded: {input_path}\n" + "\n".join(summary_parts)
        print(f"[LTXProfiler] {summary}")
        return (summary, name)

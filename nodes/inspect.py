from __future__ import annotations

from ..core.stores import AttentionStore, get_registry, QKVStore


class LTXAttentionStoreInspect:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"trigger": ("*", {})}}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("store_summary",)
    FUNCTION     = "inspect"
    CATEGORY     = "g_raw/LTX/Profiler"
    OUTPUT_NODE  = True

    def inspect(self, **kwargs):
        reg = get_registry()
        h = reg._cur_attn or reg.create('default')
        reg.switch_attn(h)
        store = AttentionStore()

        lines = ["=" * 50, "  AttentionStore Inspect", "=" * 50]
        lines.append(f"cfg: {store.cfg or '⚠ EMPTY'}")
        lines.append("")

        for label, src in [("SA", store.sa), ("CA", store.ca)]:
            lines.append(f"── {label} ──")
            if not src:
                lines.append("  (empty)")
                continue
            for blk in sorted(src.keys()):
                steps = src[blk]
                if not steps:
                    continue
                last    = steps[max(steps.keys())]
                n_heads = len(last.get("entropy", []))
                has_map = last.get("map") is not None
                ts      = last.get("timestep", "?")
                ts_str  = f"{ts:.3f}" if isinstance(ts, float) else str(ts)
                lines.append(
                    f"  Block {blk:2d}: {len(steps)} steps | "
                    f"{n_heads} heads | ts≈{ts_str} | "
                    f"map={'present '+str(list(last['map'].shape)) if has_map else 'absent'}"
                )

        summary = "\n".join(lines)
        print(summary)
        return (summary,)


class LTXQKVStoreInspect:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"trigger": ("*", {})}}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("store_summary",)
    FUNCTION     = "inspect"
    CATEGORY     = "g_raw/LTX/Profiler"
    OUTPUT_NODE  = True

    def inspect(self, **kwargs):
        reg = get_registry()
        h = reg._cur_qkv or reg.create_qkv('default')
        reg.switch_qkv(h)
        store = QKVStore()

        lines = ["=" * 50, "  QKVStore Inspect", "=" * 50]
        total_mb = 0.0

        for atype in sorted(store.data.keys()):
            lines.append(f"── {atype} ──")
            for blk in sorted(store.data[atype].keys()):
                for step in sorted(store.data[atype][blk].keys()):
                    heads = sorted(store.data[atype][blk][step].keys())
                    if not heads:
                        continue
                    sample  = store.data[atype][blk][step][heads[0]]
                    step_mb = sum(
                        store.data[atype][blk][step][h][t].numel() *
                        store.data[atype][blk][step][h][t].element_size()
                        for h in heads for t in ("q", "k", "v")
                    ) / 1e6
                    total_mb += step_mb
                    lines.append(
                        f"  Block {blk} Step {step} | "
                        f"heads={heads} | Q={list(sample['q'].shape)} | "
                        f"~{step_mb:.2f}MB"
                    )

        lines.append(f"\nTotal: ~{total_mb:.1f} MB")
        summary = "\n".join(lines)
        print(summary)
        return (summary,)


class LTXMapStoreInspect:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"map_store": ("ATTN_MAP_STORE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION     = "inspect"
    CATEGORY     = "g_raw/LTX/Profiler"
    OUTPUT_NODE  = True

    def inspect(self, map_store):
        if not map_store:
            msg = "⚠ map_store EMPTY"
            print(f"[MapStoreInspect] {msg}")
            return (msg,)

        lines = ["── Map Store Inspect ──"]
        total_entries = 0
        for blk in sorted(map_store.keys()):
            for step in sorted(map_store[blk].keys()):
                heads = map_store[blk][step]
                n     = len(heads)
                total_entries += n
                if heads:
                    sample = next(iter(heads.values()))
                    views  = [k for k in sample if k != "timestep"]
                    shapes = {k: list(sample[k].shape)
                              for k in views if hasattr(sample[k], "shape")}
                    lines.append(
                        f"  Block {blk:2d} | Step {step} | "
                        f"{n} heads | views={views} | shapes={shapes}"
                    )

        lines.append(f"Total: {total_entries} entries")
        summary = "\n".join(lines)
        print(f"[MapStoreInspect]\n{summary}")
        return (summary,)

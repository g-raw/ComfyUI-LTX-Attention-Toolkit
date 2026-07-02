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
                last     = steps[max(steps.keys())]
                n_heads  = len(last.get("entropy", []))
                map_val  = last.get("map")
                has_kq   = last.get("key_map") is not None and last.get("query_map") is not None
                ts       = last.get("timestep", "?")
                ts_str   = f"{ts:.3f}" if isinstance(ts, float) else str(ts)
                if isinstance(map_val, dict):
                    map_str = f"present (sparse, heads={sorted(map_val.keys())})"
                elif map_val is not None:
                    map_str = f"present {list(map_val.shape)}"
                else:
                    map_str = "absent"
                lines.append(
                    f"  Block {blk:2d}: {len(steps)} steps | "
                    f"{n_heads} heads | ts≈{ts_str} | "
                    f"map={map_str} | "
                    f"key/query_map={'present' if has_kq else 'absent'}"
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

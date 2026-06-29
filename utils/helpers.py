from __future__ import annotations
import threading
from typing import Optional

# ── Call counter thread-local ──────────────────────────────────────────────

_CALL_COUNTER_TL = threading.local()


def get_call_counts() -> dict:
    if not hasattr(_CALL_COUNTER_TL, "counts"):
        _CALL_COUNTER_TL.counts = {}
    return _CALL_COUNTER_TL.counts


def reset_call_count(block_idx: int):
    get_call_counts()[f"block_{block_idx}"] = 0


def increment_call_count(block_idx: int) -> int:
    """Increments and returns the NEW value (post-increment)."""
    counts = get_call_counts()
    key    = f"block_{block_idx}"
    n      = counts.get(key, 0)
    counts[key] = n + 1
    return n   # returns the old value = current call_n


# ── Helpers store ──────────────────────────────────────────────────────────

def resolve_entry(src: dict, block_idx: int, step_idx: int, attn_type: str) -> dict:
    """Resolve block/step in a SA or CA store, raises ValueError if not found."""
    if not src:
        raise ValueError(f"No {attn_type} data captured.")
    if block_idx not in src:
        raise ValueError(
            f"Block {block_idx} not found. Available: {sorted(src.keys())}"
        )
    block_data = src[block_idx]
    steps      = sorted(block_data.keys())
    if step_idx == -1:
        step_idx = steps[-1]
    if step_idx not in block_data:
        raise ValueError(f"Step {step_idx} not found. Available: {steps}")
    entry = block_data[step_idx]
    if entry.get("map") is None:
        raise ValueError("No map. Re-run with store_full_maps=True.")
    return entry


def parse_heads(head_indices: str, H_heads: int) -> list:
    if head_indices.strip().lower() == "all":
        return list(range(H_heads))
    head_list = [
        int(x.strip()) for x in head_indices.split(",") if x.strip()
    ]
    head_list = [h for h in head_list if 0 <= h < H_heads]
    if not head_list:
        raise ValueError("No valid heads.")
    return head_list


def parse_int_set(s: str, all_range: Optional[range] = None) -> Optional[set]:
    """
    Parse a string into an int set.
    'all' → returns None (meaning "all") or set(all_range) if provided.
    Returns None for empty input after stripping.
    """
    if s.strip().lower() == "all":
        return None if all_range is None else set(all_range)
    result = set(int(x.strip()) for x in s.split(",") if x.strip())
    if not result:
        return None
    return result


def log_node(node_name: str, attn_type: str, block_idx: int,
             step_idx: int, entry: dict, n_heads: int,
             T: int, Lh: int, Lw: int):
    ts     = entry.get("timestep", "?")
    ts_str = f"{ts:.3f}" if isinstance(ts, float) else str(ts)
    print(
        f"[LTXProfiler] {node_name} — {attn_type} | "
        f"bloc {block_idx} | step {step_idx} | timestep≈{ts_str} | "
        f"{n_heads} têtes | latent {T}×{Lh}×{Lw}"
    )
from __future__ import annotations
import threading
from typing import Dict, Optional, Any

import torch
import torch.nn.functional as F


# ===== Internal data containers (one per named store) =====

class _AttnInst:
    """Data container for one named AttentionStore."""
    __slots__ = ("name", "sa", "ca", "_step_counter", "cfg", "_save_callback", "_parsed_heads")

    def __init__(self, name: str):
        self.name         = name
        self.sa           = {}   # block_idx → step_idx → entry
        self.ca           = {}
        self._step_counter: Dict[str, int] = {}
        self.cfg          = dict()
        self._save_callback = None
        self._parsed_heads  = None

    def reset_data(self):
        self.sa.clear()
        self.ca.clear()
        self._step_counter.clear()


class _QKVInst:
    """Data container for one named QKVStore."""
    __slots__ = ("name", "data", "_step_counter", "cfg")

    def __init__(self, name: str):
        self.name          = name
        self.data          = {"sa": {}, "ca": {}}   # block_idx → step_idx → head_idx → data
        self._step_counter: Dict[str, int] = {}
        self.cfg           = dict()

    def reset_data(self):
        self.data["sa"].clear()
        self.data["ca"].clear()
        self._step_counter.clear()


# ===== StoreRegistry — singleton manager =====

class StoreRegistry:
    """Manages named instances. Thread-safe."""

    def __init__(self):
        self._lock         = threading.Lock()
        self._attn: Dict[str, _AttnInst] = {}
        self._qkv:  Dict[str, _QKVInst]  = {}
        self._cur_attn: Optional[str] = None  # handle (name) of current AttentionStore
        self._cur_qkv:  Optional[str] = None  # handle (name) of current QKVStore

    def create(self, name: Optional[str] = None) -> str:
        """Create a new named AttentionStore. Returns the handle (unique name). Auto-selects as current."""
        with self._lock:
            if not name:
                h = f"store_{id(self._attn)}"
                i = 2
                while h in self._attn:
                    h = f"store_{id(self._attn)}_{i}"
                    i += 1
                name = h
            else:
                base = name
                i = 2
                while name in self._attn:
                    name = f"{base}_{i}"
                    i += 1
            inst = _AttnInst(name)
            self._attn[name] = inst
            self._cur_attn = name
        return name

    def create_qkv(self, name: Optional[str] = None) -> str:
        """Create a new named QKVStore. Returns the handle."""
        with self._lock:
            if not name:
                h = f"qkv_{id(self._qkv)}"
                i = 2
                while h in self._qkv:
                    h = f"qkv_{id(self._qkv)}_{i}"
                    i += 1
                name = h
            else:
                base = name
                i = 2
                while name in self._qkv:
                    name = f"{base}_{i}"
                    i += 1
            inst = _QKVInst(name)
            self._qkv[name] = inst
            self._cur_qkv = name
        return name

    def switch_attn(self, handle: str):
        """Set current AttentionStore by handle. Raises KeyError if not found."""
        with self._lock:
            if handle not in self._attn:
                raise KeyError(f"AttentionStore '{handle}' does not exist.")
            self._cur_attn = handle

    def switch_qkv(self, handle: str):
        """Set current QKVStore by handle. Raises KeyError if not found."""
        with self._lock:
            if handle not in self._qkv:
                raise KeyError(f"QKVStore '{handle}' does not exist.")
            self._cur_qkv = handle

    def list_names(self) -> list:
        """Return list of existing AttentionStore names."""
        with self._lock:
            return list(self._attn.keys())

    def list_qkv_names(self) -> list:
        """Return list of existing QKVStore names."""
        with self._lock:
            return list(self._qkv.keys())

    def delete(self, handle: str) -> bool:
        """Remove an AttentionStore by handle. Returns True if deleted."""
        with self._lock:
            if handle in self._attn:
                del self._attn[handle]
                if self._cur_attn == handle:
                    self._cur_attn = None
                return True
            return False

    def delete_qkv(self, handle: str) -> bool:
        """Remove a QKVStore by handle. Returns True if deleted."""
        with self._lock:
            if handle in self._qkv:
                del self._qkv[handle]
                if self._cur_qkv == handle:
                    self._cur_qkv = None
                return True
            return False

    def _get_attn(self, handle: str) -> _AttnInst:
        if handle not in self._attn:
            raise KeyError(f"AttentionStore '{handle}' not found")
        return self._attn[handle]

    def _get_qkv(self, handle: str) -> _QKVInst:
        if handle not in self._qkv:
            raise KeyError(f"QKVStore '{handle}' not found")
        return self._qkv[handle]


_REGISTRY: Optional[StoreRegistry] = None


def get_registry() -> StoreRegistry:
    """Thread-safe global singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        with threading.Lock():
            if _REGISTRY is None:
                _REGISTRY = StoreRegistry()
    return _REGISTRY


# ===== Proxy wrappers — behave like old singletons but delegate to registry =====

class AttentionStore:
    """Proxy for the currently-selected AttentionStore instance. Backward compat with old get()."""

    @property
    def _inst(self) -> _AttnInst:
        reg = get_registry()
        h = reg._cur_attn
        if not h:
            h = reg.create("default")
            reg.switch_attn(h)
        return reg._get_attn(h)

    @property
    def sa(self):
        return self._inst.sa

    @sa.setter
    def sa(self, v):
        self._inst.sa = v

    @property
    def ca(self):
        return self._inst.ca

    @ca.setter
    def ca(self, v):
        self._inst.ca = v

    @property
    def cfg(self):
        return self._inst.cfg

    @cfg.setter
    def cfg(self, v):
        self._inst.cfg = v

    @property
    def _step_counter(self):
        return self._inst._step_counter

    @property
    def name(self):
        return self._inst.name

    @property
    def _save_callback(self):
        return self._inst._save_callback

    @_save_callback.setter
    def _save_callback(self, v):
        self._inst._save_callback = v

    @property
    def _parsed_heads(self):
        return self._inst._parsed_heads

    @_parsed_heads.setter
    def _parsed_heads(self, v):
        self._inst._parsed_heads = v

    def reset(self):
        self._inst.reset_data()

    def record(self, attn_type: str, block_idx: int, timestep: float,
               attn_weights: torch.Tensor, num_frames: int, patches_per_frame: int):
        """Delegate to _attn_record helper."""
        _attn_record(self._inst, attn_type, block_idx, timestep, attn_weights, num_frames, patches_per_frame)


class QKVStore:
    """Proxy for the currently-selected QKVStore instance. Backward compat with old get()."""

    @property
    def _inst(self) -> _QKVInst:
        reg = get_registry()
        h = reg._cur_qkv
        if not h:
            h = reg.create_qkv("default")
            reg.switch_qkv(h)
        return reg._get_qkv(h)

    @property
    def data(self):
        return self._inst.data

    @data.setter
    def data(self, v):
        self._inst.data = v

    @property
    def cfg(self):
        return self._inst.cfg

    @cfg.setter
    def cfg(self, v):
        self._inst.cfg = v

    @property
    def _step_counter(self):
        return self._inst._step_counter

    @property
    def name(self):
        return self._inst.name

    def reset(self):
        self._inst.reset_data()

    def record(self, attn_type: str, block_idx: int, timestep: float,
               q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int):
        """Delegate to _qkv_record helper."""
        _qkv_record(self._inst, attn_type, block_idx, timestep, q, k, v, heads)

    def get_qkv(self, attn_type: str, block_idx: int, step_idx: int, head_idx: int):
        """Retrieve captured Q/K/V tensors for a specific head/step/block."""
        try:
            e = self._inst.data[attn_type][block_idx][step_idx][head_idx]
            return e["q"].float(), e["k"].float(), e["v"].float()
        except (KeyError, TypeError, AttributeError):
            return None


# ===== Standalone record implementations =====

def _attn_record(inst: _AttnInst, attn_type: str, block_idx: int, timestep: float,
                 attn_weights: torch.Tensor, num_frames: int, patches_per_frame: int):
    """Record one attention step into an _AttnInst. Identical logic to the old AttentionStore.record()."""
    cfg = inst.cfg
    if not cfg:
        return

    target_blocks = cfg.get("target_blocks")
    if target_blocks is not None and block_idx not in target_blocks:
        return
    if attn_type == "sa" and not cfg.get("capture_sa", True):
        return
    if attn_type == "ca" and not cfg.get("capture_ca", True):
        return

    step_key = f"{attn_type}_{block_idx}"
    n = inst._step_counter.get(step_key, 0)
    step_idx = n + 1
    inst._step_counter[step_key] = step_idx

    capture_steps = cfg.get("capture_steps")
    if capture_steps is not None and step_idx not in capture_steps:
        return

    store_dict = inst.sa if attn_type == "sa" else inst.ca
    W = attn_weights.detach()
    H_heads, Sq, Sk = W.shape
    CHUNK = 4

    # ── Entropy ─────────────────────────────────────────────────────────
    eps = 1e-6
    entropy_list: list[torch.Tensor] = []
    for h0 in range(0, H_heads, CHUNK):
        h1  = min(h0 + CHUNK, H_heads)
        wc  = W[h0:h1].float()
        ent = -(wc * (wc + eps).log()).sum(dim=-1).mean(dim=-1)
        entropy_list.append(ent.cpu())
        del wc
    entropy = torch.cat(entropy_list)
    del entropy_list

    # ── Temporal / spatial locality ─────────────────────────────────────
    temporal_scores = torch.zeros(H_heads)
    spatial_scores  = torch.zeros(H_heads)

    if attn_type == "sa" and patches_per_frame > 1 and num_frames > 1:
        expected = num_frames * patches_per_frame
        if Sq == expected and Sk == expected:
            F_, P = num_frames, patches_per_frame
            for h0 in range(0, H_heads, CHUNK):
                h1   = min(h0 + CHUNK, H_heads)
                wc   = W[h0:h1].float()
                W_r  = wc.view(h1 - h0, F_, P, F_, P)
                intra = torch.diagonal(W_r, dim1=1, dim2=3)
                intra_m = intra.sum(dim=(1, 2, 3)).cpu()
                spatial_scores[h0:h1]  = intra_m
                temporal_scores[h0:h1] = 1.0 - intra_m
                del wc, W_r, intra

    # ── Sink mass ───────────────────────────────────────────────────────
    sink_mass = (W[:, :, 0].mean(dim=-1) + W[:, :, -1].mean(dim=-1)).cpu()

    # ── Full map ────────────────────────────────────────────────────────
    full_map = None
    if cfg.get("store_full_maps", False):
        ds = cfg.get("map_downsample", 1)
        if ds > 1 and Sq > ds and Sk > ds:
            full_map = F.avg_pool2d(
                W.unsqueeze(0).float(), kernel_size=ds, stride=ds
            ).squeeze(0).half().cpu()
        else:
            full_map = W.half().cpu()

    del W
    torch.cuda.empty_cache()

    entry = {
        "map":      full_map,
        "entropy":  entropy,
        "temporal": temporal_scores,
        "spatial":  spatial_scores,
        "sink":     sink_mass,
        "timestep": timestep,
        "step_idx": step_idx,
    }

    if block_idx not in store_dict:
        store_dict[block_idx] = {}
    store_dict[block_idx][step_idx] = entry


def _qkv_record(inst: _QKVInst, attn_type: str, block_idx: int, timestep: float,
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int):
    """Record one QKV capture step into a _QKVInst. Identical logic to the old QKVStore.record()."""
    cfg = inst.cfg
    if not cfg:
        return

    target_blocks = cfg.get("target_blocks", set())
    if block_idx not in target_blocks:
        return

    target_heads  = cfg.get("target_heads")
    step_key      = f"qkv_{attn_type}_{block_idx}"
    n             = inst._step_counter.get(step_key, 0)
    step_idx      = n + 1
    inst._step_counter[step_key] = step_idx

    capture_steps = cfg.get("capture_steps")
    if capture_steps is not None and step_idx not in capture_steps:
        return

    B, Sq, HD = q.shape
    if HD % heads != 0:
        return
    D_head = HD // heads

    def split_heads(t: torch.Tensor) -> torch.Tensor:
        b, s, hd = t.shape
        return t[0].view(s, heads, hd // heads).permute(1, 0, 2)  # [H, S, D]

    with torch.no_grad():
        q_h = split_heads(q.detach().float())
        k_h = split_heads(k.detach().float())
        v_h = split_heads(v.detach().float())
        del q, k, v  # free original inputs early

    store_dict = inst.data[attn_type]
    if block_idx not in store_dict:
        store_dict[block_idx] = {}
    if step_idx not in store_dict[block_idx]:
        store_dict[block_idx][step_idx] = {}

    for h in range(heads):
        if target_heads is not None and h not in target_heads:
            continue
        store_dict[block_idx][step_idx][h] = {
            "q":        q_h[h].half().cpu(),
            "k":        k_h[h].half().cpu(),
            "v":        v_h[h].half().cpu(),
            "timestep": timestep,
        }

    # Free head-split tensors after extracting data (no longer needed)
    del q_h, k_h, v_h


# ===== Helper functions =====

def get_current_attn() -> AttentionStore:
    """Get proxy for current AttentionStore. (New code should prefer this.)"""
    return AttentionStore()


def get_current_qkv() -> QKVStore:
    """Get proxy for current QKVStore."""
    return QKVStore()

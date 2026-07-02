from __future__ import annotations
import math
import threading
from typing import Dict, Optional, Any

import torch
import torch.nn.functional as F


# ===== Internal data containers (one per named store) =====

class _AttnInst:
    """Data container for one named AttentionStore."""
    __slots__ = ("name", "sa", "ca", "_step_counter", "cfg")

    def __init__(self, name: str):
        self.name          = name
        self.sa            = {}   # block_idx → step_idx → entry
        self.ca            = {}
        self._step_counter : Dict[str, int] = {}  # key like "sa_N" or "ca_N"
        self.cfg           = dict()

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

    def create_and_get_attn(self, name: Optional[str] = None) -> _AttnInst:
        """Get-or-create the named instance and make it current (atomic).

        A blank name always allocates a brand-new, uniquely-named instance
        (anonymous handles never collide). An explicit name is reused if it
        already exists rather than silently suffixed — callers rely on the
        name staying stable across repeated runs (e.g. Setup Capture's
        reset_store clears *this* instance, it doesn't create a new one).
        """
        with self._lock:
            if not name:
                h = f"store_{id(self._attn)}"
                i = 2
                while h in self._attn:
                    h = f"store_{id(self._attn)}_{i}"
                    i += 1
                name = h
            if name in self._attn:
                inst = self._attn[name]
            else:
                inst = _AttnInst(name)
                self._attn[name] = inst
            self._cur_attn = name
        return inst

    def create(self, name: Optional[str] = None) -> str:
        """Get-or-create the named AttentionStore. Returns the handle. Auto-selects as current."""
        return self.create_and_get_attn(name).name

    def create_and_get_qkv(self, name: Optional[str] = None) -> _QKVInst:
        """Get-or-create the named instance and make it current (atomic).
        See create_and_get_attn for the get-or-create rationale."""
        with self._lock:
            if not name:
                h = f"qkv_{id(self._qkv)}"
                i = 2
                while h in self._qkv:
                    h = f"qkv_{id(self._qkv)}_{i}"
                    i += 1
                name = h
            if name in self._qkv:
                inst = self._qkv[name]
            else:
                inst = _QKVInst(name)
                self._qkv[name] = inst
            self._cur_qkv = name
        return inst

    def create_qkv(self, name: Optional[str] = None) -> str:
        """Get-or-create the named QKVStore. Returns the handle."""
        return self.create_and_get_qkv(name).name

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

    def reset(self):
        self._inst.reset_data()

    def record(self, attn_type: str, block_idx: int, timestep: float,
               attn_weights: torch.Tensor, num_frames: int, patches_per_frame: int,
               latent_h: int = 1, latent_w: int = 1):
        """Delegate to _attn_record helper."""
        _attn_record(self._inst, attn_type, block_idx, timestep, attn_weights,
                     num_frames, patches_per_frame, latent_h, latent_w)


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

def _reshape_spatial(vec: torch.Tensor, num_frames: int, latent_h: int, latent_w: int) -> torch.Tensor:
    """[H, S] -> [H, F, Lh, Lw] when S matches the known geometry, else a safe 1x1 fallback."""
    H_heads, S = vec.shape
    expected = num_frames * latent_h * latent_w
    if expected > 0 and S == expected:
        return vec.view(H_heads, num_frames, latent_h, latent_w)
    return vec.view(H_heads, 1, 1, S)


_FRAME_DIST_CACHE: Dict[int, torch.Tensor] = {}


def _get_frame_dist_matrix(num_frames: int) -> torch.Tensor:
    """[F, F] = |fk - fq|, content-independent (depends only on num_frames) —
    memoized since target_blocks="all" calls _attn_record many times per run
    with the same geometry."""
    cached = _FRAME_DIST_CACHE.get(num_frames)
    if cached is None:
        idx = torch.arange(num_frames, dtype=torch.float32)
        cached = (idx.view(-1, 1) - idx.view(1, -1)).abs()
        _FRAME_DIST_CACHE[num_frames] = cached
    return cached


_SPATIAL_DIST_CACHE: Dict[tuple, torch.Tensor] = {}


def _get_spatial_dist_matrix(latent_h: int, latent_w: int) -> torch.Tensor:
    """[P, P] = Euclidean distance between patch-grid positions, in patch
    units (not pixels — the latent is already patch_size-downsampled).
    Content-independent, memoized per (latent_h, latent_w) like the frame
    version above."""
    key = (latent_h, latent_w)
    cached = _SPATIAL_DIST_CACHE.get(key)
    if cached is None:
        rows = torch.arange(latent_h, dtype=torch.float32).view(-1, 1).expand(latent_h, latent_w).reshape(-1)
        cols = torch.arange(latent_w, dtype=torch.float32).view(1, -1).expand(latent_h, latent_w).reshape(-1)
        dr = rows.view(-1, 1) - rows.view(1, -1)
        dc = cols.view(-1, 1) - cols.view(1, -1)
        cached = (dr.pow(2) + dc.pow(2)).sqrt()
        _SPATIAL_DIST_CACHE[key] = cached
    return cached


def _attn_record(inst: _AttnInst, attn_type: str, block_idx: int, timestep: float,
                 attn_weights: torch.Tensor, num_frames: int, patches_per_frame: int,
                 latent_h: int = 1, latent_w: int = 1):
    """Record one attention step into an _AttnInst — metrics, reduced key/query maps,
    and (depending on store_mode) the full map."""
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

    target_heads = cfg.get("target_heads")
    if target_heads is not None:
        head_idx = sorted(h for h in target_heads if h < W.shape[0])
        W = W[head_idx]

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

    # ── Temporal / spatial locality + frame-pair / patch-pair distance ──
    # *_norm variants divide by the max possible distance (num_frames - 1
    # for frames, the patch-grid diagonal for space) so runs at different
    # durations/resolutions stay comparable — the raw fields alone aren't,
    # since e.g. a mean frame distance of 3 means something different at
    # num_frames=8 than at num_frames=32.
    temporal_scores        = torch.zeros(H_heads)
    spatial_scores         = torch.zeros(H_heads)
    frame_dist_mean        = torch.zeros(H_heads)
    frame_dist_std         = torch.zeros(H_heads)
    frame_dist_mean_norm   = torch.zeros(H_heads)
    frame_dist_std_norm    = torch.zeros(H_heads)
    spatial_dist_mean      = torch.zeros(H_heads)
    spatial_dist_std       = torch.zeros(H_heads)
    spatial_dist_mean_norm = torch.zeros(H_heads)
    spatial_dist_std_norm  = torch.zeros(H_heads)

    if attn_type == "sa" and patches_per_frame > 1 and num_frames > 1:
        expected = num_frames * patches_per_frame
        if Sq == expected and Sk == expected:
            F_, P = num_frames, patches_per_frame
            dist_mat  = _get_frame_dist_matrix(F_).to(W.device)
            frame_max = float(F_ - 1)  # num_frames > 1 guaranteed above

            want_spatial_dist = (latent_h > 0 and latent_w > 0
                                 and P == latent_h * latent_w)
            if want_spatial_dist:
                spatial_dist_mat = _get_spatial_dist_matrix(latent_h, latent_w).to(W.device)
                spatial_max = math.sqrt(latent_h ** 2 + latent_w ** 2)

            for h0 in range(0, H_heads, CHUNK):
                h1   = min(h0 + CHUNK, H_heads)
                wc   = W[h0:h1].float()
                W_r  = wc.view(h1 - h0, F_, P, F_, P)
                intra = torch.diagonal(W_r, dim1=1, dim2=3)   # [chunk, Pq, Pk, F]
                intra_m = intra.sum(dim=(1, 2, 3)).cpu()
                spatial_scores[h0:h1]  = intra_m
                temporal_scores[h0:h1] = 1.0 - intra_m

                # frame_mass[fq, fk] = total attention mass from all query
                # patches in frame fq to all key patches in frame fk.
                frame_mass = W_r.sum(dim=(2, 4))                      # [chunk, F, F]
                total_mass = frame_mass.sum(dim=(1, 2)).clamp_min(1e-8)  # [chunk]
                mean_d     = (frame_mass * dist_mat).sum(dim=(1, 2)) / total_mass
                mean_d2    = (frame_mass * dist_mat.pow(2)).sum(dim=(1, 2)) / total_mass
                var_d      = (mean_d2 - mean_d.pow(2)).clamp_min(0.0)
                std_d      = var_d.sqrt()
                frame_dist_mean[h0:h1]      = mean_d.cpu()
                frame_dist_std[h0:h1]       = std_d.cpu()
                frame_dist_mean_norm[h0:h1] = (mean_d / frame_max).cpu()
                frame_dist_std_norm[h0:h1]  = (std_d / frame_max).cpu()

                if want_spatial_dist:
                    # spatial_mass[pq, pk] = same-frame attention mass
                    # between query patch pq and key patch pk, summed
                    # across frames.
                    spatial_mass = intra.sum(dim=3)                       # [chunk, Pq, Pk]
                    total_smass  = spatial_mass.sum(dim=(1, 2)).clamp_min(1e-8)
                    mean_sd      = (spatial_mass * spatial_dist_mat).sum(dim=(1, 2)) / total_smass
                    mean_sd2     = (spatial_mass * spatial_dist_mat.pow(2)).sum(dim=(1, 2)) / total_smass
                    var_sd       = (mean_sd2 - mean_sd.pow(2)).clamp_min(0.0)
                    std_sd       = var_sd.sqrt()
                    spatial_dist_mean[h0:h1]      = mean_sd.cpu()
                    spatial_dist_std[h0:h1]       = std_sd.cpu()
                    spatial_dist_mean_norm[h0:h1] = (mean_sd / spatial_max).cpu()
                    spatial_dist_std_norm[h0:h1]  = (std_sd / spatial_max).cpu()
                    del spatial_mass, total_smass, mean_sd, mean_sd2, var_sd, std_sd

                del wc, W_r, intra, frame_mass, total_mass, mean_d, mean_d2, var_d, std_d

    # ── Sink mass ───────────────────────────────────────────────────────
    sink_mass = (W[:, :, 0].mean(dim=-1) + W[:, :, -1].mean(dim=-1)).cpu()

    # ── Reduced key/query maps (cheap, always computed) ───────────────────
    key_map   = _reshape_spatial(W.mean(dim=1).float().cpu(), num_frames, latent_h, latent_w)
    query_map = _reshape_spatial(W.mean(dim=2).float().cpu(), num_frames, latent_h, latent_w)

    # ── Full map ────────────────────────────────────────────────────────
    # Either a dense [H, Sq, Sk] tensor (full_fp16, or hybrid picking whole
    # blocks), or — when full_target_map restricts to specific heads — a
    # sparse {head_idx: [Sq, Sk]} dict, to actually save RAM instead of
    # storing a full-width tensor with unused heads. Single-head consumers
    # (Head Freeze, QKV Transfer's use_map) index `map[head_idx]` either
    # way, so they don't need to care which form it is; multi-head
    # consumers (Query/Key Map, Zone Analysis) only support the dense form.
    store_mode      = cfg.get("store_mode", "reduced")
    full_blocks     = cfg.get("full_blocks", set())
    full_target_map = cfg.get("full_target_map")
    ds              = cfg.get("map_downsample", 1)

    want_full_targets = (store_mode == "hybrid" and full_target_map is not None
                         and block_idx in full_target_map)
    want_full_block   = (store_mode == "full_fp16" or
                         (store_mode == "hybrid" and not want_full_targets
                          and block_idx in full_blocks))

    full_map = None
    if want_full_targets:
        full_map = {}
        for h in sorted(h for h in full_target_map[block_idx] if h < H_heads):
            Wh = W[h].unsqueeze(0).unsqueeze(0)  # [1, 1, Sq, Sk]
            if ds > 1 and Sq > ds and Sk > ds:
                Wh = F.avg_pool2d(Wh.float(), kernel_size=ds, stride=ds)
            full_map[h] = Wh.squeeze(0).squeeze(0).half().cpu()
    elif want_full_block:
        if ds > 1 and Sq > ds and Sk > ds:
            full_map = F.avg_pool2d(
                W.unsqueeze(0).float(), kernel_size=ds, stride=ds
            ).squeeze(0).half().cpu()
        else:
            full_map = W.half().cpu()

    del W
    torch.cuda.empty_cache()

    entry = {
        "map":                    full_map,
        "key_map":                key_map,
        "query_map":              query_map,
        "entropy":                entropy,
        "temporal":               temporal_scores,
        "spatial":                spatial_scores,
        "sink":                   sink_mass,
        "frame_dist_mean":        frame_dist_mean,
        "frame_dist_std":         frame_dist_std,
        "frame_dist_mean_norm":   frame_dist_mean_norm,
        "frame_dist_std_norm":    frame_dist_std_norm,
        "spatial_dist_mean":      spatial_dist_mean,
        "spatial_dist_std":       spatial_dist_std,
        "spatial_dist_mean_norm": spatial_dist_mean_norm,
        "spatial_dist_std_norm":  spatial_dist_std_norm,
        "timestep":               timestep,
        "step_idx":               step_idx,
    }

    if block_idx not in store_dict:
        store_dict[block_idx] = {}
    store_dict[block_idx][step_idx] = entry


def _qkv_record(inst: _QKVInst, attn_type: str, block_idx: int, timestep: float,
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int):
    """Record one QKV capture step into a _QKVInst."""
    cfg = inst.cfg
    if not cfg:
        return

    # target_block_map: {block_idx: {head_idx, ...}} — per-block head
    # selection (mirrors _attn_record's full_target_map), set by
    # LTXAttentionCaptureSetup's qkv_targets field.
    target_block_map = cfg.get("target_block_map") or {}
    if block_idx not in target_block_map:
        return
    target_heads = target_block_map[block_idx]

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
        if h not in target_heads:
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

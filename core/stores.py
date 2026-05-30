from __future__ import annotations
import threading
from typing import Dict, Optional

import torch
import torch.nn.functional as F


class AttentionStore:
    """
    Singleton — stocke les maps d'attention et métriques par bloc/step.

    Layout SA/CA :
      store[block_idx][step_idx] = {
          "map":      Optional[Tensor] [H, Sq, Sk] fp16
          "entropy":  Tensor [H]
          "temporal": Tensor [H]
          "spatial":  Tensor [H]
          "sink":     Tensor [H]
          "timestep": float
          "step_idx": int
      }
    """

    _instance: Optional["AttentionStore"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.reset()

    @classmethod
    def get(cls) -> "AttentionStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def reset(self):
        self.sa:            Dict[int, Dict[int, dict]] = {}
        self.ca:            Dict[int, Dict[int, dict]] = {}
        self._step_counter: Dict[str, int]             = {}
        self.cfg:           dict                       = {}
        # Optionnel — utilisé par LTXAttentionMapStore
        self._save_callback = None
        self._parsed_heads  = None

    def _next_step(self, key: str) -> int:
        n = self._step_counter.get(key, 0)
        self._step_counter[key] = n + 1
        return n

    def record(self, attn_type: str, block_idx: int, timestep: float,
               attn_weights: torch.Tensor,
               num_frames: int, patches_per_frame: int):
        """
        attn_weights : [H, Sq, Sk] fp16 sur GPU.
        Calcule les métriques (chunked) et stocke selon cfg.
        """
        cfg = self.cfg
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
        step_idx = self._next_step(step_key)

        capture_steps = cfg.get("capture_steps")
        if capture_steps is not None and step_idx not in capture_steps:
            return

        store_dict = self.sa if attn_type == "sa" else self.ca
        W          = attn_weights.detach()
        H_heads, Sq, Sk = W.shape
        CHUNK = 4

        # ── Entropie ────────────────────────────────────────────────────────
        eps          = 1e-6
        entropy_list = []
        for h0 in range(0, H_heads, CHUNK):
            h1  = min(h0 + CHUNK, H_heads)
            wc  = W[h0:h1].float()
            ent = -(wc * (wc + eps).log()).sum(dim=-1).mean(dim=-1)
            entropy_list.append(ent.cpu())
            del wc
        entropy = torch.cat(entropy_list)

        # ── Localité temporelle / spatiale ──────────────────────────────────
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


class QKVStore:
    """
    Singleton — stocke les tenseurs Q, K, V bruts par bloc/step/tête.

    Layout :
      data[attn_type][block_idx][step_idx][head_idx] = {
          "q": [Sq, D_head] fp16 CPU
          "k": [Sk, D_head] fp16 CPU
          "v": [Sk, D_head] fp16 CPU
          "timestep": float
      }
    """

    _instance = None
    _lock      = threading.Lock()

    def __init__(self):
        self.reset()

    @classmethod
    def get(cls) -> "QKVStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def reset(self):
        self.data:          Dict = {"sa": {}, "ca": {}}
        self._step_counter: Dict[str, int] = {}
        self.cfg:           dict           = {}

    def _next_step(self, key: str) -> int:
        n = self._step_counter.get(key, 0)
        self._step_counter[key] = n + 1
        return n

    def record(self, attn_type: str, block_idx: int, timestep: float,
               q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int):
        cfg = self.cfg
        if not cfg:
            return

        target_blocks = cfg.get("target_blocks", set())
        if block_idx not in target_blocks:
            return

        target_heads  = cfg.get("target_heads")
        step_key      = f"qkv_{attn_type}_{block_idx}"
        step_idx      = self._next_step(step_key)

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

        store_dict = self.data[attn_type]
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

    def get_qkv(self, attn_type: str, block_idx: int,
                step_idx: int, head_idx: int):
        """Retourne (q, k, v) float32 CPU ou None."""
        try:
            e = self.data[attn_type][block_idx][step_idx][head_idx]
            return e["q"].float(), e["k"].float(), e["v"].float()
        except KeyError:
            return None
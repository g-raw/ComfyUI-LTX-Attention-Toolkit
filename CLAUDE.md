# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LTX Attention Profiler** — a ComfyUI custom node suite for profiling, visualizing and steering attention heads in LTX-Video 2.3 (48 transformer blocks, 32 heads, patch_size=1).

No build system, no tests, no extra dependencies beyond `torch` + `numpy`. This is a pure ComfyUI custom node package.

## Architecture (4 layers, bottom to top)

```
nodes/   ← ComfyUI node classes (INPUT_TYPES, RETURN_TYPES, FUNCTION)
  │
ops/     ← Intervention logic (freeze, QKV transfer, map store callbacks)
  │
core/    ← Data stores + hook infrastructure
  │        stores.py  — StoreRegistry singleton manager + AttentionStore / QKVStore proxy wrappers (thread-safe)
  │        hooks.py   — Universal hook on comfy.ldm.modules.attention.optimized_attention
  │        model_patch.py — diffusion_model._forward wrapping for block injection
  │
utils/   ├── helpers.py  — parse helpers, call counter, resolve_entry
           └── graphics.py — colormaps, grid rendering, Bresenham line drawing
```

### Key design patterns

- **Named store instances via StoreRegistry**: `get_registry()` returns the singleton registry managing named `_AttnInst` / `_QKVInst` containers. Proxy classes (`AttentionStore`, `QKVStore`) expose the current instance via property dispatch — auto-create "default" if none active. Data layout: `store[attn_type][block_idx][step_idx] = {"map": Tensor, "entropy": Tensor, ...}`.
- **Universal attention hook** (`core/hooks.py:_make_full_hook`): Installed on both `optimized_attention` and `optimized_attention_masked. Priority order per call: (1) Profiling → AttentionStore, (2) QKV Capture → QKVStore, (3) QKV Transfer substitution, (4) Head Freeze map injection, (5) Normal pass-through.
- **Block-level call counting** (`utils/helpers.py:increment_call_count`): Each transformer block makes 2 attention calls per step — call_n==0 is self-attention (SA), call_n==1 is cross-attention (CA). The hook uses thread-local counters keyed by `block_idx`.
- **Diffusion model patching**: Nodes that need intervention wrap `diffusion_model._forward` via `types.MethodType`, inject `patches_replace["dit"]` into `transformer_options`, and delegate to the original. Use `wrap_diffusion_model()` / `unwrap_diffusion_model()` for setup nodes (auto-detects geometry: num_frames, latent_h, latent_w); intervention nodes (Freeze, QKVTransfer) hand-roll their own inline `patched_forward`.
- **Patching chains**: Multiple patches can stack on the same block — each layer's hook stores `existing_hook` reference and forwards to it if present. Hooks are tagged with `_is_profiler_hook`, `_is_freeze_hook`, etc. to detect chain members.

## Token layout

1 token = 1 latent pixel (SymmetricPatchifier, patch_size=1). For 1280×720 video with 16 frames:
`Seq_len = 16 × (720/32) × (1280/32) = 14080 tokens`

Attention map `W: [H, Sq, Sk] fp16`. Key map = `W.mean(dim=1)` (what's looked at). Query map = `W.mean(dim=2)` (who's looking).

## Nodes by category

### Capture
- **LTX Attn — Setup Capture** (`nodes/capture.py:LTXAttentionCaptureSetup`): Patches model to capture attention metrics (entropy/temporal/spatial/sink) plus reduced key/query maps, and optionally full attention maps per `store_mode` (`reduced`/`full_fp16`/`hybrid` + `full_blocks`/`target_heads` RAM filters). Geometry (num_frames/latent_h/latent_w) is auto-detected — no manual dims. Most nodes depend on a prior setup run, read via the `STORE_HANDLE` string output (a separate later run — see hook architecture note below).
- **LTX QKV — Capture Source** (`nodes/capture.py:LTXQKVCapture`): Captures raw Q/K/V tensors per-head.

### Visualization
- **LTX Attn — Key Map** (`nodes/visualize.py:LTXAttentionKeyMap`): "What is being looked at?" — reduces query dim, reshapes keys to [F, H_lat, W_lat]. SA only.
- **LTX Attn — Query Map** (`nodes/visualize.py:LTXAttentionQueryMap`): "Who is actively looking?" — reduces key dim, works for SA+CA.
- **LTX Attn — Metrics Heatmap** (`nodes/visualize.py:LTXAttentionMetricsViz`): 2D heatmap [blocks × heads] for entropy/temporal/spatial/sink metrics. Returns IMAGE + stats string.
- **LTX Attn — Grid Viz** (`nodes/visualize.py:LTXAttentionGridViz`): Full overview grid read from a `store_handle`. Views: key_map/query_map/diff. Supports frame modes: avg/all/sequence/specific frames. Normalize modes: global/per_cell/per_block/per_head.
- **LTX Attn — Timestep Evolution** (`nodes/evolution.py`): Line chart of metric vs denoising step per head.

### Intervention
- **LTX Attn — Head Freeze** (`nodes/transfer.py:LTXAttentionHeadFreeze`): Locks attention map for a specific head from a pivot step. Requires prior capture with `store_mode=full_fp16` (or `hybrid` for that block). Single head per instance. Optional `store_handle` to target a specific named store.
- **LTX QKV — Transfer** (`nodes/transfer.py:LTXQKVTransfer`): Injects Q/K/V from source generation into target. Supports multi-block, multi-head targeting. Modes: use_k+use_v (style), use_map (raw softmax), full QKV replace. `sim_filter` for content-preserving transfer via cosine similarity gating. Optional `qkv_handle` to target a specific named QKV store.

### IO & Debug
- **Store Dump/Load** (`nodes/io.py`): Save/load AttentionStore and QKVStore to `.pt` files. All four nodes take an optional `store_handle`/`qkv_handle` to target a specific named store instead of implicitly acting on whatever is "current" in the registry.
- **LTX Attn — Compare Runs** (`nodes/utils.py`): Diff heatmap between two capture runs.
- **Store Inspect nodes** (`nodes/inspect.py`): Print store contents for debugging (AttentionStore summary includes key/query map presence).
- **LTX — Latent Dims** (`nodes/utils.py:LTXLatentDims`): Extract T/H/W from a LATENT tensor.

## Common development tasks

### Adding a new node
1. Create a new class in the appropriate `nodes/` module with standard ComfyUI interface: `INPUT_TYPES()`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`, `CATEGORY = "g_raw/LTX/Profiler"`.
2. Import and register in `__init__.py` under `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
3. If the node reads/writes store data, instantiate via `AttentionStore()` or `QKVStore()` (proxy classes that auto-select current instance from the registry). For named stores, use `get_registry().create()` / `get_registry().switch_attn()`.

### Adding a new intervention operation
1. Implement logic in `ops/` (e.g., `freeze.py`, `qkv_transfer.py`).
2. Inject via `transformer_options` keys (follow the naming convention: `_{"opname"}_*`) in the hook chain.
3. The universal hook in `core/hooks.py:_make_full_hook` dispatches to all operations by checking these keys in priority order.

### Adding metrics
In `AttentionStore.record()` (`core/stores.py`), new metrics are computed chunked on GPU (CHUNK=4) and stored as `[H]` tensors alongside the map entry.

## RF Inversion nodes (newest addition)
- **LTX RF-Inv Forward** / **Reverse** (`nodes/rf_inversion.py`): Random Fourier feature-based forward (x0→xT) and reverse (xT→x0) samplers for inversion workflows.

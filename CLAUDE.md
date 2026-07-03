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
- **Universal attention hook** (`core/hooks.py:_make_full_hook`): Installed on both `optimized_attention` and `optimized_attention_masked. Priority order per call: (1) Profiling → AttentionStore, (2) QKV Capture → QKVStore, (3) QKV Transfer substitution, (4) QKV Multiplier per-head Q/K/V/O scaling, (5) Head Freeze map injection, (6) Normal pass-through. `install_hook()` (idempotent, `_HOOK_ACTIVE`-gated) must run before any of this does anything -- Setup Capture calls it, and so does every intervention node (Head Freeze, QKV Transfer, QKV Multiplier) on its non-disabled path, since none of them can assume Setup Capture already ran earlier in the same session.
- **Block-level call counting** (`utils/helpers.py:increment_call_count`): Each transformer block makes 2 attention calls per step — call_n==0 is self-attention (SA), call_n==1 is cross-attention (CA). The hook uses thread-local counters keyed by `block_idx`.
- **Diffusion model patching — shared layer registry** (`core/model_patch.py`): every intervention/capture node registers its own contribution into `dm._ltx_layers` (an insertion-ordered dict keyed by ComfyUI's stable per-node `unique_id`) via `register_layer(dm, node_id, blocks, make_hook, on_call=None)` instead of hand-rolling its own `dm._forward` wrap. A single shared `_forward` wrap (installed once, idempotent via `_install_layer_runner`) rebuilds `patches_replace["dit"]` every call by iterating `dm._ltx_layers` in insertion order, chaining each layer's `make_hook(block_idx, existing_hook) -> hook|None` onto whatever the previous layer already contributed for that block (`None` means "don't touch this call", e.g. a step-range gate that isn't active yet — falls through to the earlier layer/original block), then delegates to the pristine forward captured once on first install (`dm._ltx_pristine_forward`, never reassigned again). Re-registering the same `node_id` (a node re-running on requeue) replaces its dict entry in place — old closures (and any tensors they hold, e.g. `frozen_map`) are dropped for GC without touching other nodes' layers, which is what lets multiple *different* node instances coexist and chain within one run while still avoiding the old wrap-on-top-of-wrap leak across repeated runs of the *same* node. `unregister_layer(dm, node_id)` removes only that node's own layer (used by every disabled/blank-input path). `reset_all_layers(dm)` (exposed via the `LTX Attn — Reset Patches` node) is the manual escape hatch for a layer orphaned by deleting/rewiring a node out of the graph — nothing calls `unregister_layer` for a `node_id` that stops executing.
- **Setup Capture's geometry tracking**: `make_profiler_hook_factory()` builds the profiler/QKV-capture block hook; geometry (`num_frames`/`latent_h`/`latent_w`) and `timestep` are refreshed once per `_forward` call via the `on_call` callback passed to `register_layer`, written into shared mutable-box lists (`timestep_ref`, `num_frames_ref`, etc.) that the block hook closures read.

## Token layout

1 token = 1 latent pixel (SymmetricPatchifier, patch_size=1). For 1280×720 video with 16 frames:
`Seq_len = 16 × (720/32) × (1280/32) = 14080 tokens`

Attention map `W: [H, Sq, Sk] fp16`. Key map = `W.mean(dim=1)` (what's looked at). Query map = `W.mean(dim=2)` (who's looking).

## Nodes by category

### Capture
- **LTX Attn — Setup Capture** (`nodes/capture.py:LTXAttentionCaptureSetup`): Patches model to capture attention metrics (entropy/temporal/spatial/sink/frame_dist_mean/frame_dist_std/spatial_dist_mean/spatial_dist_std (+ frame_dist_*_norm/spatial_dist_*_norm, divided by max possible distance for cross-run comparability)) plus reduced key/query maps, and optionally full attention maps per `store_mode` (`reduced`/`full_fp16`/`hybrid` + `full_blocks`/`target_heads` RAM filters), and optionally raw Q/K/V per head (`capture_qkv`) into an independent QKV store for `QKV Transfer`. QKV capture is fully decoupled from `target_blocks`/`target_heads`: `qkv_targets` (same `parse_block_head_pairs` format as `full_targets`/Head Freeze's `targets`, plus a whole-string `"all"`/`"all:all"` for every block/head) is parsed into a `{block_idx: {head_idx, ...}}` map (`qkv_target_map`), used directly in `setup()` to compute `qkv_blocks` and read by `qkv_inst.cfg`'s `target_block_map` for `_qkv_record`'s own per-block head filtering (mirroring `_attn_record`'s `full_target_map`). `register_layer(dm, node_id, attn_blocks | qkv_blocks, make_profiler_hook_factory(...), on_call=...)` registers a single layer covering the *union* of the attn and QKV block sets; `make_profiler_hook_factory` (`core/model_patch.py`) injects `_profiler_*`/`_qkv_*` keys independently per block (`is_profiler_block`/`is_qkv_block`, computed from membership in `attn_blocks`/`qkv_blocks`) so a QKV-only block doesn't trigger a wasted attention-map computation in `core/hooks.py`. `capture_steps`/`capture_sa`/`capture_ca`/`store_name`/`reset_store` stay shared by both capture paths. `capture_qkv` is not redundant with `store_mode=full_fp16`/`hybrid`: the stored attention map only replays the exact historical pattern (Head Freeze, QKV Transfer's `use_map`), raw Q/K/V lets QKV Transfer's `use_q`/`use_k`/`use_v` recombine components from two different generations into a new pattern — softmax is lossy, you can't derive one from the other. In `hybrid` mode, `full_targets` restricts full-map storage to specific heads for the blocks it lists, overriding `full_blocks` for just those blocks — `full_blocks` still applies normally to any block not covered by `full_targets`, so the two combine. A `full_targets`-covered block is stored as a sparse `{head_idx: tensor}` dict in `entry["map"]` instead of a dense `[H, Sq, Sk]` tensor. Single-head consumers (Head Freeze, QKV Transfer) index it the same way either way; multi-head consumers (Query Map, Key Map, Zone Analysis) require the dense form and will error/skip on a sparse one. Geometry (num_frames/latent_h/latent_w) is auto-detected — no manual dims. Outputs a single `handle` STRING (plain `STRING` type, not a custom one — must match every downstream node's plain-STRING `store_handle`/`qkv_handle` widgets or ComfyUI refuses the connection) shared by both the attn store and, if `capture_qkv` is on, the QKV store (`qkv_inst` is created with the same name — the two live in independent registry namespaces and can't collide). Most nodes depend on a prior setup run, read via that `handle` string (a separate later run — see hook architecture note below).

### Visualization
- **LTX Attn — Key Map** (`nodes/visualize.py:LTXAttentionKeyMap`): "What is being looked at?" — reduces query dim, reshapes keys to [F, H_lat, W_lat]. SA only.
- **LTX Attn — Query Map** (`nodes/visualize.py:LTXAttentionQueryMap`): "Who is actively looking?" — reduces key dim, works for SA+CA.
- **LTX Attn — Metrics Heatmap** (`nodes/visualize.py:LTXAttentionMetricsViz`): 2D heatmap [blocks × heads] for entropy/temporal/spatial/sink/frame_dist_mean/frame_dist_std/spatial_dist_mean/spatial_dist_std (+ frame_dist_*_norm/spatial_dist_*_norm, divided by max possible distance for cross-run comparability) metrics. Returns IMAGE + stats string.
- **LTX Attn — Grid Viz** (`nodes/visualize.py:LTXAttentionGridViz`): Full overview grid read from a `store_handle`. Views: key_map/query_map/diff. Supports frame modes: avg/all/sequence/specific frames. Normalize modes: global/per_cell/per_block/per_head.
- **LTX Attn — Timestep Evolution** (`nodes/evolution.py`): Line chart of metric vs denoising step per head.

### Intervention
- **LTX Attn — Head Freeze** (`nodes/transfer.py:LTXAttentionHeadFreeze`): Locks attention map(s) from a pivot step. Requires prior capture with `store_mode=full_fp16` (or `hybrid` for that block). `targets` STRING accepts multiple `(block, head)` pairs in one node instance (parsed by `utils/helpers.py:parse_block_head_pairs`, accepts both Head Candidates' `candidates_csv` and manual `block:head | block:head` entry, plus `block:all` for every head actually captured for that block) — `freeze_from_step`/`freeze_step_source`/`blend_weight` are shared across all targets, no per-head override yet. Optional `store_handle` to target a specific named store. Blank `targets` calls `unregister_layer` (this node's own layer only) and passes the model through unmodified — the reliable way to disable it, since `diffusion_model` is shared across every `model.clone()` and ComfyUI's node bypass/mute skips `apply_freeze()` (and its cleanup) entirely, leaving a stale patch from a previous run in effect. Its `make_freeze_hook_factory()` (`ops/freeze.py`) owns a fresh per-block step counter per node run, gating the freeze to `current_step >= freeze_from_step` by returning `None` (no-op passthrough) before that.
- **LTX Attn — QKV Transfer** (`nodes/transfer.py:LTXQKVTransfer`): Injects Q/K/V from source generation into target. `targets` STRING uses the same format as Head Freeze's `targets` (`utils/helpers.py:parse_block_head_pairs` — CSV/`block:head`/`block:h1,h2,...`/`block:all`), plus a whole-string `"all"` for every block/head captured in the QKV store; `_resolve_targets()` expands `all` against `source_step`'s actual captured heads. Modes: use_k+use_v (style), use_map (raw softmax), full QKV replace. `sim_filter` for content-preserving transfer via cosine similarity gating. Optional `handle` to target a specific named QKV store — the same `handle` string Setup Capture outputs. No selected `use_*` flag, or blank `targets`, cleanly unregisters this node's own layer and passes the model through (`_disable()`) instead of silently returning a possibly-still-patched model — same `diffusion_model`-is-shared rationale as Head Freeze's blank `targets`.
- **LTX Attn — QKV Multiplier** (`nodes/transfer.py:LTXQKVMultiplier`, `ops/qkv_multiply.py:apply_qkv_multiply`): Live per-head attention-sharpness/output-magnitude scaling, no prior capture needed. `targets` uses the same `parse_block_head_pairs` format as Head Freeze/QKV Transfer, plus a whole-string `"all"`/`"all:all"` for every block (0-47) and head (0-31) — resolved directly (no store to check against, unlike QKV Transfer's `"all"`). Only two knobs, not four: `qk_mult` scales Q alone before the attention dot product (softmax logit scale — changes attention sharpness, does **not** ablate: 0 just makes the head attend uniformly, it still contributes via V); a separate `k_mult` would be redundant since Q/K are each a uniform per-head scalar, so only `q_mult × k_mult` ever shows up in the logits. `vo_mult` scales V alone before the `attn_weights @ V` matmul (equivalently scaling the output after — same result either way since the matmul is linear in V, so `v_mult × o_mult` is the only thing that matters, hence one knob applied at the V side to avoid cloning the output tensor); 0 genuinely zeroes the head's contribution (true ablation). Both multipliers are shared across every target, same as Head Freeze's `blend_weight`. `apply_sa`/`apply_ca` gate which attention type(s) it fires on. `from_step`/`to_step` (default full range `0`-`999`) gate which denoising steps it's active for, per targeted block's own step counter (owned by `_make_multiply_hook_factory`, fresh per node run) — same pattern as QKV Transfer's `_make_transfer_hook_factory`. Same blank-`targets`-disables convention as Head Freeze/QKV Transfer (clone+`unregister_layer`+passthrough, reliable regardless of ComfyUI bypass/mute). Dispatches via `core/hooks.py`'s `_qkvmul_active`/`_qkvmul_cfg` transformer_options keys, priority 4 (after QKV Transfer, before Head Freeze).

### IO & Debug
- **LTX Attn — Store Dump/Load** (`nodes/io.py:LTXStoreDump`/`LTXStoreLoad`): Save/load AttentionStore and/or QKVStore in one combined `.pt` (`{"attn": {...}, "qkv": {...}}`, either section optional). Single `handle` STRING resolves both store types (the same name Setup Capture outputs covers both, since attn/QKV live in independent registry namespaces); on dump, a handle explicitly given but with no attn store under it raises (typo protection), a missing QKV store under the same name is silently skipped (capture_qkv may have been off), blank falls back to whichever store(s) are current, and dump only errors if *neither* resolves. `LTXStoreLoad` returns the resolved `handle` as an output for wiring into downstream nodes.
- **LTX Attn — Compare Runs** (`nodes/utils.py`): Diff heatmap between two capture runs.
- **LTX Attn — Head Candidates** (`nodes/utils.py:LTXAttentionHeadCandidates`): Combines several metrics' zscore diff (reuses `LTXAttentionCompareRuns`'s store-loading/extraction/zscore logic) into one composite `mean(|zscore|)` score per (block, head), outputs a ranked candidate shortlist + a control group (lowest-score or random) as plain-text reports and `block,head` CSV lists — for picking heads to feed into Head Freeze.
- **Store Inspect nodes** (`nodes/inspect.py`): Print store contents for debugging (AttentionStore summary includes key/query map presence).
- **LTX Attn — Reset Patches** (`nodes/reset.py:LTXResetPatches`): Calls `reset_all_layers()` — clears every registered layer on the model's `diffusion_model` and restores the pristine `_forward`. Manual escape hatch for a layer orphaned by deleting/rewiring an intervention/capture node out of the graph (nothing calls `unregister_layer` for a `node_id` that stops executing, so its last-registered layer would otherwise linger).
- **LTX Attn — Latent Dims** (`nodes/utils.py:LTXLatentDims`): Extract T/H/W from a LATENT tensor.

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
- **LTX Attn — RF-Inv Forward** / **Reverse** (`nodes/rf_inversion.py`): Random Fourier feature-based forward (x0→xT) and reverse (xT→x0) samplers for inversion workflows.

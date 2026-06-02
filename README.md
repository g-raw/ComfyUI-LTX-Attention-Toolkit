# LTX Attention Profiler

A ComfyUI custom node suite for profiling, visualizing and steering
attention heads in LTX-Video 2.3 (distill & dev).

Built for research on attention-based video generation control —
spatial/temporal head specialization, attention map transfer,
keypoint tracking and cross-modal (audio↔video) dynamics.

---

## Features

- **Profiling** — capture self-attention and cross-attention maps
  for any subset of blocks, heads and denoising steps
- **Metrics** — per-head entropy, temporal locality, spatial locality,
  sink mass — computed chunked on GPU, stored on CPU
- **Visualization** — key maps, query maps, metrics heatmaps,
  timestep evolution curves, full grid overview
- **Intervention** — head freeze (lock an attention map at a pivot step),
  Q/K/V transfer between two generations
- **IO** — dump/load stores to `.pt` for offline analysis and
  cross-run comparison (dev vs distill, prompt A vs prompt B)

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ltx-attention-profiler
```

No extra dependencies beyond what ComfyUI already provides
(`torch`, `numpy`).

---

## Project structure

```
nodes_ltx_attention_profiler/
├── __init__.py              ← ComfyUI entry point
│
├── core/
│   ├── stores.py            ← AttentionStore + QKVStore singletons
│   ├── hooks.py             ← Universal hook on optimized_attention
│   └── model_patch.py       ← _forward wrap/unwrap + block hooks
│
├── ops/
│   ├── freeze.py            ← Head freeze intervention
│   ├── qkv_transfer.py      ← Q/K/V substitution transfer
│   └── map_store.py         ← Reduced/full map storage callback
│
├── nodes/
│   ├── capture.py           ← CaptureSetup, QKVCapture
│   ├── transfer.py          ← HeadFreeze, QKVTransfer
│   ├── visualize.py         ← QueryMap, KeyMap, MetricsViz, GridViz
│   ├── evolution.py         ← TimestepEvolution
│   ├── io.py                ← Dump/Load (Attn + QKV)
│   ├── inspect.py           ← Store inspect nodes
│   ├── map_store_node.py    ← MapStore node
│   └── utils.py             ← LatentDims, CompareRuns
│
└── utils/
    ├── graphics.py          ← Colormaps, grid rendering, Bresenham
    └── helpers.py           ← Call counter, parse helpers, logging
```

---

## Nodes reference

### Capture

#### `LTX Attn — Setup Capture`
Patches an LTX-2.3 model to capture attention maps and metrics
during inference.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | LTX model to patch |
| `capture_sa` | BOOL | Capture self-attention |
| `capture_ca` | BOOL | Capture cross-attention (video→text) |
| `store_full_maps` | BOOL | Store full [H, Sq, Sk] tensors (heavy) |
| `map_downsample` | INT | Spatial downsample factor for full maps |
| `target_blocks` | STRING | `"all"` or `"0,8,16,24,32,40,47"` |
| `capture_steps` | STRING | `"all"` or `"0,1,2,3"` |
| `reset_store` | BOOL | Clear previous capture data |

Outputs a **patched MODEL** — plug between loader and KSampler.

---

#### `LTX QKV — Capture Source`
Captures raw Q, K, V tensors for use with `LTX QKV — Transfer`.

| Input | Type | Description |
|---|---|---|
| `target_blocks` | STRING | Blocks to capture |
| `target_heads` | STRING | `"all"` or `"8,12,16"` |
| `capture_steps` | STRING | Steps to capture |
| `capture_sa` | BOOL | Capture self-attention QKV |
| `capture_ca` | BOOL | Capture cross-attention QKV |

---

#### `LTX Attn — Map Store`
Stores reduced (key_map + query_map per head) or full fp16 maps
in a dedicated `ATTN_MAP_STORE` object passed to visualization nodes.

| Input | Type | Description |
|---|---|---|
| `store_mode` | ENUM | `reduced` / `full_fp16` / `hybrid` |
| `full_blocks` | STRING | Blocks in full mode when `hybrid` |
| `latent_frames` | INT | Number of latent frames |
| `latent_height` | INT | `input_height / 32` |
| `latent_width` | INT | `input_width / 32` |

**Memory estimates (1280×720, 16 frames, 32 heads, 4 steps) :**

| Mode | RAM |
|---|---|
| `reduced` (all 48 blocks) | ~332 MB |
| `full_fp16` (5 blocks) | ~16 GB |
| `hybrid` (5 full + 43 reduced) | ~16.3 GB |

---

### Visualization

#### `LTX Attn — Key Map`
*"Which tokens are being looked at?"*

Reduces the query dimension → reshapes keys into `[F, H_lat, W_lat]`.
SA only (keys are video tokens with spatial geometry).

| Input | Type | Description |
|---|---|---|
| `block_idx` | INT | Block to visualize |
| `step_idx` | INT | `-1` = last captured step |
| `head_indices` | STRING | `"all"` or `"8,12,16"` |
| `query_token_idx` | INT | `-1` = average over all queries |
| `aggregate_frames` | ENUM | `mean` / `max` / `first` |
| `cell_size` | INT | Pixel height of each head cell |

---

#### `LTX Attn — Query Map`
*"Which tokens are actively looking?"*

Reduces the key dimension → reshapes queries into `[F, H_lat, W_lat]`.
Works for both SA and CA.

For CA: shows which video regions are attending to text tokens.
Set `key_token_idx` to isolate a specific text token.

---

#### `LTX Attn — Metrics Heatmap`
2D heatmap: **X = blocks, Y = heads, color = metric value**.

| Metric | Meaning |
|---|---|
| `entropy` | High = diffuse attention (global head). Low = focused. |
| `temporal` | High = attends across frames (motion/coherence head). |
| `spatial` | High = attends within same frame (texture/structure head). |
| `sink` | High = attention mass on first/last token (sink head). |

`step_idx = -1` averages across all captured steps.

---

#### `LTX Attn — Grid Viz`
Full overview grid from a `ATTN_MAP_STORE`.
X = blocks, Y = heads, each cell = key_map or query_map.

`frame_mode` options:

| Value | Result |
|---|---|
| `avg` | Average over all frames → 1 grid |
| `all` | Frames stacked vertically in each cell → 1 grid |
| `sequence` | One grid per frame → IMAGE batch |
| `0` or `3,7` | Specific frame index(es) |

`normalize` options: `global` / `per_cell` / `per_block` / `per_head`

---

#### `LTX Attn — Timestep Evolution`
Line chart: metric value vs denoising step for selected heads.
One colored curve per head.

Useful to identify:
- **Flat curves** → structurally fixed role
- **Monotone decreasing** → specializes progressively
- **Crossing curves** → heads swap roles mid-denoising
- **Late rise** → semantic tracking activated once signal emerges

---

### Intervention

#### `LTX Attn — Head Freeze`
Locks the attention map of a specific head starting from a pivot step.

Requires a prior capture run with `store_full_maps=True`.

| Input | Type | Description |
|---|---|---|
| `block_idx` | INT | Target transformer block |
| `head_idx` | INT | Target head |
| `freeze_from_step` | INT | Step at which freeze activates |
| `freeze_step_source` | INT | Which captured step's map to use |
| `blend_weight` | FLOAT | 1.0 = pure frozen, 0.5 = 50/50 blend |

**Effect on head 8, block 24:**
Prevents the temporal window from shrinking during denoising →
the model maintains long-range temporal coherence.

---

#### `LTX QKV — Transfer`
Injects Q/K/V from a source generation into a target generation.

Supports multi-block, multi-head targeting:
```
# Simple syntax (same heads for all blocks)
target_blocks = "24,32,40"
head_indices  = "8,12,16"

# Extended syntax (per-block head lists)
target_blocks = "24:8,12 | 32:all | 40:0,4,8"
```

Transfer modes (combinable):

| Flag | Effect |
|---|---|
| `use_k + use_v` | Classic style transfer (mode D) |
| `use_k` only | Key-only steering |
| `use_map` | Inject raw softmax map, bypass Q/K/V |
| `use_q + use_k + use_v` | Full QKV replacement |

`sim_filter`: only transfer tokens where Q_target ≈ Q_source
(cosine similarity threshold) — useful for content-preserving transfer.

---

### IO & Debug

| Node | Description |
|---|---|
| `LTX Attn — Store Dump` | Save AttentionStore to `.pt` |
| `LTX Attn — Store Load` | Load `.pt` into AttentionStore |
| `LTX QKV — Dump` | Save QKVStore to `.pt` |
| `LTX QKV — Load` | Load `.pt` into QKVStore |
| `LTX Attn — Compare Runs` | Diff heatmap between two `.pt` files |
| `LTX Attn — Store Inspect` | Print AttentionStore contents |
| `LTX QKV — Store Inspect` | Print QKVStore contents |
| `LTX Map Store — Inspect` | Print ATTN_MAP_STORE contents |
| `LTX — Latent Dims` | Extract T/H/W from a LATENT |

---

## Typical workflows

### Workflow 1 — Profiling run

```
[Load LTX Model]
      │
[LTX Attn — Setup Capture]
  capture_sa=True
  store_full_maps=False
  target_blocks="all"
      │
[KSampler]
      │
      ├── [LTX Attn — Metrics Heatmap]  metric=entropy
      ├── [LTX Attn — Metrics Heatmap]  metric=temporal
      └── [LTX Attn — Timestep Evolution]  block_idx=24
```

### Workflow 2 — Head freeze experiment

```
# Step 1: capture reference maps
[Load LTX] → [Setup Capture, store_full_maps=True, target_blocks="24"]
           → [KSampler] → [Store Dump → "ref.pt"]

# Step 2: apply freeze
[Load LTX] → [Store Load ← "ref.pt"]
           → [Head Freeze, block=24, head=8, from_step=3]
           → [KSampler] → [Save Video]
```

### Workflow 3 — QKV transfer between prompts

```
# Step 1: capture source
[Load LTX] → [QKV Capture, blocks="24,32", heads="8,12,16"]
           → [KSampler, prompt="chrome robot on rails"]
           → [QKV Dump → "source.pt"]

# Step 2: transfer to target
[Load LTX] → [QKV Load ← "source.pt"]
           → [QKV Transfer, use_k=True, use_v=True, blend=0.7]
           → [KSampler, prompt="golden robot on rails"]
           → [Save Video]
```

---

## Architecture notes

### Token layout
LTX-2.3 uses `SymmetricPatchifier(patch_size=1)`:
**1 token = 1 latent pixel = ~32×32 pixels in image space**.

For a 1280×720 video with 16 latent frames:
```
Sequence length = 16 × (720/32) × (1280/32) = 16 × 22 × 40 = 14080 tokens
                               ↑ or 11×20 depending on workflow upscale step
```

### Attention map interpretation

```
W : [H=32, Sq, Sk]   (self-attention)
         ↑  ↑   ↑
         heads  sequence length

Key map   = W.mean(dim=1) → [Sk]  "what is being looked at"
Query map = W.mean(dim=2) → [Sq]  "who is actively looking"
```

### Hook architecture
A single universal hook is installed on both
`optimized_attention` and `optimized_attention_masked`.
Priority order per call:
1. Profiling → AttentionStore
2. MapStore → ATTN_MAP_STORE callback
3. QKV Capture → QKVStore
4. QKV Transfer → Q/K/V substitution
5. Head Freeze → map injection
6. Normal pass-through

---

## Limitations & known issues

- LTX-2.3 only (48 transformer blocks, 32 heads, `patch_size=1`)
- SA freeze currently supports single head per node instance
  (chain multiple HeadFreeze nodes for multi-head intervention)
- Full map storage at native resolution (3520×3520 per head)
  requires ~25 MB/head — use `map_downsample` or `hybrid` mode
- Audio stream is not profiled (video stream only)

---

## References

- [LTX-Video 2.3](https://huggingface.co/Lightricks/LTX-Video)
- [Sparse VideoGen (arXiv:2504.10317)](https://arxiv.org/abs/2504.10317)
  — attention head classification methodology
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

---

## License

GPL 3.0


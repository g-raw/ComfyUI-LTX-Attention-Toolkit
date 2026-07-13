# LTX Attention Profiler

A ComfyUI custom node suite for profiling, visualizing and steering
attention heads in LTX-Video 2.3 (distill & dev).

Built for research on attention-based video generation control —
spatial/temporal head specialization, attention map transfer,
keypoint tracking and cross-modal (audio↔video) dynamics.

**⚠️ Work in progress.** This toolkit is under active development. Node
inputs/outputs, store formats, and naming can change at any time —
including breaking changes without backward compatibility. Existing
workflows and saved `.pt` files may stop working after an update.

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
git clone https://github.com/g-raw/ComfyUI-LTX-Attention-Toolkit.git
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
│   ├── stores.py            ← StoreRegistry (named, non-singleton) + AttentionStore/QKVStore proxies
│   ├── hooks.py             ← Universal hook on optimized_attention
│   └── model_patch.py       ← _forward wrap/unwrap + block hooks
│
├── ops/
│   ├── freeze.py            ← Head freeze intervention
│   └── qkv_transfer.py      ← Q/K/V substitution transfer
│
├── nodes/
│   ├── capture.py           ← CaptureSetup (metrics + key/query/full maps + optional raw QKV)
│   ├── transfer.py          ← HeadFreeze, QKVTransfer
│   ├── visualize.py         ← QueryMap, KeyMap, MetricsViz, GridViz
│   ├── evolution.py         ← TimestepEvolution
│   ├── io.py                ← StoreDump/StoreLoad (Attn + QKV, one file)
│   ├── inspect.py           ← Store inspect nodes
│   └── utils.py             ← LatentDims, CompareRuns, HeadCandidates
│
└── utils/
    ├── graphics.py          ← Colormaps, grid rendering, Bresenham
    └── helpers.py           ← Call counter, parse helpers, logging
```

---

## Nodes reference

### Capture

#### `LTX Attn — Setup Capture`
Patches an LTX-2.3 model to capture attention metrics, reduced
key/query maps, optionally full attention maps, and optionally raw
Q/K/V — one capture path, one node, real metrics in every mode.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | LTX model to patch |
| `capture_sa` | BOOL | Capture self-attention (applies to both the metrics and the QKV capture below) |
| `capture_ca` | BOOL | Capture cross-attention (video→text), same scope |
| `target_blocks` | STRING | `"all"` or `"0,8,16,24,32,40,47"` — drives the metrics/key/query/full maps only |
| `target_heads` | STRING | `"all"` or `"8,12,16"` — RAM filter for the metrics/key/query maps |
| `capture_steps` | STRING | `"all"` or `"0,1,2,3"` — shared by metrics and QKV capture |
| `store_mode` | ENUM | `reduced` / `full_fp16` / `hybrid` |
| `full_blocks` | STRING | Blocks stored at full res (every head) when `hybrid`. A block also listed in `full_targets` uses that block's per-head selection instead; `full_blocks` still applies normally to any block `full_targets` doesn't cover — the two combine |
| `full_targets` | STRING | Optional, `hybrid` only — for the blocks listed here, restrict full-map storage to specific `(block, head)` pairs instead of every head. Same format as `qkv_targets`/`Head Freeze`'s `targets` below |
| `map_downsample` | INT | Spatial downsample factor for full maps |
| *(QKV capture — separate from the attention-map settings above)* | | |
| `capture_qkv` | BOOL | Also capture raw Q/K/V per head into a separate QKV store, for `QKV Transfer` |
| `qkv_targets` | STRING | Only used when `capture_qkv` is on. **Independent of `target_blocks`/`target_heads` above** — its own `(block, head)` list, since raw Q/K/V is far more expensive than the metrics/key/query maps. Same format as `full_targets`/`Head Freeze`'s `targets`: paste `Head Candidates`' `candidates_csv`, or type `block:head \| block:h1,h2,... \| block:all \| ...`. A whole-string `"all"` (or `"all:all"`) captures every block and head |
| `store_name` | STRING | Empty = new auto-named handle every run. Given a name, re-running reuses that same handle (get-or-create) instead of spawning `name_2`, `name_3`, … Applies to both the attn and the QKV store — they can't collide, they live in separate registries |
| `reset_store` | BOOL | With a named `store_name`: clear that handle (both stores) before capturing. With it blank, the handle is always fresh already, so this has no effect |

`reduced` always includes the real `entropy`/`temporal`/`spatial`/`sink`/
`frame_dist_mean`/`frame_dist_std`/`spatial_dist_mean`/`spatial_dist_std`
metrics (plus `_norm` variants of the last four, see below) and
`key_map`/`query_map` (geometry auto-detected from the live latent — no
manual frame/height/width inputs needed). `full_fp16`/`hybrid`
additionally store the full `[H, Sq, Sk]` map for the relevant blocks —
except for any block listed in `full_targets`, where only the listed
heads' maps are stored for *that block* (a sparse `{head_idx: [Sq,
Sk]}` dict instead of the full `[H, Sq, Sk]` tensor), to avoid paying
for every head when you already know exactly which ones you'll feed
into `Head Freeze`. `full_blocks` still applies normally to every other
block. `Head Freeze` and `QKV Transfer` read a single head either way
and don't care which form it is; `Query Map`/`Key Map`/`Zone Analysis`
need every head and will raise (or silently skip that block/step) on a
sparse map — put that block in `full_blocks` instead if you need those.

**`capture_qkv` is not redundant with `full_fp16`/`hybrid`** — see
"Attention map vs. raw Q/K/V" in Architecture notes below for why.

**Memory estimates (1280×720, 16 frames, 32 heads, 4 steps) :**

| Mode | RAM |
|---|---|
| `reduced` (all 48 blocks) | ~332 MB |
| `full_fp16` (5 blocks) | ~16 GB |
| `hybrid` (5 full + 43 reduced) | ~16.3 GB |

Outputs a **patched MODEL** and a single **`handle`** STRING — the same
name is used for both the attn store and (if `capture_qkv` is on) the
QKV store, since they live in independent registry namespaces and can't
collide. Plug the model between loader and KSampler, and type `handle`
into any visualization/intervention node's `store_handle`/`qkv_handle`
widget in a later run (see
"Hook architecture" below for why this is a separate-run handle rather
than a wired socket).

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
2D heatmap: **X = blocks, Y = heads, color = metric value**. Always
normalized per-image (min→max of whatever the metric's actual range is
in this store — most of these metrics aren't natively in `[0, 1]`, so
without normalizing the whole heatmap just clips to a single color) —
a numeric colorbar is stamped along the bottom showing that literal
`[min, max]` range, so the direction is never divorced from the
magnitude.

| Metric | Meaning |
|---|---|
| `entropy` | High = diffuse attention (global head). Low = focused. |
| `temporal` | High = attends across frames (motion/coherence head). |
| `spatial` | High = attends within same frame (texture/structure head). |
| `sink` | High = attention mass on first/last token (sink head). |
| `frame_dist_mean` | Attention-mass-weighted average `\|frame_k - frame_q\|`, in frames (SA only). High = looks at temporally distant frames. |
| `frame_dist_std` | Spread of that frame-distance distribution, in frames (SA only). High = mixes near and far frames; low = attends at a consistent temporal offset. |
| `frame_dist_mean_norm` / `frame_dist_std_norm` | Same, divided by `num_frames - 1` so they stay in `[0, 1]` and comparable across runs with a different number of frames. |
| `spatial_dist_mean` | Attention-mass-weighted average Euclidean distance, in patch units (same-frame pairs only), between query and key patch positions (SA only). High = looks at spatially distant tokens within the frame. |
| `spatial_dist_std` | Spread of that spatial-distance distribution, in patch units (SA only). High = mixes near and far patches; low = attends at a consistent spatial offset. |
| `spatial_dist_mean_norm` / `spatial_dist_std_norm` | Same, divided by the patch-grid diagonal `sqrt(latent_h² + latent_w²)` so they stay in `[0, 1]` and comparable across runs at a different spatial resolution. |

`step_idx = -1` averages across all captured steps. All eight
`frame_dist_*`/`spatial_dist_*` fields are `0` for cross-attention
(frames/patch positions don't apply to text tokens) and whenever the SA
map doesn't match the `num_frames × patches_per_frame` geometry. Prefer
the `_norm` variants whenever comparing across runs that don't share the
exact same `num_frames`/`latent_height`/`latent_width` — the raw values
alone aren't apples-to-apples in that case.

---

#### `LTX Attn — Grid Viz`
Full overview grid read from a capture `STORE_HANDLE`.
X = blocks, Y = heads, each cell = key_map, query_map, or their diff.
Every 8th row/column gets a separator line, same as `Metrics Heatmap`/
`Compare Runs`, to make it easier to count blocks/heads at a glance.

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
Locks the attention map of one or more heads starting from a pivot step
— a single node instance can target several `(block, head)` pairs at
once (no need to chain one node per head).

Requires a prior capture run with `store_mode=full_fp16` (or `hybrid` for
that block).

| Input | Type | Description |
|---|---|---|
| `targets` | STRING | One or more `(block, head)` pairs. Paste `Head Candidates`' `candidates_csv` directly (one `block,head` per line), or type manually as `block:head \| block:head \| ...` — use `block:all` to freeze every captured head of a block in one entry. Blank = disable (see below) |
| `freeze_from_step` | INT | Step at which freeze activates |
| `freeze_step_source` | INT | Which captured step's map to use |
| `blend_weight` | FLOAT | 1.0 = pure frozen, 0.5 = 50/50 blend — shared across every target, no per-head override yet |
| `store_handle` | STRING | Optional — target a specific named store. Blank = whichever store is currently active |

`freeze_from_step`/`freeze_step_source`/`blend_weight` apply to every
target the same way. If you need different values per head, chain
multiple `Head Freeze` nodes instead — `targets` only saves the chaining
when the shared settings are fine.

**To disable, clear `targets` — don't use ComfyUI's node bypass/mute.**
This node patches the diffusion model's `_forward` directly, and that
underlying model object is shared across every `model.clone()` in the
session (not recreated per run). A normal execution unwraps that patch
before reapplying it, but ComfyUI's bypass/mute skips the node's
function entirely, so a stale patch from an earlier run is never
cleaned up and stays in effect even though the node "looks" disabled.
An empty `targets` field still runs the node's code, so it reliably
unwraps and passes the model through untouched.

---

#### `LTX Attn — QKV Transfer`
Injects Q/K/V from a source generation into a target generation.

`targets` uses the same format as `Head Freeze`'s `targets` (both parsed
by `parse_block_head_pairs`), plus a whole-string `all`:
```
# Head Candidates CSV (one 'block,head' pair per line)
24,8
24,12
32,16

# Manual — one block, several heads
targets = "24:8,12,16"

# Manual — several blocks, mixed
targets = "24:8,12 | 32:all | 40:0,4,8"

# Every block/head captured in the QKV store
targets = "all"
```
`block:all` (or a whole-string `all`) resolves against whatever heads
were actually captured for that block/`source_step` in the QKV store.

Transfer modes (combinable):

| Flag | Effect |
|---|---|
| `use_k + use_v` | Classic style transfer (mode D) |
| `use_k` only | Key-only steering |
| `use_map` | Inject raw softmax map, bypass Q/K/V |
| `use_q + use_k + use_v` | Full QKV replacement |

`sim_filter`: only transfer tokens where Q_target ≈ Q_source
(cosine similarity threshold) — useful for content-preserving transfer.

`handle` (STRING, optional): target a specific named QKV store — the
same `handle` string output by Setup Capture. Blank = whichever QKV
store is currently active.

**To disable, don't select any transfer flag (or blank `targets`) —
don't use ComfyUI's node bypass/mute.** Same reasoning as `Head
Freeze`: this node patches the shared `diffusion_model` directly, and
bypass/mute skips this node's cleanup entirely, so a stale patch from
an earlier run stays in effect. Turning off every `use_*` flag, or
blanking `targets`, still runs the node's code and reliably unwraps
instead.

---

#### `LTX Attn — QKV Multiplier`
Scales attention sharpness and/or output magnitude per targeted head —
a live multiply, no prior capture needed. Amplify a head (`> 1.0`) or
try to kill it (`0.0`), or several at once.

| Input | Type | Description |
|---|---|---|
| `targets` | STRING | Same format as `Head Freeze`/`QKV Transfer`: paste `Head Candidates`' `candidates_csv` directly, or `block:head \| block:h1,h2,... \| block:all \| ...`. A whole-string `all` (or `all:all`) targets every block (0-47) and head (0-31). Blank = disable (see below) |
| `apply_sa` / `apply_ca` | BOOLEAN | Which attention type(s) to affect |
| `qk_mult` / `vo_mult` | FLOAT | Per-head multipliers, shared across every target (`-10.0`–`10.0`, default `1.0` = no-op) |
| `from_step` / `to_step` | INT | Denoising step range (per targeted block) the multiply is active for. Defaults to the full range (`0`–`999`) |

**`qk_mult` vs. `vo_mult` — these are not interchangeable:** `qk_mult`
rescales the softmax logits before the attention weights are computed —
it changes how *sharp or flat* the head's attention distribution is.
Setting it to `0` doesn't ablate the head, it just makes it attend
uniformly over every key (still contributes via V, just unfocused).
`vo_mult` scales the head's actual contribution to the output — `0`
genuinely zeroes the head out. Use `vo_mult` to kill/amplify a head's
influence; use `qk_mult` to experiment with attention sharpness instead.

(There's no separate `q_mult`/`k_mult` or `v_mult`/`o_mult` pair:
scaling Q and K independently only ever matters through their product
on the logits — Q and K are each a uniform per-head scalar, so
`q_mult × k_mult` is all that shows up in the math, hence one `qk_mult`
knob. Same reasoning collapses V/O into `vo_mult`: V is scaled before
the (linear) `attn_weights @ V` matmul, O after, so only the product
affects the result.)

**To disable, clear `targets` — don't use ComfyUI's node bypass/mute.**
Same reasoning as `Head Freeze`/`QKV Transfer`: this node patches the
shared `diffusion_model` directly, and bypass/mute skips this node's
cleanup entirely, leaving a stale patch from an earlier run in effect.
Blanking `targets` still runs the node's code and reliably unwraps
instead.

---

### IO & Debug

| Node | Description |
|---|---|
| `LTX Attn — Store Dump` | Save the AttentionStore and/or QKVStore to one `.pt` |
| `LTX Attn — Store Load` | Load AttentionStore and/or QKVStore sections from a `.pt` |
| `LTX Attn — Compare Runs` | Diff heatmap + ranked (block, head) table for one metric between two runs |
| `LTX Attn — Head Candidates` | Combine several metrics' zscore diff into one composite score, shortlist candidate + control (block, head) groups |
| `LTX Attn — Store Inspect` | Print AttentionStore contents (incl. key/query map presence) |
| `LTX Attn — QKV Store Inspect` | Print QKVStore contents |
| `LTX Attn — Latent Dims` | Extract T/H/W from a LATENT |
| `LTX Attn — Reset Patches` | Clear every registered intervention/capture layer on a model and restore the pristine forward — see "Chaining multiple intervention/capture nodes" below |

`Store Dump`/`Store Load` take a single `handle` STRING input — the
same name is used to resolve both the attn store and the QKV store,
mirroring Setup Capture's unified `handle` output. On dump, an empty
handle falls back to whichever stores are currently active; an explicit
handle raises clearly if no attn store exists under that name (a typo
isn't silently swallowed), but a missing QKV store under that name is
silently skipped (capture_qkv may simply have been off for that run) —
the dump still succeeds as long as at least one of the two resolves. On
load, each section present in the file goes into that name (`"default"`
if blank, get-or-create); `Store Load` returns the resolved `handle` as
an output so you can wire it straight into downstream nodes instead of
retyping it. This matters once multiple stores coexist — parallel
branches, multiple captures in one session.

#### `LTX Attn — Compare Runs` details

Compares one metric — including the `_norm` variants of the distance
metrics, which is what you want here if the two runs don't share the
exact same `num_frames`/resolution — between two captures, block-by-block
and head-by-head, for self- or cross-attention. Reads both stores live
from the registry by handle — to compare a dumped `.pt`, load it into a
handle first with `Store Load`.

| Input | Type | Description |
|---|---|---|
| `store_handle_a` / `store_handle_b` | STRING | The two stores to compare |
| `attn_type` | ENUM | `sa` / `ca` |
| `metric` | ENUM | `entropy` / `temporal` / `spatial` / `sink` / `frame_dist_mean` / `frame_dist_std` / `frame_dist_mean_norm` / `frame_dist_std_norm` / `spatial_dist_mean` / `spatial_dist_std` / `spatial_dist_mean_norm` / `spatial_dist_std_norm` |
| `step_idx` | INT | `-1` averages across all captured steps |
| `top_k` | INT | How many `(block, head)` pairs to list, ranked by `\|diff_mode score\|` |
| `norm_percentile` | FLOAT | Clip the heatmap color scale at this percentile of the diff_mode score (default 0.98) so a few outlier cells don't wash the rest out to white — `1.0` uses the true max |
| `colormap` | ENUM | `diverging` (default, 0 = black) / `coolwarm` (0 = near-white) / `viridis` / `inferno` |
| `diff_mode` | ENUM | `absolute` / `relative_pct` / `zscore` — see below |

**Sign convention:** raw diff `= A - B` — positive (red) means A's value
is higher, negative (blue) means B's is higher. `stats_text` prints
which handle is A and which is B so this is never ambiguous from the
image alone. The output IMAGE has a numeric colorbar stamped along the
bottom (`-clip_val` / `0` / `+clip_val`, in `diff_mode`'s units) so the
actual magnitude is readable, not just the direction.

**Why `diff_mode` matters:** the four metrics don't share a scale —
`sink` is a bounded probability-like quantity while `temporal`/`spatial`
are unnormalized raw scores that can range much wider. A raw `A - B` of
similar magnitude can mean "huge proportional change" on one metric and
"noise" on another, so don't compare raw diffs across metrics directly.

| Mode | Formula | Use for |
|---|---|---|
| `absolute` (default) | `A - B`, in the metric's own units | Looking at one metric in isolation |
| `relative_pct` | `(A - B) / max(\|A\|, \|B\|) * 100` | "% change" — comparable in spirit across metrics |
| `zscore` | `(A - B) / std(A and B combined)` | Diff in units of that metric's own spread — the most apples-to-apples way to ask whether one metric moved proportionally more than another |

`stats_text` also reports `min`/`max` (alongside `mean`/`std`) for A and
B separately *before* any diffing, so you can see each metric's
intrinsic value range up front.

Blocks are aligned by their actual index (not column position), so the
two runs don't need identical `target_blocks`. Outputs a diff heatmap
IMAGE plus a `stats_text` STRING with summary stats and the full
top-`top_k` ranked table — run it once per metric, then compare which
`(block, head)` pairs recur across metrics to spot structurally divergent
heads vs. metric-specific noise.

#### `LTX Attn — Head Candidates` details

Automates that last step: instead of running Compare Runs once per
metric and manually cross-referencing which `(block, head)` pairs recur,
this combines several metrics' zscore diffs into one **composite score**
per head — `mean(|zscore(A - B)|)` across the metrics you list — and
outputs a ranked shortlist plus a control group, as plain text (not a
heatmap) so you can copy `block,head` pairs straight into `Head Freeze`.

| Input | Type | Description |
|---|---|---|
| `store_handle_a` / `store_handle_b` | STRING | The two stores to compare |
| `attn_type` | ENUM | `sa` / `ca` |
| `metrics` | STRING | Comma-separated metric names, e.g. `temporal,frame_dist_mean_norm,frame_dist_std_norm` |
| `step_idx` | INT | `-1` averages across all captured steps |
| `top_k` | INT | Candidate shortlist size |
| `control_mode` | ENUM | `lowest_score` (heads least implicated by these metrics) / `random` |
| `control_k` | INT | Control group size (`0` to skip it) |
| `seed` | INT | Only used when `control_mode = random` |

Use `_norm` distance metrics here rather than the raw ones if the two
runs don't share the same `num_frames`/resolution — same rationale as
Compare Runs. The `report` STRING lists both groups ranked by composite
score, each entry showing every individual metric's zscore (not just the
composite) so you can tell whether a head is consistently implicated
across all of them or only carried by one. `candidates_csv` and
`control_csv` are just `block,head` pairs, one per line, for pasting
into other nodes/scripts. The control group excludes anything already in
the candidate shortlist.

---

## Typical workflows

### Workflow 1 — Profiling run

```
[Load LTX Model]
      │
[LTX Attn — Setup Capture]
  capture_sa=True
  store_mode=reduced
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
[Load LTX] → [Setup Capture, store_mode=full_fp16, target_blocks="24"]
           → [KSampler] → [Store Dump → "ref.pt"]

# Step 2: apply freeze
[Load LTX] → [Store Load ← "ref.pt"]
           → [Head Freeze, targets="24:8", from_step=3]
           → [KSampler] → [Save Video]
```

### Workflow 3 — QKV transfer between prompts

```
# Step 1: capture source
[Load LTX] → [Setup Capture, capture_qkv=True, qkv_targets="24:8,12,16 | 32:8,12,16"]
           → [KSampler, prompt="chrome robot on rails"]
           → [Store Dump, handle=<from Setup Capture> → "source.pt"]

# Step 2: transfer to target
[Load LTX] → [Store Load ← "source.pt"]  (wire its handle output)
           → [QKV Transfer, targets="24:8,12,16", use_k=True, use_v=True, blend=0.7]
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

### Attention map vs. raw Q/K/V

`Setup Capture`'s `store_mode=full_fp16`/`hybrid` (the attention map
`W = softmax(QK^T)`) and its `capture_qkv` toggle (raw Q/K/V, before the
softmax) capture genuinely different things — one doesn't substitute for
the other:

- **Attention map** — lets you *replay* the exact historical pattern
  later: `Head Freeze` and `QKV Transfer`'s `use_map` mode both just
  reuse the stored `W` against the *current* run's V
  (`ops/freeze.py`/`ops/qkv_transfer.py:98`).
- **Raw Q/K/V** — lets `QKV Transfer`'s `use_q`/`use_k`/`use_v` modes
  *recombine* components from two different generations and recompute a
  brand new `softmax(Q_eff·K_eff^T)·V_eff` (`ops/qkv_transfer.py:126-130`)
  — something no stored map alone can do, since softmax is lossy: you
  can't reconstruct Q/K from a saved `W`.

### Hook architecture
A single universal hook is installed on both
`optimized_attention` and `optimized_attention_masked`.
Priority order per call:
1. Profiling → AttentionStore (metrics + key/query/full maps)
2. QKV Capture → QKVStore
3. QKV Transfer → Q/K/V substitution
4. QKV Multiplier → per-head Q/K/V/O scaling
5. Head Freeze → map injection
6. Normal pass-through

### Chaining multiple intervention/capture nodes

Every intervention/capture node (`Setup Capture`, `Head Freeze`, `QKV
Transfer`, `QKV Multiplier`) registers its own contribution into a
shared, named layer registry on the underlying `diffusion_model` (keyed
by ComfyUI's stable per-node `unique_id`) instead of hand-rolling its
own `_forward` wrap. This means you *can* chain several of these nodes
in the same graph — e.g. two `QKV Multiplier` nodes targeting different
blocks, or `Head Freeze` + `QKV Multiplier` together — and each one's
effect stacks rather than silently overwriting the others. Re-running
the same node in place (a normal requeue) replaces only its own entry,
so nothing accumulates across repeated runs.

**Caveat:** if you delete or rewire a node out of the graph entirely,
its last-registered layer has no way to know it should remove itself —
nothing calls its cleanup on a node_id that stops executing. Run `LTX
Attn — Reset Patches` once (any point downstream of the model) to clear
every registered layer and fully restore the pristine model if you've
been iterating on your graph and suspect a stale layer is still applying.

### Why visualization/intervention nodes use a typed `store_handle` string

Captured data is written into the registry as a side effect of the
KSampler run, *after* the Setup node itself has already returned. Nodes
that read it back (`Query Map`, `Key Map`, `Metrics Heatmap`, `Grid Viz`,
`Head Freeze`, `Compare Runs`-adjacent IO nodes, …) take the handle as a
plain `STRING` widget rather than a wired socket on purpose: ComfyUI
schedules nodes by wire dependency, so a typed socket straight off the
Setup node's output would let these nodes run *before* the KSampler ever
populates the store, always producing empty results. Typing the handle
into a `STRING` widget instead means these are a separate, later queue
run against the already-populated registry instance — leave it blank to
fall back to whichever store is currently active.

---

## Limitations & known issues

- LTX-2.3 only (48 transformer blocks, 32 heads, `patch_size=1`)
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


# The model: what this tool computes, and why each number is what it is

Research and derivation date: **August 2026**. This document is the specification the
numeric core implements. Every formula carries its primary source and a tag:

- **[verbatim]** — transcribed from the primary source
- **[derived]** — arithmetic performed here from a verbatim source
- **[prior]** — a fitted constant, *not* a vendor specification
- **[unverified]** — could not be traced to a primary source; flagged, not shipped as fact

## Posture

A browser calculator cannot profile a machine. This is a first-order analytical model whose
job is to make *regimes and crossovers* legible — where a workload stops being memory-bound,
why more batch does not help, why more tensor parallelism eventually hurts. It is not a
benchmark and does not imply benchmark accuracy.

The governing rule, inherited from `FitEstimator.kt` in the OpenWeights Android app: **an
invented number is worse than none.** Where a constant cannot be sourced, it is labelled a
prior and exposed for the user to replace. Where a configuration cannot run at all, the tool
reports *infeasible* rather than a fast time.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| `b` | batch size (concurrent sequences) |
| `q` | query positions processed this step (`= s` in prefill, `= 1` in decode) |
| `s` | sequence length (context, prompt + generated) |
| `L` | number of transformer layers |
| `h` | hidden / residual-stream width |
| `f` | FFN intermediate width |
| `n_q`, `n_kv` | query heads, key/value heads (`n_q = n_kv` ⟹ MHA, `n_kv = 1` ⟹ MQA) |
| `d_h` | head dimension |
| `V` | vocabulary size |
| `N` | parameter count (`N_active` for MoE) |
| `w` | bytes per element of the tensor in question |
| `t`, `p`, `d`, `e` | tensor / pipeline / data / expert parallel degree |
| `C` | peak compute rate, FLOP/s (dense) |
| `B_mem`, `B_net` | peak memory, network bandwidth (B/s) |
| `α` | collective latency floor |

Two notation collisions in the literature, called out because they cause real errors:
Megatron uses `v` for both vocabulary size and interleaved-pipeline chunk count; Korthikanti's
`a` is PaLM's `H` is `n_q`.

---

## 2. The machine

### 2.1 The sparsity trap

NVIDIA's public H100 table marks TF32/BF16/FP16/FP8/INT8 rows with `*`, footnoted *"With
sparsity"*, and publishes **only** the sparse number. The Hopper whitepaper Table 3 lists
`dense/sparse` pairs with footnote 2: *"Effective TFLOPS / TOPS using the Sparsity feature."*

**Dense = published ÷ 2.** H100 SXM5 BF16 is **989.5 TFLOP/s dense**, 1,979 sparse. 2:4
structured sparsity requires explicitly pruned weights and is not used in production LLM work,
so every ceiling here is dense. FP64/FP32 have no sparse variant and are already dense.

The same trap at system scale: the DGX H100 headline **"32 PFLOPS" is sparse FP8**. Dense FP8
is 15.8 PF; **dense BF16 is 7.9 PF**. A tool quoting 32 PF is off by 2× or 4×.

### 2.2 H100 constants [verbatim, NVIDIA H100 product page + Hopper whitepaper v1.02]

| | H100 SXM5 | H100 PCIe | H100 NVL |
|---|---|---|---|
| BF16/FP16 tensor, dense | 989.5 TF/s | 756.5 TF/s | 835.5 TF/s |
| FP8 tensor, dense | 1,979 TF/s | 1,513 TF/s | 1,670.5 TF/s |
| TF32 tensor, dense | 494.5 TF/s | 378 TF/s | 417.5 TF/s |
| FP32 vector | 67 TF/s | 51 TF/s | 60 TF/s |
| FP64 / FP64 tensor | 34 / 67 TF/s | 26 / 51 TF/s | 30 / 60 TF/s |
| Memory | 80 GB HBM3 | 80 GB HBM2e | 94 GB HBM3 |
| Bandwidth | 3.35 TB/s | 2.0 TB/s | 3.9 TB/s |
| SMs | 132 | 114 | 132 |
| L2 | 50 MB | 50 MB | 50 MB |
| NVLink 4, per direction | 450 GB/s | **300 GB/s** | 300 GB/s |
| TDP | 700 W | 350 W | 350–400 W |

**NVLink direction, settled** [verbatim, Hopper whitepaper]: *"fourth-generation NVLink uses
only two high-speed differential pairs in each direction to form a single link, also delivering
25 GB/sec effective bandwidth in each direction. H100 includes 18 fourth-generation NVLink links
to provide 900 GB/sec total bandwidth."* So 18 × 25 = **450 GB/s each way**; the 900 GB/s figure
is the bidirectional sum. Using 900 where 450 belongs is the classic 2× error.

Only the SXM5 part carries the full 18-link 900 GB/s NVLink. The **PCIe and NVL parts use a
bridge rated 600 GB/s bidirectional = 300 GB/s each way**; giving them the SXM figure overstates
their intra-node bandwidth by 1.5×.

PCIe Gen5 ×16 is **64 GB/s each direction**, 128 GB/s bidirectional [verbatim].

**8×H100 (DGX/HGX)**: 4 × **third-generation NVSwitch**, 7.2 TB/s aggregate **bidirectional**
(= 8 × 900 GB/s),
bisection **3.6 TB/s bidirectional = 1.8 TB/s each direction** (a 4-vs-4 cut of four GPUs at
450 GB/s per direction), 8 × ConnectX-7 @ 400 Gb/s = 3.2 Tb/s = 400 GB/s each way node network.

Every bandwidth in this document is labelled *per direction* or *bidirectional*. Unlabelled
interconnect figures are the most reliable source of 2× errors in this field, so the catalog
type carries both and names them.

**NVLink generation ≠ NVSwitch generation, and they differ by one on Hopper.** The whitepaper is
explicit: *"The eight NVIDIA H100 GPUs in the DGX H100 use the new high-performance
fourth-generation NVLink technology to interconnect through four third-generation NVSwitches."*
So H100 is NVLink **4** over NVSwitch **3**. A Gate A reviewer flagged "gen3" as an Ampere-era
error and proposed gen4; that is the confusion this note exists to prevent, and the whitepaper
quote is why the catalog keeps gen3.

NVIDIA also quotes "13.6 Tbit/s" per NVSwitch, which reconciles with neither a raw nor an
effective reading of 64 ports × 25 GB/s. Nothing here is built on that number.

**Do not compute peak FLOPS from clock × cores** [derived]. `132 SM × 4096 FLOP/SM/clk ×
1.98 GHz = 1070 TF/s`, but NVIDIA publishes 989.5 — an implied tensor clock of ~92.4% of boost,
consistent across SXM and PCIe. Deriving instead of quoting runs ~8% high. NVIDIA never
published an official H100 boost clock (the whitepaper says "Not Finalized").

**Units — bandwidth is decimal, capacity is binary.** These follow different conventions and
assuming one for both is a 7.4% error on the check that decides OOM.

- **Bandwidth is decimal**: 3.35 TB/s means 3.35 × 10¹² bytes/s.
- **Capacity is binary**: the "80 GB" H100 is **80 GiB** — five active HBM3 stacks of 16 GiB —
  which is 85.9 × 10⁹ bytes, not 80 × 10⁹ [derived]. `nvidia-smi` reports **81,559 MiB**, i.e.
  80 GiB less 361 MiB of ECC and context reserve.

So the capacity model is: physical `80 GiB`, addressable `81,559 MiB`, and a further allocator
and workspace reserve on top. Feasibility checks use the addressable figure minus reserve — never
`80 × 10⁹`, which would falsely reject configurations that fit, and never a bare `80 × 2³⁰`,
which would accept configurations that do not.

### 2.3 Derating: what is actually achievable

Peak is never reached. The model computes both a theoretical and a derated result; these are
the defaults, all user-adjustable.

| Resource | Peak | Default | Achieved | Source |
|---|---|---|---|---|
| BF16 GEMM, large square | 989.5 TF/s | 72% | ~713 TF/s | measured cuBLAS 4096³ |
| BF16 GEMM, LLM-shaped skinny | 989.5 TF/s | 55–60% | ~550–590 TF/s | measured, FFN shapes |
| End-to-end training | 989.5 TF/s | **40%** | ~400 TF/s | **Llama 3 405B, 16,384 H100s** |
| HBM, streaming | 3.35 TB/s | 85–90% | ~2.85–3.0 TB/s | STREAM / BabelStream |
| HBM, KV-cache decode | 3.35 TB/s | 60–75% | ~2.0–2.5 TB/s | gather ≠ stream |
| NVLink all-reduce, ring | 450 GB/s | 82% | ~370 GB/s busbw | nccl-tests |
| InfiniBand inter-node | 400 GB/s | ~80% | ~320 GB/s | Azure ND_H100_v5 |

The anchor is Llama 3 405B [The Llama 3 Herd of Models, Table 4]: **400 TFLOP/s per GPU, BF16
MFU 38–43%** on 16,384 H100s — ~40% of dense peak end-to-end, including all communication,
pipeline bubbles, and optimizer overhead.

---

## 3. The workload

### 3.1 Parameter count

Derived structurally from `config.json`, not from a scalar. Dense, SwiGLU/RMSNorm family:

```
d_h        = config.head_dim ?? h / n_q
attn       = h·(n_q·d_h) + 2·h·(n_kv·d_h) + (n_q·d_h)·h        // Q, K, V, O
mlp        = 3·h·f                                              // gate, up, down (SwiGLU)
norms      = 2·h                                                // pre-attn, pre-mlp RMSNorm
per_layer  = attn + mlp + norms
N          = V·h + L·per_layer + h + (tied ? 0 : V·h)
```

MoE replaces `3·h·f` with `E·3·h·f_moe + h·E` (experts + router). Qwen3 adds `2·d_h` per layer
for `q_norm`/`k_norm`. Validated to **0.0000% error** against safetensors metadata for
Qwen3-8B, Llama-3.1-8B, Mistral-7B-v0.3, Mixtral-8x7B, and Qwen3-30B-A3B.

**`safetensors.total` is stored elements, not unique parameters.** Qwen3-0.6B declares
`tie_word_embeddings: true` yet physically stores `lm_head.weight`; HF reports 752 M against a
true 596 M — a 20% error, exactly `V·h`. Memory footprint follows the **stored** count (those
bytes really are loaded); FLOPs per token follows the **analytic** count. The tool computes
both and shows both when they disagree.

### 3.2 config.json landmines

| Concept | Llama/Mistral | Mixtral | Qwen3-MoE | DeepSeek-V3 |
|---|---|---|---|---|
| expert count | — | `num_local_experts` | `num_experts` | `n_routed_experts` |
| expert FFN width | `intermediate_size` | `intermediate_size` | `moe_intermediate_size` | `moe_intermediate_size` |
| dense-layer prefix | — | — | `mlp_only_layers` | `first_k_dense_replace` |
| head dim | absent → `h/n_q` | absent | **present, ≠ `h/n_q`** | MLA: `qk_nope + qk_rope`, `v_head_dim` |

Qwen3-30B-A3B has `head_dim: 128` while `h/n_q = 2048/32 = 64`. Deriving instead of reading is
a **2× KV-cache error**. Always prefer explicit `head_dim`.

---

## 4. FLOPs

### 4.1 Why not the closed form

The widely-cited Megatron scalar [verbatim, Narayanan et al. 2021, Appendix]:

```
96·B·s·l·h²·(1 + s/(6h) + V/(16·l·h))
```

is correct for what it describes and unusable as the general path: it assumes MHA, a 2-matrix
`4h` GeLU MLP, a dense (non-causal) attention mask, and full activation recomputation baked in
via its 4× factor. Llama-class models use SwiGLU (3 matrices) and GQA. The tool therefore
computes FLOPs from actual weight shapes, mirroring the current reference implementation
— `megatron/training/training.py :: num_floating_point_operations` — which supersedes the 2021
appendix and handles GQA, SwiGLU, MoE, MLA, and Mamba/GatedDeltaNet layers. The closed form is
retained only as a cross-check for the architecture it was derived for.

### 4.2 Per-kernel-class FLOPs (forward, per rank)

With `T = b·q` tokens processed this step and tensor-parallel degree `t`:

| Class | FLOPs |
|---|---|
| QKV projection | `2·T·h·(n_q + 2·n_kv)·d_h / t` |
| Attention output projection | `2·T·h·n_q·d_h / t` |
| Core attention | `4·b·n_q·d_h·[q·(s−q) + q(q+1)/2] / t` — see below |
| MLP (SwiGLU, 3 matrices) | `6·T·h·f / t` |
| MLP (classic, 2 matrices) | `4·T·h·f / t` |
| MoE routed experts | `6·T·h·f_moe·k / t` (`k` = top-k) |
| LM head | `2·T·h·V / t` |
| Norms, RoPE, residual | `O(T·h)` — negligible in FLOPs, **not** in bytes |

Backward costs 2× forward (`dX` and `dW` GEMMs), so training is 3× forward per GEMM
[verbatim, Megatron-LM source: *"3x: Each GEMM in the model needs to be performed 3 times
(forward pass, backward wgrad, backward dgrad)"*].

**The causal discount is not a blanket ÷2** [derived]. A causal mask halves the work only when
averaged over a *full* prefill from an empty context, where the mean query attends `s/2` keys.
It does **not** apply to decode, where the single new query attends the entire existing context.
The general count of query–key pairs for `q` new queries following a prefix of `s−q` is

```
pairs = q·(s − q) + q·(q + 1)/2
```

which specialises correctly in both directions: at `q = s` (full prefill) it is `s(s+1)/2 ≈ s²/2`,
recovering the ÷2; at `q = 1` (decode) it is exactly `s`, with **no discount at all**. Applying a
blanket ÷2 understates decode attention by 2×, and understates chunked prefill by an amount that
varies with chunk position. The tool implements the general form.

### 4.3 The causal-mask divergence — a live inconsistency in the literature

Narayanan et al. 2021 and PaLM count core attention as `4·b·s²·h` forward (full `s×s` matrix).
Kaplan et al. Table 1 and **current Megatron-LM** count `2·b·s²·h` (causal ⟹ only half the
score matrix is needed). Current Megatron's comment is explicit: *"2 * sum(L^2) * h * p: core
attention (causal mask -> /2 cancels with FMA *2)."*

Consequence: **published MFU figures from before and after this change are not comparable at
long context.** The tool exposes the convention as a switch and labels which one produced a
number. Default: causal (÷2), matching current practice.

### 4.4 `6ND` and where it breaks

`C_train ≈ 6·N·D` [verbatim, Kaplan et al. §2.1] is the right first-order estimate, with three
documented failure modes:

1. `N` **excludes embeddings** in Kaplan's definition, and the reason is not double-counting —
   it is that an embedding lookup is a *gather*, costing ~0 FLOPs rather than the `2N` per token
   a dense weight matrix costs. Feeding a HuggingFace total parameter count into `6ND` therefore
   attributes matrix-multiply cost to a table lookup. The fix is to define `N` as the
   non-embedding transformer weight count and model the LM-head projection separately where it
   actually executes — not to subtract a supposed duplicate from a shape-derived total.
2. It **omits the quadratic attention term**, error ≈ `s/(6h)` — 4% at `s=2048, h=8192`, 17% at
   `s=h=8192`, and **133% at `s=131072, h=16384`**, i.e. at long context `6ND` is wrong by more
   than a factor of two, not by a fraction.
3. The **LM head** breaks it at small scale: measured discrepancies up to ~8.5× in extreme
   small-model/large-vocab corners.

The tool exposes the head as a separate toggleable term. It is 30–60% of total FLOPs for a 1B
model with a 256k vocab, and under 2% for a 400B model.

### 4.5 MoE

FLOPs scale with **active**, not total, parameters, and linearly in top-`k` rather than in the
expert count. `N_active` = non-expert parameters + shared experts + `k` routed experts.
DeepSeek-V3 is the canonical case: **671B total, 37B activated per token**.

With token dropping, *hardware* FLOPs scale by the capacity factor while *model* FLOPs do not.
Router FLOPs (`2·T·h·E`) are omitted as negligible.

---

## 5. Memory traffic

Rooflining requires bytes moved, not just FLOPs. Per kernel class, per rank, lower bounds:

**These are per kernel invocation — that is, per layer — not per step.** A single attention
kernel evaluates one layer. Multiplying by `L` here and then summing over layers in the scheduler
would overstate traffic by `L`, an 80× error on Llama-3-70B. The `L` factor belongs in the
scheduler, which instantiates `L` copies of each class.

| Class (one layer) | HBM bytes |
|---|---|
| Any weight-stationary GEMM | `w·(|X| + |W| + |Y|)` |
| Decode attention | `≥ 2·b·s·(n_kv/min(t,n_kv))·d_h·w_kv` — the KV read |
| Prefill attention, FlashAttention | linear in `s` for Q/K/V/O tiles; **no materialised `s²`** |
| Prefill attention, naive | adds `O(w·b·(n_q/t)·s²)` of score/probability traffic |
| Norms / RoPE / residual | `O(w·T·h)` read + write — memory-bound and non-trivial at `b=1` |
| KV-cache write (prefill) | `2·b·s·(n_kv/min(t,n_kv))·d_h·w_kv` |

FlashAttention is the default and matters enormously: it leaves attention FLOPs quadratic in
`s` while keeping HBM traffic linear. Modelling attention as materialising an `s²` score matrix
overstates decode and long-context prefill traffic by orders of magnitude.

---

## 6. KV cache

### 6.1 Standard (MHA / GQA / MQA)

[verbatim, NVIDIA: *"Size of KV cache per token in bytes = 2 * (num_layers) * (num_heads *
dim_head) * precision_in_bytes"*; *"The first factor of 2 accounts for the K and V matrices."*]

```
KV_bytes (cluster) = 2 · L · n_kv · d_h · s · b · w_kv
KV_bytes (per GPU) = 2 · L · (n_kv/t) · d_h · s · b · w_kv
```

**Scope is stated explicitly here and everywhere KV appears**, because getting it wrong is the
error that started this project: a cluster-wide KV figure quoted as per-GPU is a `t`-fold
overstatement, 8× at TP=8. Tensor parallelism shards KV by attention head, so the per-GPU form
divides `n_kv` by `t` — valid while `t ≤ n_kv`. Beyond that heads cannot be split further and
implementations replicate KV instead, so the per-GPU divisor is `min(t, n_kv)`.

The factor of 2 is K and V. `n_kv` — not `n_q` — is the point of GQA: Llama-3-70B has `n_q=64,
n_kv=8`, an 8× reduction. Using `n_q` overstates KV traffic by 8×.

Real-world corrections the tool exposes: paged allocation rounds up to whole blocks
(`ceil(s/block)·block`, vLLM default block 16); engines that preallocate use `max_model_len`
rather than actual length.

### 6.2 MLA (DeepSeek) [verbatim, DeepSeek-V2 Table 1]

MLA caches a single low-rank latent plus a shared decoupled RoPE key, so **there is no factor
of 2** — there is no separate V:

```
KV_bytes_MLA = (kv_lora_rank + qk_rope_head_dim) · L · s · b · w_kv
```

DeepSeek-V3: `512 + 64 = 576` elements per token per layer. Applying the standard GQA formula
to DeepSeek (`2 · 128 · 128 = 32768`) overstates its KV cache by ~57×. This is a guarded case
in the golden tests.

### 6.3 Sliding-window and hybrid

```
KV_bytes (per GPU) = 2 · (n_kv/min(t, n_kv)) · d_h · b · w_kv
                     · (L_global · s + L_local · min(s, W))
```

Layers with no attention KV at all (Mamba/SSM, linear attention, gated delta) carry a
fixed-size recurrent state independent of `s` and are excluded from `L`.

---

## 7. Capacity — checked before any timing

An out-of-memory configuration has no step time, and reporting a fast one is the worst
possible failure. Feasibility runs first.

**Inference**, per GPU:

```
weights/(t·p) + KV/p + activations/workspace + runtime reserve
```

Pipeline parallelism divides the *layers* across `p` stages, so a GPU holds only `L/p` layers'
worth of both weights and KV cache. Omitting `p` falsely rejects valid pipeline configurations as
infeasible. (Stages are only equal when `L` divides `p`; the tool sizes against the largest stage,
`ceil(L/p)` layers, since that is the one that OOMs first.)

**Training**, per GPU, mixed-precision Adam [verbatim, ZeRO §3.1]:

| Component | Precision | Bytes/param |
|---|---|---|
| Working parameters | bf16 | 2 |
| Gradients | bf16 | 2 |
| Master parameters | fp32 | 4 |
| Adam momentum | fp32 | 4 |
| Adam variance | fp32 | 4 |
| **Total** | | **16Ψ** (`K = 12`) |

**The commonly cited "18" and "20 bytes/param" are not paper-sourced.** They are framework
configurations — fp32 gradients (Megatron `--accumulate-allreduce-grads-in-fp32`, FSDP fp32
reduce) give 18; an fp32 master plus a bf16 shadow copy gives 20. They are offered as named
configuration variants, never as cited constants. SGD+momentum is `K=4` ⟹ 8Ψ; 8-bit Adam ≈ 8Ψ.

**ZeRO / FSDP stages** [verbatim, ZeRO Table 1]:

| Stage | Per-device model state (bytes) | Comm volume (**elements**) |
|---|---|---|
| Baseline DP | `4Ψ + KΨ` = 16Ψ | 2Ψ |
| Stage 1 (`P_os`) | `4Ψ + KΨ/N_d` | 2Ψ |
| Stage 2 (`P_os+g`) | `2Ψ + (2+K)Ψ/N_d` | 2Ψ |
| Stage 3 (`P_os+g+p`) | `(4+K)Ψ/N_d` | **3Ψ (1.5×)** |

**Unit warning — the two columns use different conventions.** In the memory column `Ψ` is a
parameter *count* and the coefficients are bytes-per-parameter, so `16Ψ` is bytes. In the
communication column ZeRO's `2Ψ`/`3Ψ` are counts of *elements moved*, not bytes. Feeding them
into `T = α + M/algbw` unconverted underestimates a bf16 all-reduce by 2× and an fp32 one by 4×.
The conversion happens exactly once, at the boundary: `M = (volume in elements) · w_grad`.

**Activation memory** [verbatim, Korthikanti Eq. 1], per layer:

```
s·b·h·(34 + 5·a·s/h)
```

Under tensor+sequence parallelism, Eq. (4) divides **the entire bracket** by `t`, not just the
34 — the `5·a·s/h` attention-square term is sharded too:

```
per layer, TP+SP:   (s·b·h/t)·(34 + 5·a·s/h)
```

which is simply Eq. (1) ÷ `t`. Sharding only the 34 would overstate activation memory by up to
`t` on the GPT-style non-FlashAttention path. Selective recomputation (Eq. 6) gives
`34·s·b·h·L/t`, dropping the attention-square term entirely.

**The constant 34 is architecture-specific** — it assumes GPT-2 style:
dropout ≠ 0, 2-matrix GeLU MLP, LayerNorm, no FlashAttention. Modern models (dropout 0, SwiGLU,
RMSNorm, FlashAttention) differ, and FlashAttention removes the entire `5as²b` term
unconditionally. It is therefore a parameter, not a constant.

Pipeline parallelism multiplies by `(1 + (p−1)/(p·m))`; the first stage stores `p` microbatches
of activations regardless of holding only `L/p` layers.

### 7.1 A confirmed arithmetic error in the published source

**Korthikanti et al. Eq. (8) does not follow from its own derivation.** The paper states that
selective recomputation adds an extra forward attention-matrix computation (`2Bs²h`) and
attention-over-values (`2Bs²h`) — `+4·B·s²·h` per layer, so `4·B·L·s²·h` in total. Against the
`72·B·L·s·h²` prefactor that extra work is:

```
4·B·L·s²·h / (72·B·L·s·h²)  =  4s/(72h)  =  s/(18h)
```

The paper's own base attention term is `s/(6h) = 3s/(18h)`, so the corrected **total** should be
`3s/(18h) + s/(18h) = 4s/(18h) = 2s/(9h)`. **The paper prints `s/(3h) = 6s/(18h)`** — 1.5× the
term it just derived.

Eq. (9) is the *ratio* `hardware/model`, and it inherits the error the same way:

```
(1 + 2s/(9h)) / (1 + s/(6h))  ≈  1 + (4−3)s/(18h)  =  1 + s/(18h)
```

against the paper's printed `≈ 1 + s/(6h)`.

Impact on the hardware-FLOPs prefactor: the paper overstates by **1.8% at `s/h = 1/6`** and
**9.1% at `s/h = 1`**. The tool implements:

```
model_FLOPs    = 72·B·L·s·h²·(1 + s/(6h)  + V/(12hL))
hardware_FLOPs = 72·B·L·s·h²·(1 + 2s/(9h) + V/(12hL))   ← corrected
```

with the paper's `s/(3h)` available behind a "match published paper" toggle.

---

## 8. Collectives

### 8.1 The busbw double-counting trap

`nccl-tests` reports two bandwidths [verbatim, `doc/PERFORMANCE.md`]:

```
algbw = S/t
busbw = algbw · 2(n−1)/n        (all-reduce)
```

and — the sentence that makes this a trap — *"this is independent of the algorithm used (ring,
tree, or other)."* That is to say **`nccl-tests` multiplies the ring factor in unconditionally**,
whatever algorithm actually ran. So a published busbw already contains 1.75 at `n=8`. Dividing by
it without re-applying the factor underpredicts time by 1.75×; applying the factor to an algbw
figure overpredicts by the same.

To be unambiguous about what *this tool* does: **it never applies a ring factor when timing a
collective.** It stores algbw, times as `T = α + M/algbw` where `M` is the algorithmic message
size, and that is the whole of it. The `2(n−1)/n` factor appears in exactly two other places —
converting a published ring busbw into algbw at the catalog boundary, and computing physical wire
traffic when that is what is being reported to the user — and nowhere else.

This is the single most likely place for this model to go wrong, and it was independently
identified as such by both reviewers. **Structural fix: the core stores `algbw` only**, and
computes `T = α + M / algbw`. Published busbw is converted at the data-entry boundary by one
documented helper, so the factor cannot be applied twice inside the model.

Correction factors, algbw → busbw [verbatim]: all-reduce `2(n−1)/n`; all-gather and
reduce-scatter `(n−1)/n`; broadcast and reduce `1`; all-to-all `(n−1)/n` (reported in the
summary table but not derived in the document body — flagged).

### 8.2 Cost model

```
T_coll(M) = α + M / algbw(algo, M)
```

Ring and NVLS differ in the physical bytes crossing each port [derived]:

- **Ring**: each rank sends and receives `2(p−1)/p · M` — 1.75 M at `p=8`.
- **NVLS** (NVSwitch in-network reduction): each rank sends `M` up and receives `M` down. The
  ring factor genuinely vanishes.

This was a live disagreement between reviewers, settled by head-to-head: the ring factor does
vanish under NVLS, **but** the denominator is not the 450 GB/s line rate. Measured large-message
algbw is what belongs there.

| Algorithm | busbw [prior] | **algbw** [prior] | α [prior] |
|---|---|---|---|
| Ring, 8×H100 | ~370 GB/s | **~211 GB/s** | 5 µs (3–10) |
| NVLS, 8×H100 | ~480 GB/s | **~274 GB/s** | 5 µs (3–10) |
| NCCL ≥ 2.27 symmetric memory | — | — | 3 µs (2–6) |

The reported NVLS busbw of 480 GB/s **exceeds the 450 GB/s line rate**, which is itself proof
the conversion is wrong for in-network reduction; nccl-tests PR #239 proposes `(n−1)/(n+1)` for
NVLS. It is a benchmark metric, not wire traffic.

Algorithm selection is `argmin_a T_a(M)` over eligible algorithms. NVLS requires NVSwitch and
NCCL ≥ 2.17, and loses to ring below roughly 1 MB.

**These are priors, not specifications.** No credible published `all_reduce_perf` table exists
for a single 8×H100 node with the NCCL version, command line, and NVLS state needed to fit
them, and none is invented here. The calibration command is documented so a user with hardware
can replace them:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
  ./build/all_reduce_perf -b 1K -e 1G -f 2 -g 8 -n 100
```

Fit `α` from the small-message intercept and `algbw` from the large-message asymptote, running
`NCCL_ALGO=Ring`, `NCCL_ALGO=NVLS`, and unset. The **default** curve is the one to use.

### 8.3 Alpha-beta reference [unverified — standard forms, sources returned 403]

| Algorithm | Cost |
|---|---|
| Ring all-reduce | `2(p−1)·α + 2·((p−1)/p)·n·β + ((p−1)/p)·n·γ` |
| Rabenseifner (RS+AG) | `2·lg(p)·α + 2·((p−1)/p)·n·β + ((p−1)/p)·n·γ` |
| Recursive doubling (short) | `lg(p)·(α + n·β + n·γ)` |

Both `α·(p−1)` (ring) and `α·log p` (tree) appear in practice because NCCL selects tree for
small messages and ring for large.

---

## 9. Parallelism communication volumes

**Tensor parallel** [verbatim, Shoeybi §3]: `f` is *"an identity operator in the forward pass
and all reduce in the backward pass"*, `g` the reverse — *"only two all-reduces in the forward
path and two in the backward path"* per layer.

```
per all-reduce, algorithmic:  b·q·h·w
per layer per device, fwd+bwd (wire):  8·b·s·h·w·(t−1)/t      [verbatim ×w]
```

Note the `8·b·s·h·(t−1)/t` figure already folds the NCCL factor in for 4 collectives
(`4 × 2(t−1)/t`). Mixing it with a further ring factor is the trap of §8.1 again.

**Sequence parallel** [verbatim, Korthikanti §4.2.2]: replaces 4 all-reduces with 4 all-gathers
+ 4 reduce-scatters. *"the communication bandwidth used for tensor parallelism and tensor
together with sequence parallelism are the same."* Wire volume is unchanged; **latency doubles**
(8 launches instead of 4), and the paper measures the split pair as slightly *slower* than the
fused all-reduce. Its win is memory, not communication.

**Data parallel**: `2·(d−1)/d · Ψ · w` per step. ZeRO-1/2 = 2Ψ (same); ZeRO-3 = 3Ψ (1.5×).

**Pipeline parallel** [verbatim]: `b·s·h` between consecutive stages per microbatch. Bubble
fraction `(p−1)/m`, or `(1/v)·(p−1)/m` interleaved — *"the amount of communication also
increases by v."*

**Expert parallel** [derived — no canonical primary source; flagged]: dispatch + combine per
forward, mirrored in backward ⟹ 4 all-to-alls per MoE layer per iteration.

```
V_a2a = b·s·k·h·w·(e−1)/e     per direction, per device, per MoE layer
```

Multiply by capacity factor if the implementation pads. Shared experts are not routed and
contribute no all-to-all.

### 9.1 Tensor-parallel decode, and the result worth building the tool around

```
per step, algorithmic:  2·L·b·h·w
per step, wire:         4·L·b·h·w·(t−1)/t
```

[Corroborated verbatim: *"In the decode stage, the communication size of each All-Reduce is
`2bh` bytes under 16-bit precision."*]

Network intensity of a TP decode step [derived]. Both numerator and denominator must be taken
**at the same scope** — mixing cluster FLOPs with per-rank bytes inflates the result by exactly
`t`, which at TP=8 is another 8× scope error of precisely the kind §1 exists to prevent.

Per rank, per decode step:

```
F_rank  = 2·b·N_active/t  +  4·b·L·s·n_q·d_h/t        (linear + attention)
M_alg   = 2·L·b·h·w                                    (algorithmic: 2 all-reduces per layer)
M_wire  = 4·L·b·h·w·(t−1)/t                            (ring physical traffic)
```

Under the algbw convention this model uses throughout — `T = α + M_alg/algbw`, no ring factor —
the intensity is:

```
I_N = (N_active + 2·L·s·n_q·d_h) / (L·h·w·t)
```

and against physical wire bytes at physical link bandwidth it would instead be
`(N_active + 2·L·s·n_q·d_h) / (2·L·h·w·(t−1))`. **Use one convention or the other, never a
numerator from one and a denominator from the other.** The tool uses the algbw form.

Two properties survive, and they are the point:

- **`I_N` is independent of batch size.** Both FLOPs and network bytes scale with `b`, so it
  cancels. You cannot batch your way out of the network roof in tensor-parallel decode — which is
  exactly the opposite of the HBM roof, where batching is the whole remedy.
- **Raising `t` lowers `I_N` while raising peak FLOPS linearly.** That is the hard scaling limit
  of tensor-parallel inference.

It is *not* independent of sequence length: the attention term `2·L·s·n_q·d_h` grows with
context, so long-context decode drifts away from the network roof even as it sinks deeper into
the memory roof. An earlier draft of this document omitted that term and thereby declared a
long-context effect nonexistent.

Neither property is visible on a classic two-roof roofline, and showing them is the clearest
justification for this tool existing.

### 9.2 Vocabulary-parallel logits and sampling

The "two all-reduces per layer" accounting covers the transformer body only. With a
vocabulary-parallel LM head there is additional communication at the output, and for a small
model with a large vocabulary it can dominate the decode-step classification:

- **Distributed sampling** (the cheap path): a max-reduce and a sum-reduce over `b` partial
  logits, plus candidate exchange. Tiny payloads, so it adds ~2 more α-dominated collectives.
- **Logits gather** (the expensive path): an all-gather of order `b·V·w` bytes. At `V = 151936`,
  `b = 32`, bf16 that is **9.7 MB** — comfortably bandwidth-bound and larger than the entire
  transformer body's collective traffic for that step.

Which one applies is an implementation choice, so it is an explicit mode rather than a silent
omission. Default: distributed sampling.

### 9.3 Worked check

Llama-3-70B, `b=1`, `t=8`, bf16: message is `1 × 8192 × 2` = **16 KiB**;
bandwidth term across all 160 collectives ≈ 10 µs; latency term ≈ `160 × 5 µs` = **800 µs**,
roughly **80× the bandwidth term**. Batch-1 TP decode is *latency*-bound, and no amount of
faster interconnect fixes it.

---

## 10. Aggregation: why per-kernel, and how

### 10.1 Step-level rooflining is a category error

Computing one aggregate `t_compute`, `t_memory`, `t_network` for a whole step and taking the
argmax is invalid. Within a *single kernel*, memory and compute genuinely overlap, so `max()`
is correct there. Across kernels it is not, by Jensen's inequality:

```
Σ max(tᶜᵢ, tᵐᵢ)  ≥  max(Σ tᶜᵢ, Σ tᵐᵢ)
```

Averaging a compute-bound MLP with a memory-bound attention parks the aggregate intensity near
the ridge point and predicts near-peak utilisation that never occurs. A step-level arithmetic
intensity `Σ F / Σ B` is a fine descriptive statistic and a bad latency predictor.

**Therefore: roofline per kernel class, then combine.**

### 10.2 The scheduler

No global rule ("step time is max" / "is sum") is valid — the answer depends on dependency
edges and resource contention. The core runs a small list scheduler over a task DAG:

```
task     = (id, deps, resource ∈ {SM, HBM, NVLink}, duration)
kernel   duration = max(F/C_eff, H/B_eff, T_launch)
coll     duration = α + M/algbw
start_i  = max( max_{d ∈ deps} finish_d , resourceAvailable(resource_i) )
T_step   = makespan
```

The phase-specific behaviour then falls out as a *consequence* rather than a hardcoded policy:

- **TP decode** — each all-reduce blocks the next GEMM; there is no independent work to hide it,
  so collectives land on the critical path.
- **Prefill** — large GEMMs create genuinely independent work, so collectives are partly hidden.
- **DP training** — gradient buckets reduce while lower layers are still computing backward, so
  only the tail is exposed: `T ≈ T_fwd+bwd + max(0, finish_last_bucket − end_of_backward)`.

Concurrency is credited only where the DAG permits it. When compute and NCCL do run together,
a contention penalty applies — NCCL reductions consume SM and HBM resources, so crediting both
their full standalone rates is wrong.

---

## 11. Metrics

### 11.1 Inference [verbatim, NVIDIA GenAI-Perf / vLLM]

```
TTFT          = queue + prefill + first-token network
e2e_latency   = TTFT + ITL·(n_out − 1)
ITL / TPOT    = (e2e_latency − TTFT) / (n_out − 1)
TPS           = Σ output tokens / wall clock
tokens/s/user = 1 / TPOT
```

**Terminology collision, resolved explicitly**: NVIDIA treats ITL and TPOT as synonyms (a
per-request average). vLLM distinguishes them — TPOT is the per-request average, ITL is the raw
list of per-token inter-arrival gaps whose percentiles capture jitter. This tool reports a
single steady-state figure and labels it **TPOT**, noting it equals NVIDIA's ITL.

**Goodput**: fraction of requests meeting *all* specified SLOs simultaneously.

### 11.2 MFU, HFU, MBU

```
MFU = model_FLOPs_per_iter    / t_iter / (n_devices · C_peak)
HFU = hardware_FLOPs_per_iter / t_iter / (n_devices · C_peak)
MBU = HBM_bytes_moved         / t_iter / (n_devices · B_mem_peak)
```

MFU and HFU share a denominator and differ only in the numerator [verbatim, Korthikanti §6.3]:
model FLOPs are *"the floating point operations required to perform a single forward and
backward pass regardless of the implementations and hardware limitations"*; hardware FLOPs are
*"the floating point operations that are actually performed on the hardware."* So **HFU ≥ MFU
always**, and full recomputation *raises* HFU while *lowering* MFU.

PaLM's form [verbatim, Appendix B]: `R = P / (6N + 12·L·H·Q·T)`, MFU = observed tokens/s ÷ R.

**The conflation trap, guarded in the golden tests**: Megatron 2021's widely-quoted "52% of
peak" is **HFU** — its formula bakes in the 4× recomputation factor. PaLM independently
computes the *same run's* MFU as **30.2%**. PaLM's own figures: MFU 46.2%, HFU 57.8%.

**MFU is meaningless for decode.** A memory-bound decode step at 2–4% MFU is not a failure; it
is the correct answer. MBU is the metric that carries information there, which is why both are
always reported.

Reference MFU values: GPT-3 21.3%, Gopher 32.5%, MT-NLG 30.2%, PaLM 46.2% [PaLM Table 3];
Llama 3 405B 38–43% [Llama 3 Table 4].

---

## 12. Bottleneck classification

"Communication-bound" is not one thing, and collapsing it hides the fix. Five classes:

| Class | Criterion | What actually helps |
|---|---|---|
| Compute-bound | compute on critical path, `I > I_ridge` | lower precision, better kernels |
| HBM-bandwidth-bound | memory on critical path | quantization, larger batch, cache reuse |
| Network-bandwidth-bound | collectives on critical path, `M ≫ α·algbw` | less TP, resharding, faster fabric |
| Network-latency-bound | collectives on critical path, `M ≪ α·algbw` | fewer collectives, CUDA graphs, less TP |
| Dependency/serialisation-bound | critical path dominated by ordering | restructure the schedule |

The knee between the two network classes is `M ≈ α · algbw` — about **1.4 MB** at α=5 µs,
algbw=274 GB/s. Batch-1 TP=8 decode sits at 16 KiB, two orders of magnitude below it. A
three-way model would label that "communication-bound" and send the user to buy faster
interconnect, which would change nothing.

---

## 13. Plot geometry

### 13.1 Roofline (primary)

Classic two-roof, log-log: `P_attainable(I) = min(C, I · B_mem)` [verbatim, Williams et al.
2009], ridge at `I = C / B_mem` — **295 FLOP/byte for H100 BF16**, 591 for FP8.

**Decode plots as two points, not one.** Both are *weight-* and *KV-dominant asymptotes*, valid
when weight traffic dwarfs activation traffic (true in decode, where activations are a single
token wide). The scheduler uses exact per-projection traffic; these closed forms exist to be read,
not to be computed with.

```
linear layers, exact:      I = 2b / (w_weight + b·w_act·(1/h + 1/d_out))
linear layers, asymptote:  I ≈ 2b / w_weight            → I ≈ b for bf16 weights
decode attention:          I = 2·n_q / (n_kv · w_kv)    → I = n_q/n_kv for bf16 KV
```

The activation term in the exact linear form is negligible while `b ≪ h`, which is the whole of
the decode regime — but it is what stops the asymptote from running away as batch grows, so the
scheduler uses the exact form and the plot annotates the asymptote.

The linear point walks right as batch grows and eventually crosses the ridge. The attention point
**never moves with batch**, because each sequence owns its KV cache and batching cannot amortise
it — that is why long-context decode never becomes compute-bound at any batch size.

Two precision subtleties, both of which are levers the tool should surface:

- `I = 2b/w_weight` gives `2b` only for 1-byte weights. But the FP8 ridge point roughly doubles
  too, so the crossover batch barely moves. W4A16 is the interesting case: weight bytes quarter
  while FLOPs stay at bf16 rate, so the point moves right by 4× against an unchanged ridge.
- The attention point **does** depend on KV precision: it is `2n_q/(n_kv·w_kv)`, not `n_q/n_kv`.
  For Llama-3-70B it is 8.0 with bf16 KV and **16.0 with fp8 KV**. Quantizing the KV cache moves
  the one point that batching cannot move.

### 13.2 Why there is no network roof here

Drawing a network ceiling at `FLOPs_total / t_net` on classic axes is a category error: adding
dummy FLOPs to a kernel makes the ceiling *rise*. Communication consumes neither the FLOPs nor
the DRAM bytes those axes measure.

### 13.3 Ridgeline (secondary, scoped)

Ridgeline [Checconi, Tithi, Petrini, arXiv:2209.01368] fixes this by re-choosing the axes:
`x = B_M/B_N`, `y = F/B_M`, so `x·y = F/B_N`. All three boundaries become **pure machine
constants**:

| Boundary | Equation | Shape |
|---|---|---|
| compute vs memory | `y = C/B_mem` | horizontal |
| memory vs network | `x = B_mem/B_net` | vertical |
| compute vs network | `x·y = C/B_net` | diagonal, slope −1 |

meeting at `(B_mem/B_net, C/B_mem)`. Dummy FLOPs now move the *workload point* up into the
compute-bound quadrant, which is physically correct.

**It is nevertheless secondary, for two reasons.**

1. **Latency breaks it.** Ridgeline assumes `T_net = B_N/B_net`. With a real α the
   compute-vs-network boundary becomes `x·y = C/B_net + C·α/B_N` — dependent on **absolute
   message size**, so no single Ridgeline covers both a 16 KiB and a 1 GiB all-reduce. Two
   workloads scaled 1000× apart share coordinates while one is α-dominated. As `B_N → small`,
   `x → ∞` and the point lands in the wrong quadrant entirely.
2. **It loses attainable performance.** Both axes are intensities, so the vertical gap that
   makes a classic roofline readable is gone. Recoverable as
   `P_pred = min(C, y·B_mem, x·y·B_net)`, but not readable off the plot.

Plus the Jensen objection applies unchanged: GEMMs have `B_N = 0` so `x → ∞`; standalone
all-reduces have `F ≈ 0` so `y → 0`; a step-level point averages them into a coordinate that
can land in the compute-bound region for a step that was never compute-bound.

**So it ships plotted per kernel class, never per step, with an explicit scope note**: *assumes
bandwidth-dominated communication; excludes collective startup latency, serial ordering, and
launch effects.* Its one good job is comparing how batch, sequence length, and parallelism
degree shift bandwidth pressure between regimes.

The other citable third-roof formulation is Cardwell & Song, HPC Asia 2019 [unverified — ACM DL
paywalled; reconstructed from Checconi et al.'s description]. No NVIDIA or Cerebras "roofline
for collectives" appears to exist.

---

## 14. Golden scenarios

Ranges, not point values — implementation choices legitimately move these by tens of percent.

**1. Llama-3-70B BF16, 8×H100 SXM5, TP=8, b=32, s=2048.** Cross-checked three independent ways.

| Quantity | Expected |
|---|---|
| Weights per GPU | 17.5 GB |
| KV read per GPU per step | **2.68 GB** (not 21.47 — that is the cluster total) |
| HBM traffic per GPU per step | 20.2 GB |
| Decode step | 7–9 ms |
| Aggregate throughput | 3,700–4,500 tok/s |
| Per-user throughput | 116–141 tok/s |
| MBU | 70–80% |
| MFU | 5–8% (**correct, not a bug**) |
| Decode intensity vs ridge | ~28 vs 295 FLOP/byte |
| Class | HBM-bandwidth-bound |

Prefill of the same batch: ~9.5 PFLOP, TTFT 2.2–3.0 s, 22k–30k tok/s, compute-bound.

Two of these were corrected once the implementation existed. An earlier draft
put MFU at 2–4% and MBU at ~85%: the MFU figure came from a reviewer's decode
step time that was 2× too slow (the 8× KV error), and 4.51 TFLOP against
7.912 PF/s over ~8.5 ms is 6.7%, not 3%. The MBU figure assumed the whole step
was memory traffic, but ~13% of it is collectives, so ~73% is the honest number.

**2. Llama-3-70B, b=1, TP=8** — the *collectives* are latency-bound (16 KiB
messages, ~60 ns of transfer against a ~5 µs floor), but the **step** stays
memory-bandwidth-bound, because a single sequence still streams all 17.6 GB of
per-rank weights. Latency is ~10% of the step, not the whole of it.

This distinction is worth stating carefully rather than sloganising: a workload
can be built from latency-bound collectives while remaining memory-bound
overall. What latency actually does here is cap tensor-parallel scaling — see
scenario 3.

**3. Llama-3.1-8B, b=1, prompt 512, TP swept 1→8** — the tool's headline result,
and the clearest thing it can show. Memory time halves with each doubling of `t`
while collective latency stays fixed, so latency's share of the step climbs
monotonically and the speedup decays:

| TP | step | speedup vs previous | latency share |
|---|---|---|---|
| 1 | 5.20 ms | — | 0% |
| 2 | 2.86 ms | 1.82× | 9% |
| 4 | 1.57 ms | 1.82× | 17% |
| 8 | 0.93 ms | 1.69× | 28% |

Nothing about the interconnect's *bandwidth* is binding at any point. This is
why "communication-bound" as a single category is not good enough: the remedy
here is fewer collectives, and faster links would change nothing.

**3. 8B at large batch** — linear-layer point must cross the ridge; attention point must not.

**4. W4A16** — must shift the class off HBM-bound by shrinking weight bytes without shrinking
FLOPs.

**5. Mixtral-8x7B** — active vs total parameters; all-to-all appears.

**6. DeepSeek-V3** — MLA KV must be `576·L·s·b·w`, guarding the ~57× overestimate.

**7. A training step** — MFU < HFU; ZeRO stage changes capacity but not (much) comm volume
except stage 3 at 1.5×.

---

## 15. Known limits

Not modelled: wave quantization and SM tail effects; cache-hierarchy reuse beyond a per-kernel
bound; scheduler and continuous-batching behaviour; clock and power throttling; paged-KV
fragmentation; kernel-selection heuristics; NUMA and host-transfer effects.

Every latency and efficiency constant marked **[prior]** is a fitted guess, not a vendor
specification. The calibration panel exists so that anyone with hardware can replace them, and
results are labelled *calibrated* or *prior* accordingly.

### Gate A record

This document was reviewed before any code existed, independently by `codex` and by `agy`, each
asked to produce a defect list rather than a summary. They returned 14 and 10 findings; four
overlapped. Everything they found was checked by hand before being accepted, and one finding was
rejected. Accepted defects, all now fixed above:

| Defect | Consequence had it shipped |
|---|---|
| Causal ÷2 applied to decode as well as prefill | decode attention understated 2× |
| TP-decode `I_N` divided cluster FLOPs by per-rank bytes | headline result inflated by `t` (8× at TP=8) |
| `I_N` omitted the decode-attention term | a long-context effect declared nonexistent |
| §5 per-kernel traffic included the layer count `L` | single-kernel traffic overstated 80× on a 70B model |
| Inference capacity omitted the pipeline divisor `p` | valid PP configurations falsely rejected as OOM |
| Capacity treated as decimal (`80×10⁹`) | 7.4% understatement on the check that decides OOM |
| H100 PCIe/NVL given the SXM5 NVLink figure | intra-node bandwidth overstated 1.5× |
| Decode attention intensity stated as `n_q/n_kv` | KV quantization silently ignored |
| Roofline points presented as exact rather than asymptotic | intensity wrong once `b` approaches `h` |
| ZeRO comm volumes (elements) not converted to bytes | bf16 gradient all-reduce underestimated 2× |
| Korthikanti Eq. (4) sharding stated as `34 → 34/t` only | activation memory overstated up to `t` |
| Extra recompute work stated as `16·B·L·s²·h` | arithmetic slip; correct is `4·B·L·s²·h` |
| `6ND` long-context error stated as 50% | actual is 133% — wrong by more than 2×, not a fraction |
| `6ND` embedding exclusion explained as "double-counting" | wrong diagnosis, would suggest the wrong fix |
| Vocabulary-parallel logits/sampling communication omitted | decode classification wrong for small-model/large-vocab |

**Rejected:** a reviewer flagged "NVSwitch gen3" as an Ampere-era error and proposed gen4. The
Hopper whitepaper states fourth-generation NVLink over *third*-generation NVSwitch; gen3 stands,
and §2.2 now carries the quote so the point is not re-litigated.

Two of the accepted defects were in text this document uses to warn against that very error — the
scope-mixing in `I_N`, and a unit confusion in the section about unit confusion. That is the
argument for the gate existing, not against it.

### Gate B record

The implementation was then reviewed against this specification, again by an independent model
asked for a defect list rather than a summary. It returned 14 findings. Every one was checked by
hand; the confirmed ones are fixed and pinned by `src/core/gate-b.test.ts`.

| Defect | Consequence had it shipped |
|---|---|
| `ceil(L/p)` layers on *every* pipeline stage | 80 layers over 6 stages modelled as 84 |
| LM head run on every pipeline stage, then multiplied by `p` | head charged `p` times |
| Selective recompute as a blanket 1.1× across all kernels | an **invented constant**, wrong at every `s/h` |
| TP-only and TP+SP activation branches algebraically identical | sequence parallelism silently saved nothing |
| DP all-reduce depending on the last compute task | overlap structurally impossible while claimed in a comment |
| DP all-reduce inside the microbatch schedule | charged `m` times per step instead of once |
| Training capacity used `unique` rather than `stored` parameters | understated OOM risk |
| Inference activations charged at KV precision | W4A16 with fp8 KV understated by 2× |
| `maxContext`/`maxBatch` held activations constant | advertised limits that do not fit; `maxBatch` could be `Infinity` |
| `attentionPairs()` unvalidated | `q > s` produced a **negative** pair count |
| `algbw` and `busbw` sharing one type | the double-count was avoided by convention, not structurally |
| `stallTime` unreachable under an earliest-start scheduler | the `serialization` class could never fire |
| `meanContext` off by half a token | trivial, fixed |

Two of these are worth naming for what they say about the method. The **invented 1.1×** was
exactly the sin this document preaches against everywhere else, sitting unremarked in the code
that implements it. And the **DP overlap** was asserted in a comment while the dependency edge
made it impossible — a claim the code contradicted.

Removing the 1.1× moved training MFU from 36.3% to 41.8% and put the HFU/MFU ratio at 1.066
against the 1 + s/(18h) = 1.056 the corrected Korthikanti formula predicts. The fix made the
numbers *more* consistent with theory, which is the sign it was a correction rather than a change.

**Rejected:** the review read `activations × p` for pipeline capacity as contradicting the
`(1 + (p−1)/(p·m))` factor quoted above. Both are in Korthikanti: the interleaved-schedule factor
*and* the statement that the first stage must hold `p` microbatches to stay pressurised. The code
implements the latter, which is the conservative one for 1F1B and the right choice for a check
that decides OOM.

### Items flagged as unverified

| Item | Status |
|---|---|
| Roofline CACM 2009 ceiling wording and ordering rule | paraphrase-verified only (ACM DL 403) |
| Thakur/Rabenseifner/Gropp α-β formulas | standard forms, not transcribed (mirrors 403) |
| Cardwell & Song 2019 equations | second-hand via Checconi et al. |
| MoE all-to-all volume | derived; no canonical primary source found |
| "18 / 20 bytes per parameter" | framework configurations, not paper-sourced |
| GH200 specifications | NVIDIA landing page has no spec table; datasheet URLs 404 |
| MPT-7B MFU | no published figure located |
| NCCL all-to-all busbw factor `(n−1)/n` | in the summary table, not derived in the doc body |

None is load-bearing for a shipped number.

---

## Sources

**Roofline** — Williams, Waterman, Patterson, CACM 52(4), 2009 · Yang, Kurth, Williams,
*Hierarchical Roofline for GPUs*, CCPE 2020 · Checconi, Tithi, Petrini, *Ridgeline*,
arXiv:2209.01368 · Cardwell & Song, HPC Asia 2019.

**Transformer FLOPs** — Kaplan et al., arXiv:2001.08361 · Shoeybi et al., arXiv:1909.08053 ·
Narayanan et al., arXiv:2104.04473 · Korthikanti et al., arXiv:2205.05198 · NVIDIA Megatron-LM
`num_floating_point_operations`.

**MFU/HFU** — Chowdhery et al. (PaLM), arXiv:2204.02311 · Dubey et al. (Llama 3),
arXiv:2407.21783.

**Inference** — NVIDIA *LLM Benchmarking: Fundamental Concepts* and *Mastering LLM Techniques:
Inference Optimization* · vLLM metrics design doc and `benchmarks/serve.py` · Pope et al.,
arXiv:2211.05102 · Google, *How To Scale Your Model* · Ainslie et al. (GQA), arXiv:2305.13245 ·
Shazeer (MQA), arXiv:1911.02150 · DeepSeek-V2, arXiv:2405.04434 · DeepSeek-V3, arXiv:2412.19437
· Mistral 7B, arXiv:2310.06825.

**Memory and collectives** — Rajbhandari et al. (ZeRO), arXiv:1910.02054 · NVIDIA nccl-tests
`doc/PERFORMANCE.md` · Thakur, Rabenseifner, Gropp, IJHPCA 2005 · Fedus, Zoph, Shazeer (Switch),
arXiv:2101.03961.

**Hardware** — NVIDIA H100 product page · Hopper H100 architecture whitepaper v1.02 · DGX H100
user guide and datasheet · H100 NVL product brief.

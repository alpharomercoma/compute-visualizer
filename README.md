# Compute Visualizer

**Why is this LLM workload slow?** Pick a model, pick H100s, set batch and context and
sharding, and the tool names what is throttling you — and what would actually help.

Live: **[alpharomercoma.github.io/compute-visualizer](https://alpharomercoma.github.io/compute-visualizer/)**

> The mathematics matters more than the picture. The numeric core was written and
> adversarially reviewed *before* any interface existed. Its full derivation, with every
> constant's provenance, is in [`docs/MODEL.md`](docs/MODEL.md).

## Why

Every LLM performance tool I could find classifies a workload as compute-bound or
memory-bound. That is not enough to act on. "Communication-bound" hides two different
problems with two opposite fixes, and the tools that mention communication at all collapse
them together.

So this classifies into **five** states:

| Class | What it means | What actually helps |
|---|---|---|
| Compute-bound | tensor cores saturated | lower precision, better kernels |
| HBM-bandwidth-bound | streaming weights and KV | quantize, raise batch, better reuse |
| Network-**bandwidth**-bound | large collectives on the wire | less tensor parallelism, resharding, faster fabric |
| Network-**latency**-bound | tiny collectives, α-dominated | fewer collectives, CUDA graphs, less TP |
| Dependency-bound | critical path is waiting, not working | restructure the schedule |

The distinction earns its keep immediately. Qwen3 0.6B at batch 1 across 8 GPUs is
*latency*-bound: its per-layer all-reduce is 2 KiB, and transfer is **279× faster than the
collective's startup cost**. A three-way model calls that "communication-bound" and sends
you to buy faster interconnect, which would change nothing at all.

## What it gets right that a spreadsheet does not

**It rooflines per kernel class, then sums.** Computing one aggregate arithmetic intensity
for a whole step and taking the argmax is a category error. Within one kernel, memory and
compute genuinely overlap, so `max()` is right there — but across kernels, Jensen's
inequality says `Σ max(tᶜ, tᵐ) ≥ max(Σ tᶜ, Σ tᵐ)`. Averaging a compute-bound MLP with a
memory-bound attention parks the result at the ridge point and predicts a utilisation that
never happens.

**It schedules a task DAG instead of assuming an overlap policy.** No global rule ("step
time is max" / "is sum") is valid, so the model states dependencies and resources and lets
the schedule fall out. Tensor-parallel decode collectives land on the critical path because
nothing independent exists to hide them; data-parallel gradient reduction overlaps backward
and only its tail is exposed. Same scheduler, opposite outcomes, neither hardcoded.

**It checks feasibility before it reports a time.** A configuration that OOMs gets a
verdict, not a fast number.

**It separates vendor specifications from fitted priors, visibly.** NVIDIA's numbers and our
guesses never look alike in the interface.

**It draws no network ceiling on the roofline.** At `FLOPs/t_net` that line is not a roof:
add dummy FLOPs to a kernel and the "ceiling" rises. Communication is reported by the time
breakdown, where it can be stated honestly. (Ridgeline, arXiv:2209.01368, does solve this
properly by re-choosing the axes — but it assumes bandwidth-dominated communication and so
systematically misplaces latency-bound points, which is a regime this tool exists to get
right.)

## Some things it will tell you

- **Decode plots as two points, not one.** Linear layers sit at `I ≈ 2b/w` and walk right as
  batch grows, crossing the H100 BF16 ridge at ~295. Attention sits at `2·n_q/(n_kv·w_kv)`
  and **never moves with batch**, because every sequence carries its own KV cache. That is
  why long-context decode never becomes compute-bound at any batch size, and why GQA, MLA
  and KV quantization are the only levers on it.
- **You cannot batch your way out of the network roof in tensor-parallel decode.** Network
  intensity is independent of batch size — both FLOPs and collective bytes scale with it.
- **Tensor parallelism has a hard scaling limit, and it is latency.** On Llama 3.1 8B at
  batch 1, memory time halves with each doubling of TP (5.18 → 2.59 → 1.29 → 0.64 ms) while
  collective latency stays pinned at 0.26 ms. Its share climbs 0% → 9% → 17% → 28% and the
  speedup decays. Interconnect *bandwidth* is never binding at any rank count.
- **Full activation recomputation raises HFU while lowering MFU.** Precisely the divergence
  PaLM used to argue MFU is the throughput-honest metric.

## How it was verified

Two independent models — OpenAI's `codex` and Google's `agy` — reviewed the formula
specification before any code existed, then the implementation, then recomputed the reference
scenarios from scratch. Disagreements were settled by explicit arithmetic, not by majority.

The spec was reviewed before any code existed, then the implementation against the spec. It
caught real errors at both gates — including in its own reviewers, and including two in code that
was already green and looked right:

- A reviewer's reference table put per-GPU decode KV traffic at 21.47 GB. Two independent
  calculations gave **2.68 GB** — it had quoted the cluster-wide figure as per-GPU. An 8×
  error that propagated into a 2× wrong step time.
- The spec applied the causal-mask ÷2 to decode. It does not hold there: the ÷2 is an average
  over a *full* prefill, and a decode query attends the entire context. Understated decode
  attention by 2×.
- The spec's headline result divided cluster FLOPs by per-rank bytes, inflating it by `t`.
- The H100's "80 GB" is 80 **GiB** — five HBM3 stacks of 16 GiB. Bandwidth is decimal,
  capacity is binary. Assuming decimal for both understates capacity by 7.4%, on the check
  that decides whether a run OOMs.
- Korthikanti et al. Eq. (8) is arithmetically inconsistent with its own derivation: it
  prints `s/(3h)` where the `+4Bs²h` per layer it just described gives `2s/(9h)`.
- **Selective activation recomputation was implemented as a blanket 1.1× across every kernel** —
  an invented constant, wrong at every `s/h`, sitting inside the project that refuses invented
  constants everywhere else. Replacing it with the attention-only term the paper specifies moved
  training MFU from 36.3% to 41.8% and put HFU/MFU at 1.066 against the 1.056 the corrected
  formula predicts. A fix that agrees with theory better than before is a correction, not a change.
- **Data-parallel gradient overlap was asserted in a comment while the dependency edge made it
  impossible**, then charged once per microbatch instead of once per optimizer step.

One reviewer finding was **rejected**: it flagged "NVSwitch gen3" as an Ampere-era error. The
Hopper whitepaper states fourth-generation NVLink over *third*-generation NVSwitch.

After the fixes, the two independent derivations agree to within 0.5% on every quantity of
the reference scenario. Full record in [`docs/MODEL.md`](docs/MODEL.md).

## Reference scenario

Llama 3.1 70B, BF16, 8×H100 SXM5, TP=8, batch 32, 2048 context, decode:

| | |
|---|---|
| Weights per GPU | 17.6 GB |
| KV read per step per GPU | 2.68 GB |
| Time per output token | 8.5 ms |
| Aggregate throughput | ~3,800 tok/s |
| MBU | 73% |
| MFU | 6.9% — **correct, not a fault** |
| Arithmetic intensity | 28 FLOP/byte, against a ridge of 295 |
| Verdict | HBM-bandwidth-bound |

## What it does not model

A browser cannot profile a machine. This is a first-order analytical model. Not modelled:
wave quantization and SM tail effects, cache reuse beyond a per-kernel bound, scheduler and
continuous-batching behaviour, clock and power throttling, paged-KV fragmentation, kernel
selection heuristics.

Every collective latency and efficiency constant is a **fitted prior, not a vendor
specification**, and is labelled as such in the interface. No credible published
`all_reduce_perf` table exists for a single 8×H100 node together with the NCCL version and
NVLS state needed to fit one, and none was invented. The calibration command is documented in
`docs/MODEL.md` so anyone with the hardware can replace them.

## Any model on the Hub

Search Hugging Face from the tool and it reads `config.json` directly — no proxy, no server, no
API key. Gated repositories fall back to an ungated mirror and the interface names the
substitution rather than quietly costing a different model. A bundled catalog covers the common
cases instantly and offline.

The parsing is where the traps are, and they are silent ones: `head_dim` is sometimes present and
*not* equal to `hidden/heads` (Qwen3-30B-A3B declares 128 where the derivation gives 64 — a 2× KV
error); the expert count is spelled `num_local_experts`, `num_experts` or `n_routed_experts`
depending on the family; and DeepSeek's MLA caches a low-rank latent rather than K and V, so the
standard formula overstates it by ~57×. Each is handled explicitly and guarded by a test.

## Replacing our guesses with your measurements

Every collective constant here is a fitted prior. Paste `all_reduce_perf` output from your own
machine and the tool refits α and the bandwidth term, shows how far off the shipped defaults were,
and feeds the result into every subsequent number:

```sh
./build/all_reduce_perf -b 1K -e 1G -f 2 -g 8 -n 100
```

If the fit says our α was 2× out, that is worth knowing about every other figure on the page.

## Build

```sh
npm install
npm test        # 96 tests, including the golden scenarios
npm run dev
npm run build
```

The numeric core in `src/core/` has **zero dependencies** — no React, no three.js, no DOM. It
is testable, reviewable as a standalone artifact, and portable. Delete the interface and the
physics still stands and still passes its tests.

Units and scopes are types rather than conventions: `PerGpu<Bytes>` and `Cluster<Bytes>` are
distinct, so the exact substitution that produced the 8× error above will not compile.

| Path | What lives there |
|---|---|
| `src/core/` | the model — units, hardware, kernels, collectives, capacity, scheduler, metrics |
| `src/hub/` | Hugging Face client (CORS-verified, no proxy, no server) |
| `src/viz/` | roofline and time breakdown, hand-rolled SVG |
| `src/scene/` | the 3D board, lazy-loaded |
| `docs/MODEL.md` | every formula, its source, and its status |

The tool loads in **76 KB gzipped**; three.js streams in behind it rather than gating first
paint.

## Related

This is the datacenter counterpart to [OpenWeights](https://github.com/alpharomercoma/openweights),
an Android app that answers the same question for a phone — including its
`FitEstimator`, which returns a *nullable* throughput estimate on the principle that an
invented number is worse than none. That rule governs here too.

## License

Apache 2.0.

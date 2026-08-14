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

## What it can show that a calculator cannot

Both of these fall out of building the model as a scheduler over a task DAG rather than as a
sum of independent terms, and of it being a pure function cheap enough to run hundreds of
times.

**The step, as time.** One layer drawn on two lanes — GPU and interconnect — with every
kernel and every collective at its real start and duration, and the critical path emphasised.
An all-reduce is not "13% of the step" floating somewhere; it sits between the output
projection and the MLP, and the gap it leaves on the GPU lane is visible. That is what makes
"tensor parallelism serialises decode" something you can see rather than something you are
told. Terms charged to the step but happening outside any one layer — pipeline hops, the
bubble, the exposed gradient reduction — are named in the caption with their share, because
the figure cannot draw them.

**Where the answer changes.** The bottleneck class swept across a whole axis, with the region
that does not fit hatched and every threshold bisected to the exact value. A single verdict is
a fact about today's configuration; the strip is a fact about the hardware. The sentence it
most often produces is the one no other tool in this space says:

> Limited by memory bandwidth at every batch size that fits — the last one is 94. Moving along
> this axis will not change the answer.

The strip is also a control: click or drag it and the page re-diagnoses.

**When nothing fits, it searches.** "Shard further, cut the batch, or quantize" is a list of
the four things the tool could have checked for you. Instead it evaluates nearby
configurations with the same analysis the page runs, ranked by how little they ask of you, and
offers the ones that work with their memory cost. Where no configuration works — a 70B in
mixed-precision Adam needs 16 bytes per parameter, which is 141 GiB on each of eight H100s
however you shard it — it names the models that do.

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

### And how the review method had to change

The maths gates worked. The interface gates did not, and the reason is worth recording:
**both reviewers were asked to judge a rendered screen, so both reviewed a picture rather than
a product.** They passed a build in which the only control that opened the settings — the one
route to every input — was rendered inside the success branch. Pressing the app's own
"Training" toggle made the default configuration infeasible, that branch unmounted, and the
page became a title, a "this will not fit" card and a footer, with no control anywhere on it.
The whole state lives in the URL, so reloading did not escape it either. Nobody had clicked
anything.

Reviewing transitions instead of frames — enumerate the reachable states, ask which have no
exit, ask what each control does when the result is absent — found in one pass:

- a plain `#hw=h100-x1` link carrying the default tensor parallelism of 8 onto a single GPU,
  where the capacity model divided the weights by eight and reported that a 70B fits
  comfortably. A confidently wrong answer from a URL, and worse than the dead end, because
  nothing about it looks broken;
- the step share divided by the wrong denominator: `schedule` covers one pipeline stage in
  inference and one microbatch in training, so training announced "memory bandwidth is the
  limit" beside a share of 5%, while the bar next to it normalised to its own sum;
- the limiting class chosen by the largest single slice rather than the largest class, so
  30% collectives plus 30% stage transfer lost to 40% compute;
- sweep thresholds quoted as actionable targets while being whichever of forty log-spaced
  samples happened to change first — out by 10% on the front page;
- a number field that could not be cleared with backspace, because an empty string parses to
  zero and zero failed the range check;
- four controls that changed nothing in the phase they were shown in, and two state fields
  serialized into the URL with no reader at all.

The rule this leaves behind: **controls belong to the page frame, never to the result.** Any
control rendered inside a conditional is a control that disappears exactly when something has
gone wrong.

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

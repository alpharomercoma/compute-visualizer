/**
 * What to do when it does not fit.
 *
 * "Shard further, cut the batch or context, or quantize" is a list of the four
 * things this tool could have checked for you and didn't. The model is a pure
 * function of the configuration, so the configurations near yours that DO fit
 * are computable, and at the moment the answer is "no" that is the only thing
 * worth showing.
 *
 * Candidates are ranked by how little they ask of you, not by how fast they run.
 * Someone who chose a batch of 64 wants the smallest edit that makes their
 * workload go, not the edit with the best throughput — and a quantized model is
 * a different model, so proposing INT4 ahead of a reshard would be answering a
 * question nobody asked.
 *
 * Combinations are allowed, in cost order, because for training they are the
 * only thing that works. A 70B with mixed-precision Adam needs 16 bytes per
 * parameter: on eight H100s that is 141 GiB each however perfectly you sharded
 * it, so no single change rescues it and a search restricted to single changes
 * reports "nothing reachable" for a configuration that has a well-known fix.
 * Sharding, ZeRO stage, optimizer width and recomputation are applied together
 * in practice; refusing to consider them together was a rule about tidiness, not
 * about the hardware.
 *
 * Feasibility is decided by running the same analysis the page runs, never a
 * cheaper approximation. A rescue offering a configuration the verdict then
 * rejects would be worse than offering nothing.
 */

import type { AppState } from './state'
import type { Dtype, Topology } from '../core/hardware'
import { type TransformerSpec, fromHuggingFaceConfig } from '../core/modelspec'
import { H100_EFFICIENCY, H100_OVERHEAD } from '../core/efficiency'
import { analyseInference } from '../core/inference'
import { analyseTraining } from '../core/training'
import { countParameters } from '../core/params'
import { shardingSplits } from '../core/search'
import { GiB } from '../core/units'
import { PRESETS } from '../data/presets'

export interface Rescue {
  /** What changes, in the user's words. */
  readonly label: string
  /** The fields to overwrite. */
  readonly patch: Partial<AppState>
  /** Peak per-GPU memory the change leaves in use. */
  readonly usedBytes: number
  readonly availableBytes: number
}

/** One edit to the configuration: what it says, what it sets, what it costs. */
interface Move {
  readonly label: string
  readonly patch: Partial<AppState>
  /** How much this asks of the person who made the original choice. */
  readonly cost: number
}

/** Doing nothing to this group, which is always the cheapest option. */
const KEEP: Move = { label: '', patch: {}, cost: 0 }

const DTYPE_WIDTH: Record<Dtype, number> = {
  fp32: 4,
  bf16: 2,
  fp16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
}

const splitLabel = (t: number, p: number, d: number) => `TP ${t} · PP ${p} · DP ${d}`

function shardingMoves(state: AppState, gpus: number, allowFewerReplicas: boolean): Move[] {
  const moves: Move[] = [KEEP]

  for (const split of shardingSplits(gpus)) {
    if (
      split.tensor === state.tensorParallel &&
      split.pipeline === state.pipelineParallel &&
      split.data === state.dataParallel
    ) {
      continue
    }
    // For inference, more data parallelism means more replicas of the same
    // weights and strictly more memory per GPU — offering it as a fix for an
    // OOM would be a lie the search would then have to explain. For training,
    // data parallelism is the dimension ZeRO shards along, so it is the fix.
    if (!allowFewerReplicas && split.data > state.dataParallel) continue

    moves.push({
      label: splitLabel(split.tensor, split.pipeline, split.data),
      patch: {
        tensorParallel: split.tensor,
        pipelineParallel: split.pipeline,
        dataParallel: split.data,
      },
      cost: 1 + (split.pipeline > 1 ? 0.5 : 0),
    })
  }

  return moves
}

function weightMoves(state: AppState): Move[] {
  const moves: Move[] = [KEEP]
  for (const dtype of ['fp8', 'int8', 'int4'] as const) {
    // Only ever proposed as a REDUCTION. Narrower costs more: losing half the
    // bits is a smaller thing to ask than losing three quarters of them.
    if (DTYPE_WIDTH[dtype] >= DTYPE_WIDTH[state.weightDtype]) continue
    moves.push({
      label: `${dtype.toUpperCase()} weights`,
      patch: { weightDtype: dtype },
      cost: 2 + 1 / DTYPE_WIDTH[dtype],
    })
  }
  return moves
}

function inferenceGroups(state: AppState, gpus: number): Move[][] {
  const kv: Move[] = [KEEP]
  if (state.kvDtype === 'bf16') {
    kv.push({ label: 'FP8 KV cache', patch: { kvDtype: 'fp8' }, cost: 1.5 })
  }

  const batch: Move[] = [KEEP]
  for (const divisor of [2, 4, 8]) {
    const next = Math.floor(state.batch / divisor)
    if (next >= 1 && next !== state.batch) {
      batch.push({ label: `batch ${next}`, patch: { batch: next }, cost: 3 + divisor / 10 })
    }
  }

  const context: Move[] = [KEEP]
  for (const divisor of [2, 4]) {
    const next = Math.floor(state.promptLength / divisor / 128) * 128
    if (next >= 128 && next !== state.promptLength) {
      context.push({
        label: `${next}-token context`,
        patch: { promptLength: next },
        cost: 3.5 + divisor / 10,
      })
    }
  }

  return [shardingMoves(state, gpus, false), weightMoves(state), kv, batch, context]
}

function trainingGroups(state: AppState, gpus: number): Move[][] {
  const zero: Move[] = [KEEP]
  for (const stage of [1, 2, 3] as const) {
    if (stage <= state.zeroStage) continue
    zero.push({ label: `ZeRO-${stage}`, patch: { zeroStage: stage }, cost: 0.5 + stage / 10 })
  }

  const optimizer: Move[] =
    state.optimizer === 'adam-8bit'
      ? [KEEP]
      : [KEEP, { label: '8-bit optimizer', patch: { optimizer: 'adam-8bit' }, cost: 1.5 }]

  const recompute: Move[] =
    state.recompute === 'full'
      ? [KEEP]
      : [
          KEEP,
          {
            label: 'full recomputation',
            patch: { recompute: 'full' },
            // Cheap in memory, expensive in time: it buys the fit by doing the
            // forward pass twice, and saying so is part of the offer.
            cost: 2,
          },
        ]

  const sequence: Move[] = [KEEP]
  for (const divisor of [2, 4]) {
    const next = Math.floor(state.sequenceLength / divisor / 512) * 512
    if (next >= 512 && next !== state.sequenceLength) {
      sequence.push({
        label: `${next}-token sequences`,
        patch: { sequenceLength: next },
        cost: 3 + divisor / 10,
      })
    }
  }

  return [shardingMoves(state, gpus, true), zero, optimizer, recompute, sequence]
}

/** Every combination of one move per group, cheapest total first. */
function combinations(groups: Move[][]): Move[] {
  let out: Move[] = [KEEP]

  for (const group of groups) {
    const next: Move[] = []
    for (const base of out) {
      for (const move of group) {
        next.push({
          label: [base.label, move.label].filter(Boolean).join(', '),
          patch: { ...base.patch, ...move.patch },
          cost: base.cost + move.cost,
        })
      }
    }
    out = next
  }

  return out.filter((c) => c.cost > 0).sort((a, b) => a.cost - b.cost)
}

/**
 * The nearest configurations that fit, cheapest edit first.
 *
 * Bounded rather than exhaustive: the product of every group is a few hundred
 * candidates and each costs a full analysis, so the search stops after
 * `maxEvaluations`. Because the list is sorted by cost before evaluation, that
 * bound trades away only the expensive rescues nobody would accept anyway.
 */
export function nearestFeasible(
  state: AppState,
  topology: Topology,
  spec: TransformerSpec,
  limit = 3,
  maxEvaluations = 80,
): Rescue[] {
  const gpus = topology.gpus
  const candidates = combinations(
    state.phase === 'inference' ? inferenceGroups(state, gpus) : trainingGroups(state, gpus),
  )

  const found: Rescue[] = []
  let evaluated = 0

  for (const candidate of candidates) {
    if (evaluated >= maxEvaluations || found.length >= limit) break
    evaluated++

    const result = run({ ...state, ...candidate.patch }, topology, spec)
    if (!result?.feasible) continue

    // A cheaper rescue already found makes every superset of it redundant:
    // offering "FP8 weights" and then "FP8 weights, batch 16" is one option
    // dressed as two.
    const redundant = found.some((f) =>
      Object.entries(f.patch).every(
        ([key, value]) => candidate.patch[key as keyof AppState] === value,
      ),
    )
    if (redundant) continue

    found.push({
      label: candidate.label,
      patch: candidate.patch,
      usedBytes: result.capacity.perGpu.total,
      availableBytes: result.capacity.available,
    })
  }

  if (found.length === 0) found.push(...smallerModels(state, topology, spec, limit))
  return found
}

/**
 * The fallback: keep the setup, change the model.
 *
 * Some configurations have no fix. A 70B in mixed-precision Adam needs 16 bytes
 * per parameter — 141 GiB on each of eight H100s however it is sharded, and
 * still 84 GiB with an 8-bit optimizer and full recomputation against 79.6
 * available. Nothing on the sharding, precision or recomputation axes reaches
 * it, and saying "try a smaller model" without naming one leaves the person
 * exactly where they started.
 *
 * Searched separately rather than as another group in the product, because a
 * dozen models multiplied by every other combination is thousands of analyses
 * to answer a question that only arises when everything else has failed.
 */
function smallerModels(
  state: AppState,
  topology: Topology,
  spec: TransformerSpec,
  limit: number,
): Rescue[] {
  const size = (candidate: TransformerSpec) => countParameters(candidate).stored
  const current = size(spec)

  const smaller = PRESETS.map((preset) => {
    try {
      return {
        preset,
        spec: fromHuggingFaceConfig(preset.config, { id: preset.id, name: preset.label }),
      }
    } catch {
      return undefined
    }
  })
    .filter((entry): entry is NonNullable<typeof entry> => entry !== undefined)
    .filter((entry) => size(entry.spec) < current * 0.95)
    // Largest first: the nearest thing to what was asked for, not the smallest
    // thing in the catalog.
    .sort((a, b) => size(b.spec) - size(a.spec))

  const out: Rescue[] = []
  for (const entry of smaller) {
    if (out.length >= limit) break
    const next = { ...state, modelId: entry.preset.id }
    const result = run(next, topology, entry.spec)
    if (!result?.feasible) continue
    out.push({
      label: `${entry.preset.label} instead`,
      patch: { modelId: entry.preset.id },
      usedBytes: result.capacity.perGpu.total,
      availableBytes: result.capacity.available,
    })
  }
  return out
}

/** One configuration, analysed exactly as the page would analyse it. */
function run(state: AppState, topology: Topology, spec: TransformerSpec) {
  const parallelism = {
    tensor: state.tensorParallel,
    pipeline: state.pipelineParallel,
    data: state.dataParallel,
  }

  try {
    return state.phase === 'inference'
      ? analyseInference({
          spec,
          topology,
          parallelism,
          batch: state.batch,
          promptLength: state.promptLength,
          generateLength: state.generateLength,
          weightDtype: state.weightDtype,
          kvDtype: state.kvDtype,
          activationDtype: 'bf16',
          efficiency: H100_EFFICIENCY,
          overhead: H100_OVERHEAD,
          cudaGraph: state.cudaGraph,
          flashAttention: state.flashAttention,
        })
      : analyseTraining({
          spec,
          topology,
          parallelism,
          microBatch: 1,
          sequenceLength: state.sequenceLength,
          gradientAccumulation: state.gradientAccumulation,
          weightDtype: state.weightDtype,
          activationDtype: 'bf16',
          optimizer: state.optimizer,
          zeroStage: state.zeroStage,
          sequenceParallel: state.sequenceParallel,
          selectiveRecompute: state.recompute === 'selective',
          fullRecompute: state.recompute === 'full',
          efficiency: H100_EFFICIENCY,
          overhead: H100_OVERHEAD,
        })
  } catch {
    return undefined
  }
}

export const gib = (b: number) => `${(b / GiB(1)).toFixed(1)} GiB`

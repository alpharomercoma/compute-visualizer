import type { AppState } from '../app/state'
import type { TransformerSpec } from '../core/modelspec'
import type { Topology } from '../core/hardware'
import type { analyseInference } from '../core/inference'
import type { analyseTraining } from '../core/training'
import { BOTTLENECK_REMEDY } from '../core/schedule'
import { countParameters, formatCount } from '../core/params'
import { explainVerdict } from '../core/capacity'
import { GiB } from '../core/units'

/**
 * The answer.
 *
 * The verdict comes first and in words, because "memory-bandwidth-bound, here is
 * what would help" is the thing someone came for; the numbers are the evidence
 * for it. Everything derived from a fitted prior rather than a vendor
 * specification is marked, so nobody mistakes a guess for a datasheet.
 */

export interface ReadoutProps {
  readonly state: AppState
  readonly spec: TransformerSpec | undefined
  readonly topology: Topology
  readonly inference: ReturnType<typeof analyseInference> | undefined
  readonly training: ReturnType<typeof analyseTraining> | undefined
}

const ms = (value: number) =>
  value >= 1 ? `${value.toFixed(2)} s` : `${(value * 1000).toFixed(2)} ms`

const pct = (fraction: number) => `${(fraction * 100).toFixed(1)}%`

const gib = (bytes: number) => `${(bytes / GiB(1)).toFixed(1)} GiB`

export function Readout({ state, spec, topology, inference, training }: ReadoutProps) {
  const isInference = state.phase === 'inference'
  const result = isInference ? inference : training
  const capacity = result?.capacity

  if (!spec || !result) {
    return (
      <aside className="panel">
        <p className="note">Choose a model to begin.</p>
      </aside>
    )
  }

  if (!result.feasible) {
    return (
      <aside className="panel">
        <div className="verdict">
          <h2>Will not fit</h2>
          <p>{capacity ? explainVerdict(capacity) : ''}</p>
        </div>
        <div className="section-head">Memory per GPU</div>
        <MemoryRows capacity={capacity!} />
        <p className="note" style={{ marginTop: 10 }}>
          No step time is shown, because this configuration cannot run. Reduce the batch or context,
          shard further, or quantize.
        </p>
      </aside>
    )
  }

  const bottleneck = isInference ? inference!.decode!.bottleneck : training!.bottleneck!
  const remedy = BOTTLENECK_REMEDY[bottleneck]
  const counted = countParameters(spec)

  return (
    <aside className="panel">
      <div className="verdict">
        <h2>{remedy.label}</h2>
        <p>{remedy.remedy}</p>
      </div>

      {isInference ? (
        <>
          <div className="section-head">Latency</div>
          <dl style={{ margin: 0 }}>
            <Metric label="TTFT" value={ms(inference!.timeToFirstToken!)} />
            <Metric label="Per output token" value={ms(inference!.timePerOutputToken!)} />
            <Metric label="End to end" value={ms(inference!.endToEndLatency!)} />
          </dl>

          <div className="section-head" style={{ marginTop: 14 }}>
            Throughput
          </div>
          <dl style={{ margin: 0 }}>
            <Metric
              label="Per user"
              value={`${inference!.tokensPerSecondPerUser!.toFixed(0)}`}
              unit="tok/s"
            />
            <Metric
              label="Aggregate"
              value={`${Math.round(inference!.aggregateTokensPerSecond!).toLocaleString()}`}
              unit="tok/s"
            />
            <Metric
              label="Cost"
              value={`$${costPerMillion(
                inference!.aggregateTokensPerSecond!,
                state.gpuHourCost,
                topology.gpus,
              ).toFixed(2)}`}
              unit="/ M tok"
            />
          </dl>

          <div className="section-head" style={{ marginTop: 14 }}>
            Utilisation
          </div>
          <dl style={{ margin: 0 }}>
            <Metric label="MBU" value={pct(inference!.decodeMbu!)} />
            <Metric label="MFU (decode)" value={pct(inference!.decodeMfu!)} />
            <Metric label="MFU (prefill)" value={pct(inference!.prefillMfu!)} />
          </dl>
          <p className="note" style={{ marginTop: 6 }}>
            Low decode MFU is the correct answer, not a fault. Decode streams weights and KV while
            doing almost no arithmetic, so bandwidth utilisation is the metric that carries
            information.
          </p>
        </>
      ) : (
        <>
          <div className="section-head">Step</div>
          <dl style={{ margin: 0 }}>
            <Metric label="Step time" value={ms(training!.stepTime!)} />
            <Metric
              label="Tokens / step"
              value={training!.globalBatchTokens!.toLocaleString()}
            />
            <Metric
              label="Throughput"
              value={`${(training!.tokensPerSecond! / 1000).toFixed(1)}k`}
              unit="tok/s"
            />
            <Metric
              label="Per day"
              value={`${(training!.tokensPerDay! / 1e9).toFixed(2)}B`}
              unit="tok"
            />
          </dl>

          <div className="section-head" style={{ marginTop: 14 }}>
            Utilisation
          </div>
          <dl style={{ margin: 0 }}>
            <Metric label="MFU" value={pct(training!.mfu!)} />
            <Metric label="HFU" value={pct(training!.hfu!)} />
            {training!.bubbleFraction! > 0 && (
              <Metric label="Pipeline bubble" value={pct(training!.bubbleFraction!)} />
            )}
          </dl>
          <p className="note" style={{ marginTop: 6 }}>
            HFU counts what the GPUs executed; MFU counts what the model required. Recomputation
            raises the first and lowers the second, which is why MFU is the throughput-honest one.
          </p>
        </>
      )}

      <div className="section-head" style={{ marginTop: 14 }}>
        Memory per GPU
      </div>
      <MemoryRows capacity={capacity!} />

      <div className="section-head" style={{ marginTop: 14 }}>
        Model
      </div>
      <dl style={{ margin: 0 }}>
        <Metric label="Parameters" value={formatCount(counted.unique)} />
        {counted.active !== counted.unique && (
          <Metric label="Active / token" value={formatCount(counted.active)} />
        )}
        {counted.stored !== counted.unique && (
          <Metric label="Stored" value={formatCount(counted.stored)} />
        )}
        <Metric label="Attention" value={spec.attention.kind.toUpperCase()} />
      </dl>

      <p className="note" style={{ marginTop: 12, display: 'flex', gap: 6, alignItems: 'center' }}>
        <span className="tag prior">prior</span>
        Collective latency and efficiency factors are fitted, not vendor specifications.
      </p>
    </aside>
  )
}

function MemoryRows({ capacity }: { capacity: NonNullable<ReturnType<typeof analyseInference>['capacity']> }) {
  const { perGpu } = capacity
  return (
    <dl style={{ margin: 0 }}>
      <Metric label="Weights" value={gib(perGpu.weights)} />
      {perGpu.kvCache > 0 && <Metric label="KV cache" value={gib(perGpu.kvCache)} />}
      {perGpu.gradients > 0 && <Metric label="Gradients" value={gib(perGpu.gradients)} />}
      {perGpu.optimizerStates > 0 && (
        <Metric label="Optimizer" value={gib(perGpu.optimizerStates)} />
      )}
      <Metric label="Activations" value={gib(perGpu.activations)} />
      <Metric label="Reserve" value={gib(perGpu.reserve)} />
      <Metric label="Total" value={`${gib(perGpu.total)} of ${gib(capacity.available)}`} />
    </dl>
  )
}

function Metric({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="metric">
      <dt>{label}</dt>
      <dd>
        {value}
        {unit && <span className="unit">{unit}</span>}
      </dd>
    </div>
  )
}

/**
 * Dollars per million output tokens.
 *
 * The number most people actually want, and it is only a division once
 * throughput exists. Cost is charged for every GPU in the deployment, whether or
 * not the model uses them well.
 */
function costPerMillion(tokensPerSecond: number, dollarsPerGpuHour: number, gpus: number): number {
  if (!(tokensPerSecond > 0)) return Number.POSITIVE_INFINITY
  const tokensPerHour = tokensPerSecond * 3600
  return (dollarsPerGpuHour * gpus * 1e6) / tokensPerHour
}

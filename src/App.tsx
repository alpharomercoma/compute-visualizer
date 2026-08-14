import { Suspense, lazy, useEffect, useMemo, useState, type CSSProperties } from 'react'
import { type AppState, readStateFromLocation, writeStateToLocation } from './app/state'
import { useHubModel } from './hub/useHubModel'
import type { CollectiveProfile } from './core/collectives'
import { DGX_H100, DTYPE_BYTES, SINGLE_H100, type Topology, ridgePoint } from './core/hardware'
import { H100_EFFICIENCY, H100_OVERHEAD } from './core/efficiency'
import { type InferenceConfig, analyseInference, computeBoundBatch } from './core/inference'
import { analyseTraining } from './core/training'
import type { BottleneckClass } from './core/schedule'
import { decodeAsymptotes, operatingPoint, roofline } from './core/roofline'
import { explainVerdict } from './core/capacity'
import { GiB } from './core/units'
import { Roofline } from './viz/Roofline'
import { TimeBreakdown } from './viz/TimeBreakdown'
import { HardwareBoard } from './viz/HardwareBoard'
import { Verdict } from './ui/Verdict'
import { Disclosure } from './ui/Disclosure'
import { ScenarioBar, Settings } from './ui/Scenario'
import { ModelPicker } from './ui/ModelPicker'
import { Advanced } from './ui/Advanced'
import { Collectives } from './ui/Collectives'
import { useTheme } from './ui/theme'

// Only loaded if someone asks for it. The flat board is the default view, so
// three.js is never on the critical path.
const Scene = lazy(() => import('./scene/Scene'))

const TOPOLOGIES: Record<string, Topology> = {
  'h100-x1': SINGLE_H100,
  'h100-x8': DGX_H100,
}

const ACCENT: Record<BottleneckClass, string> = {
  compute: 'var(--c-compute)',
  'memory-bandwidth': 'var(--c-memory)',
  'network-bandwidth': 'var(--c-net-bw)',
  'network-latency': 'var(--c-net-lat)',
  overhead: 'var(--c-overhead)',
  serialization: 'var(--c-serial)',
}

const gib = (b: number) => `${(b / GiB(1)).toFixed(1)} GiB`
const ms = (s: number) => (s >= 1 ? `${s.toFixed(2)} s` : `${(s * 1000).toFixed(1)} ms`)
const pct = (f: number) => `${(f * 100).toFixed(1)}%`

/** Grouped with commas regardless of the visitor's locale. */
const grouped = (n: number) => Math.round(n).toLocaleString('en-US')

export default function App() {
  const [state, setState] = useState<AppState>(readStateFromLocation)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [showScene, setShowScene] = useState(false)
  const { choice, setChoice } = useTheme()

  useEffect(() => {
    writeStateToLocation(state)
  }, [state])

  useEffect(() => {
    const onHashChange = () => setState(readStateFromLocation())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const topology = TOPOLOGIES[state.topologyId] ?? DGX_H100
  const hub = useHubModel(state.modelId)
  const spec = hub.model?.spec
  const [calibrated, setCalibrated] = useState<readonly CollectiveProfile[]>()

  const parallelism = {
    tensor: state.tensorParallel,
    pipeline: state.pipelineParallel,
    data: state.dataParallel,
  }

  const inferenceConfig = useMemo(
    () =>
      spec
        ? ({
            spec,
            topology,
            parallelism,
            batch: state.batch,
            promptLength: state.promptLength,
            generateLength: state.generateLength,
            weightDtype: state.weightDtype,
            kvDtype: state.kvDtype,
            activationDtype: 'bf16' as const,
            efficiency: H100_EFFICIENCY,
            overhead: H100_OVERHEAD,
            cudaGraph: state.cudaGraph,
            flashAttention: state.flashAttention,
            ...(calibrated ? { collectiveProfiles: calibrated } : {}),
          } satisfies InferenceConfig)
        : undefined,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [spec, topology, state, calibrated],
  )

  const inference = useMemo(() => {
    if (!inferenceConfig || state.phase !== 'inference') return undefined
    return analyseInference(inferenceConfig)
  }, [inferenceConfig, state.phase])

  const training = useMemo(() => {
    if (!spec || state.phase !== 'training') return undefined
    return analyseTraining({
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
      ...(calibrated ? { collectiveProfiles: calibrated } : {}),
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec, topology, state, calibrated])

  const isInference = state.phase === 'inference'
  const capacity = isInference ? inference?.capacity : training?.capacity
  const feasible = isInference ? inference?.feasible : training?.feasible
  const bottleneck = isInference ? inference?.decode?.bottleneck : training?.bottleneck
  const schedule = isInference ? inference?.decode?.schedule : training?.schedule
  const stepSeconds = isInference
    ? (inference?.timePerOutputToken ?? 0)
    : (training?.stepTime ?? 0)

  const accent = bottleneck ? ACCENT[bottleneck] : 'var(--c-memory)'

  const roof = useMemo(
    () => roofline(topology.accelerator, 'bf16', H100_EFFICIENCY),
    [topology],
  )

  const points = useMemo(() => {
    if (!isInference || !inference?.decode) return []
    return inference.decode.classes
      .map((entry) =>
        operatingPoint(
          {
            name: entry.label,
            kernelClass: entry.kernelClass as never,
            flops: entry.flops,
            hbmBytes: entry.hbmBytes,
            isGather: false,
          },
          entry.seconds,
          roof,
          entry.label,
        ),
      )
      .filter((p): p is NonNullable<typeof p> => p !== undefined)
  }, [inference, isInference, roof])

  const occupancy = capacity ? capacity.perGpu.total / capacity.available : 0
  const linkActivity =
    schedule && stepSeconds > 0
      ? Math.min(
          1,
          (schedule.breakdown['network-bandwidth'] + schedule.breakdown['network-latency']) /
            stepSeconds,
        )
      : 0

  const worldSize = parallelism.tensor * parallelism.pipeline * parallelism.data

  return (
    <div style={{ '--accent': accent } as CSSProperties}>
      <div className="shell">
        <header className="topbar">
          <h1>Compute Visualizer</h1>
          <span className="spacer" />
          <div className="segmented" aria-label="Colour theme">
            {(['light', 'system', 'dark'] as const).map((option) => (
              <button
                key={option}
                type="button"
                aria-pressed={choice === option}
                onClick={() => setChoice(option)}
                style={{ padding: '0 0.75rem' }}
              >
                {option === 'light' ? 'Light' : option === 'dark' ? 'Dark' : 'Auto'}
              </button>
            ))}
          </div>
        </header>

        <main>
          <p className="muted" style={{ margin: '1.25rem 0' }}>
            What is throttling this workload, and what would help.
          </p>

          {hub.error && (
            <section className="panel" style={{ marginBottom: 'var(--gap)' }}>
              <p style={{ margin: 0 }}>{hub.error}</p>
            </section>
          )}

          {feasible === false && capacity && (
            <section className="panel verdict" style={{ borderTopColor: 'var(--c-net-lat)' }}>
              <div>
                <h2>This will not fit</h2>
                <p className="cause">{explainVerdict(capacity)}</p>
                <p className="lever">
                  <strong>What helps:</strong> shard further, cut the batch or context, or quantize.
                </p>
              </div>
            </section>
          )}

          {feasible && bottleneck && schedule && (
            <>
              <Verdict
                bottleneck={bottleneck}
                schedule={schedule}
                stepSeconds={stepSeconds}
                phase={state.phase}
              />

              <section className="panel" style={{ marginBottom: 'var(--gap)' }}>
                <h2>Where the step goes</h2>
                <TimeBreakdown schedule={schedule} totalSeconds={stepSeconds} />
                <p className="small muted" style={{ marginTop: '0.75rem' }}>
                  Hardware figures are NVIDIA specifications. Collective latency and efficiency
                  factors are <span className="tag">fitted priors</span> — plausible guesses, not
                  measurements from your machine.
                </p>
              </section>


              <div className="figures">
                {isInference ? (
                  <>
                    <Figure
                      value={grouped(inference!.aggregateTokensPerSecond!)}
                      label="tokens / second"
                    />
                    <Figure value={ms(inference!.timePerOutputToken!)} label="per output token" />
                    <Figure
                      value={`$${costPerMillion(
                        inference!.aggregateTokensPerSecond!,
                        state.gpuHourCost,
                        worldSize,
                      ).toFixed(2)}`}
                      label="per million tokens"
                    />
                  </>
                ) : (
                  <>
                    <Figure
                      value={`${(training!.tokensPerSecond! / 1000).toFixed(1)}k`}
                      label="tokens / second"
                    />
                    <Figure value={ms(training!.stepTime!)} label="per step" />
                    <Figure value={pct(training!.mfu!)} label="model FLOPs utilisation" />
                  </>
                )}
              </div>

              <ScenarioBar
                state={state}
                topology={topology}
                resolved={hub.model}
                onEdit={() => setSettingsOpen(true)}
              />

              <Disclosure
                title="Why this diagnosis?"
                preview={
                  isInference
                    ? `Arithmetic intensity ${inference!.decode!.arithmeticIntensity.toFixed(
                        0,
                      )} against a ridge of ${ridgePoint(topology.accelerator, 'bf16').toFixed(0)}`
                    : `MFU ${pct(training!.mfu!)} against HFU ${pct(training!.hfu!)}`
                }
              >
                {isInference ? (
                  <>
                    <p>
                      A kernel is memory-bound below the machine&apos;s balance point and
                      compute-bound above it. Each dot is one kernel class over the whole step.
                    </p>
                    <Roofline
                      model={roof}
                      points={points}
                      showTheoretical={false}
                      {...(spec
                        ? {
                            asymptotes: decodeAsymptotes(
                              state.batch,
                              DTYPE_BYTES[state.weightDtype],
                              spec.attention.numQueryHeads,
                              spec.attention.numKvHeads,
                              DTYPE_BYTES[state.kvDtype],
                              roof,
                            ),
                          }
                        : {})}
                    />
                    <p className="small muted">
                      Decode&apos;s linear layers reach the ridge at batch{' '}
                      {computeBoundBatch(topology, 'bf16', state.weightDtype).toFixed(0)}. Attention
                      never reaches it at any batch size, because every sequence carries its own KV
                      cache.
                    </p>
                    {spec && state.tensorParallel > 1 && (
                      <>
                        <h2 style={{ marginTop: '1.5rem' }}>The interconnect</h2>
                        <Collectives
                          topology={topology}
                          tensorParallel={state.tensorParallel}
                          messageBytes={state.batch * spec.hiddenSize * 2}
                          collectivesPerStep={spec.numLayers * 2}
                        />
                      </>
                    )}
                  </>
                ) : (
                  <p>
                    HFU counts what the GPUs executed; MFU counts what the model required.
                    Recomputation raises the first and lowers the second, which is why MFU is the
                    throughput-honest one.
                  </p>
                )}
              </Disclosure>

              <Disclosure
                title="Will it fit in memory?"
                preview={`${gib(capacity!.perGpu.total)} of ${gib(capacity!.available)} per GPU`}
              >
                <HardwareBoard
                  gpus={topology.gpus}
                  accelerator={topology.accelerator.name}
                  occupancy={occupancy}
                  usedBytesLabel={gib(capacity!.perGpu.total)}
                  totalBytesLabel={gib(capacity!.available)}
                  linkActivity={linkActivity}
                  accent={accent}
                  infeasible={false}
                />
                <dl className="rows" style={{ marginTop: '1rem' }}>
                  <Row label="Weights" value={gib(capacity!.perGpu.weights)} />
                  {capacity!.perGpu.kvCache > 0 && (
                    <Row label="KV cache" value={gib(capacity!.perGpu.kvCache)} />
                  )}
                  {capacity!.perGpu.gradients > 0 && (
                    <Row label="Gradients" value={gib(capacity!.perGpu.gradients)} />
                  )}
                  {capacity!.perGpu.optimizerStates > 0 && (
                    <Row label="Optimizer" value={gib(capacity!.perGpu.optimizerStates)} />
                  )}
                  <Row label="Activations" value={gib(capacity!.perGpu.activations)} />
                  <Row label="Runtime reserve" value={gib(capacity!.perGpu.reserve)} />
                </dl>

                {!showScene ? (
                  <button
                    type="button"
                    className="ghost"
                    style={{ marginTop: '1rem' }}
                    onClick={() => setShowScene(true)}
                  >
                    Inspect in 3D
                  </button>
                ) : (
                  <div style={{ marginTop: '1rem' }}>
                    <Suspense
                      fallback={
                        <p className="small muted">Loading the hardware view…</p>
                      }
                    >
                      <Scene
                        gpus={topology.gpus}
                        accent={accent}
                        occupancy={occupancy}
                        onClose={() => setShowScene(false)}
                      />
                    </Suspense>
                  </div>
                )}
              </Disclosure>

              {isInference && inferenceConfig && (
                <Disclosure
                  title="Could a different split be faster?"
                  preview={`Currently TP ${state.tensorParallel} · PP ${state.pipelineParallel} · DP ${state.dataParallel}`}
                >
                  <Advanced
                    state={state}
                    onChange={setState}
                    baseConfig={inferenceConfig}
                    worldSize={topology.gpus}
                    onCalibrated={setCalibrated}
                    calibrated={calibrated !== undefined}
                  />
                </Disclosure>
              )}


            </>
          )}
        </main>

        <footer style={{ marginTop: '2rem' }}>
          <p className="small muted">
            <a href="https://github.com/alpharomercoma/compute-visualizer#readme">
              How this is calculated
            </a>{' '}
            ·{' '}
            <a href="https://github.com/alpharomercoma/compute-visualizer/blob/main/docs/MODEL.md">
              The model and its limits
            </a>
          </p>
        </footer>
      </div>

      <Settings
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        state={state}
        onChange={setState}
        topology={topology}
        modelSlot={
          <ModelPicker
            modelId={state.modelId}
            onChange={(repo) => setState({ ...state, modelId: repo })}
            resolved={hub.model}
            loading={hub.loading}
            error={hub.error}
          />
        }
      />
    </div>
  )
}

function Figure({ value, label }: { value: string; label: string }) {
  return (
    <div className="panel figure">
      <b>{value}</b>
      <span>{label}</span>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

function costPerMillion(tokensPerSecond: number, dollarsPerGpuHour: number, gpus: number): number {
  if (!(tokensPerSecond > 0)) return Number.POSITIVE_INFINITY
  return (dollarsPerGpuHour * gpus * 1e6) / (tokensPerSecond * 3600)
}

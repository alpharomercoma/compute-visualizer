import { Suspense, lazy, useEffect, useMemo, useState, type CSSProperties } from 'react'
import {
  type AppState,
  DEFAULT_STATE,
  readStateFromLocation,
  writeStateToLocation,
} from './app/state'
import { BOTTLENECK_NAME, accountInference, accountTraining } from './app/accounting'
import { nearestFeasible } from './app/rescue'
import { AXES_FOR, type Axis, sweep, sweepKey } from './app/sweep'
import { useHubModel } from './hub/useHubModel'
import type { CollectiveProfile } from './core/collectives'
import { DGX_H100, DTYPE_BYTES, SINGLE_H100, type Topology, ridgePoint } from './core/hardware'
import { H100_EFFICIENCY, H100_OVERHEAD } from './core/efficiency'
import {
  type InferenceConfig,
  analyseInference,
  computeBoundBatch,
  largestStage,
} from './core/inference'
import { analyseTraining } from './core/training'
import type { BottleneckClass } from './core/schedule'
import { decodeAsymptotes, operatingPoint, roofline } from './core/roofline'
import { explainVerdict } from './core/capacity'
import { rankSharding } from './core/search'
import { GiB } from './core/units'
import { Roofline } from './viz/Roofline'
import { TimeBreakdown } from './viz/TimeBreakdown'
import { StepTimeline } from './viz/StepTimeline'
import { Sweep } from './viz/Sweep'
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
  const schedule = isInference ? inference?.decode?.schedule : training?.schedule

  /**
   * One accounting for the whole page.
   *
   * Every share shown anywhere comes from here, so the headline percentage, the
   * bar and the timeline cannot disagree — which they did, by the pipeline
   * degree in inference and by the microbatch count in training.
   */
  const account = useMemo(() => {
    if (isInference) return inference ? accountInference(inference, state.pipelineParallel) : undefined
    return training ? accountTraining(training, state.gradientAccumulation) : undefined
  }, [isInference, inference, training, state.pipelineParallel, state.gradientAccumulation])

  const bottleneck = account?.bottleneck
  const accent = bottleneck ? ACCENT[bottleneck] : 'var(--c-memory)'

  // --- The sweep -----------------------------------------------------------
  // The primary axis always runs, because the verdict's lever quotes it. A
  // second axis only runs if someone asks for one.
  const axes = AXES_FOR(state.phase)
  const [axisKey, setAxisKey] = useState(axes[0]!.key)
  // Resolved against the axes of the CURRENT phase, and the selected state is
  // read back from the resolved axis. Comparing the buttons against the raw
  // `axisKey` left the segmented control with nothing selected after a phase
  // switch, because the retained key belongs to the other phase's axes.
  const axis: Axis = axes.find((a) => a.key === axisKey) ?? axes[0]!
  const primary = axes[0]!

  // Everything the curve depends on EXCEPT the field being swept, so dragging
  // along an axis does not recompute the curve you are dragging along.
  const primaryKey = sweepKey(state, primary, state.modelId, state.topologyId)
  const activeKey = sweepKey(state, axis, state.modelId, state.topologyId)

  const primarySweep = useMemo(() => {
    if (!spec) return undefined
    return sweep(state, spec, topology, primary, calibrated)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec, topology, calibrated, primaryKey])

  const activeSweep = useMemo(() => {
    if (!spec) return undefined
    if (axis.key === primary.key) return primarySweep
    return sweep(state, spec, topology, axis, calibrated)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec, topology, calibrated, axis.key, primarySweep, activeKey])

  const crossover = useMemo(() => {
    if (!primarySweep || !bottleneck) return undefined
    const current = state[primary.key]
    const boundary = primarySweep.boundaries.find(
      (b) => b.from !== 'infeasible' && b.to !== 'infeasible' && b.at > current,
    )
    if (!boundary) return undefined
    return {
      axis: primary.label,
      at: primary.format(boundary.at),
      // Named by the sweep, never assumed. The lever used to say "the tensor
      // cores become the limit" whatever the strip beneath it was showing.
      becomes:
        boundary.to === 'balanced'
          ? 'nothing in particular'
          : BOTTLENECK_NAME[boundary.to as BottleneckClass],
      intoBalance: boundary.to === 'balanced',
      reachable:
        primarySweep.capacityLimit === undefined || boundary.at <= primarySweep.capacityLimit,
    }
  }, [primarySweep, bottleneck, state, primary])

  // --- A better way to split the same GPUs ---------------------------------
  const ranked = useMemo(() => {
    if (!inferenceConfig || !isInference) return []
    return rankSharding(inferenceConfig, topology.gpus, 'throughput')
  }, [inferenceConfig, isInference, topology.gpus])

  const betterSplit = useMemo(() => {
    const best = ranked[0]
    if (!best?.feasible || !inference?.aggregateTokensPerSecond) return undefined
    const gain = (best.tokensPerSecond ?? 0) / inference.aggregateTokensPerSecond - 1
    if (gain < 0.05) return undefined
    return {
      label: `TP ${best.tensor} · PP ${best.pipeline} · DP ${best.data}`,
      gain,
      apply: () =>
        setState((prev) => ({
          ...prev,
          tensorParallel: best.tensor,
          pipelineParallel: best.pipeline,
          dataParallel: best.data,
        })),
    }
  }, [ranked, inference])

  // --- When it does not fit ------------------------------------------------
  const rescues = useMemo(() => {
    if (feasible !== false || !spec) return []
    return nearestFeasible(state, topology, spec)
  }, [feasible, spec, state, topology])

  const occupancy = capacity ? capacity.perGpu.total / capacity.available : 0
  const stepSeconds = account?.totalSeconds ?? 0
  const linkActivity =
    account && stepSeconds > 0
      ? Math.min(
          1,
          account.slices
            .filter((s) => s.color === 'var(--c-net-bw)' || s.color === 'var(--c-net-lat)')
            .reduce((sum, s) => sum + s.seconds, 0) / stepSeconds,
        )
      : 0

  const worldSize = parallelism.tensor * parallelism.pipeline * parallelism.data

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

  const stageLayers = spec ? largestStage(spec.numLayers, state.pipelineParallel) : 0

  // Slices that are charged to the step but happen outside any one layer, so the
  // timeline cannot show them and the caption has to say so.
  const OUTSIDE_A_LAYER = new Set(['hops', 'bubble', 'grad-allreduce'])
  const omitted = (account?.slices ?? []).filter((slice) => OUTSIDE_A_LAYER.has(slice.key))

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
              <h2>That model would not load</h2>
              <p className="muted" style={{ margin: 0 }}>
                {hub.error}
              </p>
            </section>
          )}

          {!spec && !hub.error && (
            <section className="panel" style={{ marginBottom: 'var(--gap)' }}>
              <p className="muted" style={{ margin: 0 }}>
                Reading the model configuration…
              </p>
            </section>
          )}

          {feasible === false && capacity && (
            <section className="panel verdict" style={{ borderTopColor: 'var(--c-net-lat)' }}>
              <div>
                <p className="eyebrow" style={{ margin: '0 0 0.5rem' }}>
                  {isInference ? 'Decode step' : 'Training step'}
                </p>
                <h2>This will not fit</h2>
                <p className="cause">{explainVerdict(capacity)}</p>

                {/* The moment the tool is least useful is the moment it should
                    work hardest. The configurations near this one that DO fit
                    are computable, so they are computed rather than described as
                    a category of thing the user might try. */}
                {rescues.length > 0 ? (
                  <>
                    <p className="lever">
                      <strong>What fits:</strong> the nearest working setups, smallest change
                      first.
                    </p>
                    <div className="rescues">
                      {rescues.map((rescue) => (
                        <button
                          key={rescue.label}
                          type="button"
                          className="ghost rescue"
                          onClick={() => setState({ ...state, ...rescue.patch })}
                        >
                          <span>Use {rescue.label}</span>
                          <span className="num muted">
                            {gib(rescue.usedBytes)} of {gib(rescue.availableBytes)}
                          </span>
                        </button>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="lever">
                    <strong>What helps:</strong> nothing reachable from here on this hardware —
                    a smaller model, or more GPUs.
                  </p>
                )}
              </div>
            </section>
          )}

          {account && bottleneck && (
            <Verdict
              bottleneck={bottleneck}
              share={account.dominant.seconds / account.totalSeconds}
              phase={state.phase}
              balanced={account.balanced}
              {...(account.runnerUp
                ? {
                    runnerUp: {
                      label: account.runnerUp.label,
                      share: account.runnerUp.seconds / account.totalSeconds,
                    },
                  }
                : {})}
              {...(crossover ? { crossover } : {})}
              {...(betterSplit ? { betterSplit } : {})}
            />
          )}

          {/* The controls belong to the page, not to the answer. Rendering them
              inside the success branch meant that changing one setting into an
              infeasible configuration removed the only way to change it back —
              and the state persists in the URL, so a reload did not help. */}
          <ScenarioBar
            state={state}
            topology={topology}
            resolved={hub.model}
            onEdit={() => setSettingsOpen(true)}
          />

          {account && (
            <>
              <div className="figures">
                {isInference ? (
                  <>
                    <Figure
                      value={grouped(inference!.aggregateTokensPerSecond!)}
                      label="tokens / second"
                    />
                    <Figure value={ms(account.totalSeconds)} label="per output token" />
                    <Figure
                      value={money(
                        costPerMillion(
                          inference!.aggregateTokensPerSecond!,
                          state.gpuHourCost,
                          worldSize,
                        ),
                      )}
                      label="per million tokens"
                    />
                  </>
                ) : (
                  <>
                    <Figure
                      value={`${(training!.tokensPerSecond! / 1000).toFixed(1)}k`}
                      label="tokens / second"
                    />
                    <Figure value={ms(account.totalSeconds)} label="per step" />
                    <Figure value={pct(training!.mfu!)} label="model FLOPs utilisation" />
                  </>
                )}
              </div>

              <section className="panel" style={{ marginBottom: 'var(--gap)' }}>
                <h2>Where the step goes</h2>
                <TimeBreakdown slices={account.slices} totalSeconds={account.totalSeconds} />

                {schedule && (
                  <div style={{ marginTop: '1.5rem' }}>
                    <StepTimeline
                      schedule={schedule}
                      caption={
                        <>
                          {isInference ? (
                            <>
                              One of {stageLayers} layers, in the order it runs.
                              {state.pipelineParallel > 1 &&
                                ` Each token crosses ${state.pipelineParallel} stages of these.`}{' '}
                              The whole step is <b className="num">{ms(account.totalSeconds)}</b>.
                            </>
                          ) : (
                            <>
                              One of {stageLayers} layers, forward and backward, for a single
                              microbatch. A step runs {state.gradientAccumulation} of them —{' '}
                              <b className="num">{ms(account.totalSeconds)}</b> in total.
                            </>
                          )}
                          {state.tensorParallel > 1 && (
                            <>
                              {' '}
                              The gaps on the GPU lane are time it is not computing — mostly the
                              all-reduces, since with tensor parallelism each one feeds the next
                              kernel and nothing can overlap it.
                            </>
                          )}
                          {/* Named, because this figure sits directly beneath a
                              whole-step bar and a reader is entitled to assume
                              it accounts for the same time. The terms below are
                              charged to the step and are not inside any layer,
                              and one of them can be the largest thing there. */}
                          {omitted.length > 0 && (
                            <>
                              {' '}
                              Not drawn, because it happens outside any single layer:{' '}
                              {omitted
                                .map(
                                  (slice) =>
                                    `${slice.label.toLowerCase()} (${Math.round(
                                      (slice.seconds / account.totalSeconds) * 100,
                                    )}% of the step)`,
                                )
                                .join(', ')}
                              .
                            </>
                          )}
                        </>
                      }
                    />
                  </div>
                )}
              </section>
            </>
          )}

          {/* Deliberately outside the success branch, and deliberately after the
              outcomes. Dragging the sweep into the region that does not fit used
              to unmount the sweep, which took away the control that had just
              caused the problem — a smaller version of the same mistake as the
              settings button. */}
          {activeSweep && (
            <section className="panel" style={{ marginBottom: 'var(--gap)' }}>
              <div className="panel-head">
                <h2 style={{ margin: 0 }}>What would change the answer</h2>
                {axes.length > 1 && (
                  <div className="segmented" aria-label="Axis to sweep">
                    {axes.map((option) => (
                      <button
                        key={option.key}
                        type="button"
                        aria-pressed={option.key === axis.key}
                        onClick={() => setAxisKey(option.key)}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <Sweep
                result={activeSweep}
                current={state[axis.key]}
                currentState={
                  account ? (account.balanced ? 'balanced' : account.bottleneck) : 'infeasible'
                }
                onChange={(value) => setState({ ...state, [axis.key]: value })}
              />
            </section>
          )}

          {account && (
            <>
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
                      showTheoretical={state.showTheoretical}
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
                    {/* The second roof was built and then never reachable: the
                        flag lived in the URL with no control and the chart was
                        passed a hardcoded false. The comparison is the point of
                        having both — the gap between them is the efficiency
                        model, made visible. */}
                    <label className="toggle" htmlFor="theoretical">
                      <input
                        id="theoretical"
                        type="checkbox"
                        checked={state.showTheoretical}
                        onChange={(e) =>
                          setState({ ...state, showTheoretical: e.target.checked })
                        }
                      />
                      <span>
                        Show the vendor&apos;s peak too
                        <em className="small muted">
                          The solid roof becomes NVIDIA&apos;s specification and the dashed one what
                          this model expects to reach. The gap between them is every efficiency
                          factor at once.
                        </em>
                      </span>
                    </label>
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
                  <>
                    <p>
                      HFU counts what the GPUs executed; MFU counts what the model required.
                      Recomputation raises the first and lowers the second, which is why MFU is the
                      throughput-honest one.
                    </p>
                    {/* The preview quotes two numbers, so the body owes the
                        reader the arithmetic behind them rather than a
                        definition of the terms. */}
                    <dl className="rows">
                      <Row
                        label="Model FLOPs per step"
                        value={`${(training!.modelFlopsPerStep! / 1e15).toFixed(1)} PFLOP`}
                      />
                      <Row
                        label="Hardware FLOPs per step"
                        value={`${(training!.hardwareFlopsPerStep! / 1e15).toFixed(1)} PFLOP`}
                      />
                      <Row label="Model FLOPs utilisation" value={pct(training!.mfu!)} />
                      <Row label="Hardware FLOPs utilisation" value={pct(training!.hfu!)} />
                      <Row label="Tokens per step" value={grouped(training!.globalBatchTokens!)} />
                    </dl>
                    <p className="small muted" style={{ marginTop: '1rem' }}>
                      The gap between the two is recomputation:{' '}
                      <b className="num" style={{ color: 'var(--ink)' }}>
                        {(
                          (training!.hardwareFlopsPerStep! / training!.modelFlopsPerStep! - 1) *
                          100
                        ).toFixed(1)}
                        %
                      </b>{' '}
                      more arithmetic than the model itself requires, run to avoid storing
                      activations.
                      {state.recompute === 'none' &&
                        ' With recomputation off, anything above zero here is the backward pass recomputing what it could not keep.'}
                    </p>
                  </>
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
                    <Suspense fallback={<p className="small muted">Loading the hardware view…</p>}>
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
                  preview={
                    betterSplit
                      ? `${betterSplit.label} — ${Math.round(betterSplit.gain * 100)}% faster`
                      : `TP ${state.tensorParallel} · PP ${state.pipelineParallel} · DP ${state.dataParallel} is the best of ${ranked.length}`
                  }
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

              <Disclosure
                title="How far should I trust this?"
                preview={
                  calibrated
                    ? 'Calibrated against your own measurements'
                    : 'Fitted priors, not measurements from your machine'
                }
              >
                <p>
                  Hardware peaks and capacities are NVIDIA specifications. Everything else — the
                  efficiency factors that derate them, the collective startup cost, the runtime
                  reserve — is a fitted prior: a plausible guess, chosen once and applied
                  everywhere. They are the reason two setups compare well here and an absolute
                  number may still be off.
                </p>
                <p className="small muted">
                  If you can run <code>all_reduce_perf</code> on the machine you care about, paste
                  the output under &ldquo;Could a different split be faster?&rdquo; → Calibrate and
                  the guesses are replaced by your measurements.
                </p>
              </Disclosure>
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
        onReset={() => setState(DEFAULT_STATE)}
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

/** `Infinity.toFixed(2)` is the string "Infinity", which is not a price. */
const money = (dollars: number) => (Number.isFinite(dollars) ? `$${dollars.toFixed(2)}` : '—')

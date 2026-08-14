import { Suspense, lazy, useEffect, useMemo, useState, type CSSProperties } from 'react'
import { type AppState, readStateFromLocation, writeStateToLocation } from './app/state'
import { PRESETS, presetById } from './data/presets'
import {
  DGX_H100,
  DTYPE_BYTES,
  SINGLE_H100,
  type Topology,
  ridgePoint,
} from './core/hardware'
import { H100_EFFICIENCY, H100_OVERHEAD, IDEAL_EFFICIENCY } from './core/efficiency'
import { fromHuggingFaceConfig } from './core/modelspec'
import { type InferenceConfig, analyseInference, computeBoundBatch } from './core/inference'
import { analyseTraining } from './core/training'
import type { BottleneckClass } from './core/schedule'
import { decodeAsymptotes, operatingPoint, roofline } from './core/roofline'
import { explainVerdict } from './core/capacity'
import { Roofline } from './viz/Roofline'
import { TimeBreakdown } from './viz/TimeBreakdown'
import { Advanced } from './ui/Advanced'
import { Collectives } from './ui/Collectives'
import { Controls } from './ui/Controls'
import { Readout } from './ui/Readout'

// Code-split, so the numbers and the roofline paint immediately and three.js
// streams in behind them rather than gating first render on ~115 KB.
const Scene = lazy(() => import('./scene/Scene'))

const TOPOLOGIES: Record<string, Topology> = {
  'h100-x1': SINGLE_H100,
  'h100-x8': DGX_H100,
}

/** The whole interface takes its hue from the diagnosis. */
const ACCENT: Record<BottleneckClass, string> = {
  compute: 'var(--c-compute)',
  'memory-bandwidth': 'var(--c-memory)',
  'network-bandwidth': 'var(--c-net-bw)',
  'network-latency': 'var(--c-net-lat)',
  overhead: 'var(--c-overhead)',
  serialization: 'var(--c-serial)',
}

export default function App() {
  const [state, setState] = useState<AppState>(readStateFromLocation)

  useEffect(() => {
    writeStateToLocation(state)
  }, [state])

  // The configuration lives in the URL, so the URL has to be able to drive it
  // back: browser back and forward, and a link edited by hand, both change the
  // hash without reloading the document.
  useEffect(() => {
    const onHashChange = () => setState(readStateFromLocation())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const topology = TOPOLOGIES[state.topologyId] ?? DGX_H100
  const efficiency = state.showTheoretical ? IDEAL_EFFICIENCY : H100_EFFICIENCY

  const spec = useMemo(() => {
    const preset = presetById(state.modelId)
    if (!preset) return undefined
    try {
      return fromHuggingFaceConfig(preset.config, { id: preset.id, name: preset.label })
    } catch {
      return undefined
    }
  }, [state.modelId])

  const parallelism = {
    tensor: state.tensorParallel,
    pipeline: state.pipelineParallel,
    data: state.dataParallel,
  }

  // The topology describes a node. Training routinely spans many, so the cluster
  // size is whatever the sharding degrees multiply out to.
  const worldSize = parallelism.tensor * parallelism.pipeline * parallelism.data

  // Shared with the sharding search, so both are costing the same workload.
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
            efficiency,
            overhead: H100_OVERHEAD,
            cudaGraph: state.cudaGraph,
            flashAttention: state.flashAttention,
          } satisfies InferenceConfig)
        : undefined,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [spec, topology, state, efficiency],
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
      efficiency,
      overhead: H100_OVERHEAD,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec, topology, state, efficiency])

  const isInference = state.phase === 'inference'
  const capacity = isInference ? inference?.capacity : training?.capacity
  const feasible = isInference ? inference?.feasible : training?.feasible
  const bottleneck = isInference ? inference?.decode?.bottleneck : training?.bottleneck
  const stepSchedule = isInference ? inference?.decode?.schedule : training?.schedule
  const stepSeconds = isInference
    ? (inference?.timePerOutputToken ?? 0)
    : (training?.stepTime ?? 0)

  const roof = useMemo(
    () => roofline(topology.accelerator, 'bf16', efficiency),
    [topology, efficiency],
  )

  const points = useMemo(() => {
    const phase = isInference ? inference?.decode : undefined
    if (!phase) return []
    return phase.classes
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
      .filter((point): point is NonNullable<typeof point> => point !== undefined)
  }, [inference, isInference, roof])

  const accent = bottleneck ? ACCENT[bottleneck] : 'var(--c-memory)'

  const occupancy = capacity
    ? {
        weights: capacity.perGpu.weights / capacity.available,
        working:
          (capacity.perGpu.kvCache +
            capacity.perGpu.gradients +
            capacity.perGpu.optimizerStates) /
          capacity.available,
        other: (capacity.perGpu.activations + capacity.perGpu.reserve) / capacity.available,
      }
    : { weights: 0, working: 0, other: 0 }

  const linkActivity =
    stepSchedule && stepSeconds > 0
      ? Math.min(
          1,
          (stepSchedule.breakdown['network-bandwidth'] +
            stepSchedule.breakdown['network-latency']) /
            stepSeconds,
        )
      : 0.12

  return (
    <div style={{ '--accent': accent } as CSSProperties}>
      <div className="stage">
        <Suspense fallback={null}>
          <Scene
            gpus={topology.gpus}
            accent={accent}
            occupancy={occupancy}
            linkActivity={linkActivity}
            infeasible={feasible === false}
          />
        </Suspense>
      </div>

      <div className={`overlay${feasible === false ? ' infeasible' : ''}`}>
        <header className="masthead">
          <h1>Compute Visualizer</h1>
          <span className="sub">
            {worldSize === topology.gpus
              ? topology.name
              : `${worldSize} × ${topology.accelerator.name}`}{' '}
            · {spec?.name ?? 'no model'}
          </span>
          <a href="https://github.com/alpharomercoma/compute-visualizer#readme" target="_blank" rel="noreferrer">
            How this is calculated
          </a>
        </header>

        <Controls state={state} onChange={setState} topology={topology} presets={PRESETS} />

        {/* Bottom-aligned and narrow, so the board stays visible around it.
            Panels scattered over the scene rather than a wall of chrome. */}
        <div className="centre-column">
          <div>
            {inferenceConfig && (
              <Advanced
                state={state}
                onChange={setState}
                baseConfig={inferenceConfig}
                worldSize={topology.gpus}
              />
            )}
          </div>
        </div>

        <Readout
          state={state}
          spec={spec}
          topology={topology}
          inference={inference}
          training={training}
        />

        <div className="foot">
          <section className="panel" style={{ flex: 2 }}>
            <div className="section-head">
              {isInference ? 'Decode step' : 'Training step'} — where the time went
            </div>
            {feasible && stepSchedule ? (
              <>
                <TimeBreakdown schedule={stepSchedule} totalSeconds={stepSeconds} />
                {spec && (
                  <>
                    <div className="section-head" style={{ marginTop: 16 }}>
                      Interconnect
                    </div>
                    <Collectives
                      topology={topology}
                      tensorParallel={state.tensorParallel}
                      messageBytes={
                        (isInference ? state.batch : state.sequenceLength) * spec.hiddenSize * 2
                      }
                      collectivesPerStep={
                        state.tensorParallel > 1
                          ? spec.numLayers * (isInference ? 2 : 4)
                          : 0
                      }
                    />
                  </>
                )}
              </>
            ) : (
              <p className="note">
                {capacity && feasible === false
                  ? explainVerdict(capacity)
                  : 'Choose a model to begin.'}
              </p>
            )}
          </section>

          <section className="panel" style={{ flex: 3 }}>
            <div className="section-head">Roofline — one point per kernel class</div>
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
            <p className="note">
              Ridge point {ridgePoint(topology.accelerator, 'bf16').toFixed(0)} FLOP/byte. Decode
              linear layers reach it at batch{' '}
              {computeBoundBatch(topology, 'bf16', state.weightDtype).toFixed(0)}. Attention never
              reaches it at any batch size, because every sequence carries its own KV cache.
            </p>
          </section>
        </div>
      </div>
    </div>
  )
}

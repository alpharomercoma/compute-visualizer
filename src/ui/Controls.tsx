import type { AppState } from '../app/state'
import { shardingOptions } from '../app/state'
import type { Topology } from '../core/hardware'
import type { Preset } from '../data/presets'
import type { ResolvedModel } from '../hub/useHubModel'
import { ModelPicker } from './ModelPicker'

/**
 * The inputs.
 *
 * Every control names the thing the user controls, not the thing the model
 * calls it: "prompt length", not "queryLength". The parallelism controls only
 * offer splits that actually divide the world size, because a configuration that
 * cannot exist is not a useful thing to be able to type.
 */

export interface ControlsProps {
  readonly state: AppState
  readonly onChange: (next: AppState) => void
  readonly topology: Topology
  readonly presets: readonly Preset[]
  readonly resolved: ResolvedModel | undefined
  readonly modelLoading: boolean
  readonly modelError: string | undefined
}

export function Controls({
  state,
  onChange,
  topology,
  presets,
  resolved,
  modelLoading,
  modelError,
}: ControlsProps) {
  const set = <K extends keyof AppState>(key: K, value: AppState[K]) =>
    onChange({ ...state, [key]: value })

  const preset = presets.find((p) => p.id === state.modelId)
  const splits = shardingOptions(topology.gpus)
  const currentSplit = `${state.tensorParallel}-${state.pipelineParallel}-${state.dataParallel}`
  const worldSize = state.tensorParallel * state.pipelineParallel * state.dataParallel

  return (
    <aside className="panel">
      <div className="segmented" style={{ marginBottom: 14 }}>
        {(['inference', 'training'] as const).map((phase) => (
          <button
            key={phase}
            type="button"
            aria-pressed={state.phase === phase}
            onClick={() => set('phase', phase)}
          >
            {phase}
          </button>
        ))}
      </div>

      <div className="section-head">Model</div>
      <ModelPicker
        modelId={state.modelId}
        onChange={(repo) => set('modelId', repo)}
        resolved={resolved}
        loading={modelLoading}
        error={modelError}
      />
      {preset?.note && (
        <p className="note" style={{ marginTop: -4, marginBottom: 12 }}>
          {preset.note}
        </p>
      )}

      <div className="section-head">Hardware</div>
      <div className="field">
        <label className="label" htmlFor="topology">
          Accelerators
        </label>
        <select
          id="topology"
          value={state.topologyId}
          onChange={(e) => {
            // Changing the machine can invalidate the split, so reset to a valid
            // one rather than leaving an impossible configuration on screen.
            const next = e.target.value
            const gpus = next === 'h100-x1' ? 1 : 8
            onChange({
              ...state,
              topologyId: next,
              tensorParallel: Math.min(state.tensorParallel, gpus),
              pipelineParallel: 1,
              dataParallel: 1,
            })
          }}
        >
          <option value="h100-x1">1x H100 SXM5</option>
          <option value="h100-x8">8x H100 SXM5 (NVSwitch)</option>
        </select>
      </div>

      {/* Inference is served from one node, so its split has to divide that
          node's GPU count. Training routinely spans many, so there the degrees
          are free and the cluster size is derived from them — the topology
          describes a node, not the whole machine. */}
      {state.phase === 'inference' ? (
        <div className="field">
          <label className="label" htmlFor="split">
            Sharding — tensor / pipeline / data
          </label>
          <select
            id="split"
            value={currentSplit}
            onChange={(e) => {
              const [tensor, pipeline, data] = e.target.value.split('-').map(Number)
              onChange({
                ...state,
                tensorParallel: tensor!,
                pipelineParallel: pipeline!,
                dataParallel: data!,
              })
            }}
          >
            {splits.map((split) => (
              <option
                key={`${split.tensor}-${split.pipeline}-${split.data}`}
                value={`${split.tensor}-${split.pipeline}-${split.data}`}
              >
                TP {split.tensor} · PP {split.pipeline} · DP {split.data}
              </option>
            ))}
          </select>
        </div>
      ) : (
        <div className="field">
          <label className="label">Sharding degrees</label>
          <div className="row">
            {(
              [
                ['tensorParallel', 'TP', topology.gpus],
                ['pipelineParallel', 'PP', 64],
                ['dataParallel', 'DP', 512],
              ] as const
            ).map(([key, short, max]) => (
              <div key={key}>
                <label className="label" htmlFor={key} style={{ marginBottom: 3 }}>
                  {short}
                </label>
                <select
                  id={key}
                  value={state[key]}
                  onChange={(e) => set(key, Number(e.target.value))}
                >
                  {powersOfTwo(max).map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
          <p className="note" style={{ marginTop: 6 }}>
            {worldSize.toLocaleString()} GPUs
            {topology.gpus > 1 && ` · ${Math.ceil(worldSize / topology.gpus)} node${
              Math.ceil(worldSize / topology.gpus) === 1 ? '' : 's'
            }`}
            . Tensor parallelism is capped at one node, because crossing the
            node boundary with it is rarely worth the bandwidth.
          </p>
        </div>
      )}

      <div className="section-head">Precision</div>
      <div className="row field">
        <div>
          <label className="label" htmlFor="wd">
            Weights
          </label>
          <select id="wd" value={state.weightDtype} onChange={(e) => set('weightDtype', e.target.value as AppState['weightDtype'])}>
            <option value="bf16">BF16</option>
            <option value="fp8">FP8</option>
            <option value="int8">INT8 (W8A16)</option>
            <option value="int4">INT4 (W4A16)</option>
          </select>
        </div>
        <div>
          <label className="label" htmlFor="kd">
            KV cache
          </label>
          <select id="kd" value={state.kvDtype} onChange={(e) => set('kvDtype', e.target.value as AppState['kvDtype'])}>
            <option value="bf16">BF16</option>
            <option value="fp8">FP8</option>
            <option value="int8">INT8</option>
          </select>
        </div>
      </div>

      {state.phase === 'inference' ? (
        <>
          <div className="section-head">Workload</div>
          <Slider
            id="batch"
            label="Batch — concurrent sequences"
            value={state.batch}
            min={1}
            max={512}
            onChange={(v) => set('batch', v)}
          />
          <Slider
            id="prompt"
            label="Prompt length"
            value={state.promptLength}
            min={128}
            max={32768}
            step={128}
            onChange={(v) => set('promptLength', v)}
          />
          <Slider
            id="generate"
            label="Tokens to generate"
            value={state.generateLength}
            min={1}
            max={4096}
            step={1}
            onChange={(v) => set('generateLength', v)}
          />

          <div className="section-head">Execution</div>
          <Toggle
            id="graph"
            label="CUDA graph capture"
            hint="Removes most per-kernel launch cost."
            checked={state.cudaGraph}
            onChange={(v) => set('cudaGraph', v)}
          />
          <Toggle
            id="flash"
            label="FlashAttention"
            hint="Keeps attention traffic linear in sequence length."
            checked={state.flashAttention}
            onChange={(v) => set('flashAttention', v)}
          />
        </>
      ) : (
        <>
          <div className="section-head">Workload</div>
          <Slider
            id="seq"
            label="Sequence length"
            value={state.sequenceLength}
            min={512}
            max={32768}
            step={512}
            onChange={(v) => set('sequenceLength', v)}
          />
          <Slider
            id="accum"
            label="Microbatches per step"
            value={state.gradientAccumulation}
            min={1}
            max={256}
            onChange={(v) => set('gradientAccumulation', v)}
          />

          <div className="section-head">Memory strategy</div>
          <div className="field">
            <label className="label" htmlFor="opt">
              Optimizer
            </label>
            <select id="opt" value={state.optimizer} onChange={(e) => set('optimizer', e.target.value as AppState['optimizer'])}>
              <option value="adam-mixed">Adam, mixed precision — 16 bytes/param</option>
              <option value="adam-fp32-grads">Adam, fp32 gradients — 18 bytes/param</option>
              <option value="sgd-momentum">SGD with momentum — 8 bytes/param</option>
              <option value="adam-8bit">8-bit Adam — ~10 bytes/param</option>
            </select>
          </div>
          <div className="field">
            <label className="label" htmlFor="zero">
              ZeRO stage
            </label>
            <select
              id="zero"
              value={state.zeroStage}
              onChange={(e) => set('zeroStage', Number(e.target.value) as AppState['zeroStage'])}
            >
              <option value={0}>0 — replicate everything</option>
              <option value={1}>1 — partition optimizer state</option>
              <option value={2}>2 — and gradients</option>
              <option value={3}>3 — and parameters</option>
            </select>
          </div>
          <div className="field">
            <label className="label" htmlFor="recompute">
              Activation recomputation
            </label>
            <select
              id="recompute"
              value={state.recompute}
              onChange={(e) => set('recompute', e.target.value as AppState['recompute'])}
            >
              <option value="none">None — fastest, largest</option>
              <option value="selective">Selective — attention only</option>
              <option value="full">Full — smallest, ~33% more work</option>
            </select>
          </div>
          <Toggle
            id="sp"
            label="Sequence parallelism"
            hint="Shards norms and dropout along the sequence."
            checked={state.sequenceParallel}
            onChange={(v) => set('sequenceParallel', v)}
          />
        </>
      )}

      <div className="section-head">Display</div>
      <Toggle
        id="theoretical"
        label="Show theoretical peak"
        hint="Adds the undecorated vendor roof beside the achievable one."
        checked={state.showTheoretical}
        onChange={(v) => set('showTheoretical', v)}
      />
    </aside>
  )
}

function Slider({
  id,
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  id: string
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
}) {
  return (
    <div className="field">
      <label className="label" htmlFor={id}>
        {label} <span style={{ color: 'var(--ink)' }}>{value.toLocaleString()}</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}

function Toggle({
  id,
  label,
  hint,
  checked,
  onChange,
}: {
  id: string
  label: string
  hint?: string
  checked: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <div className="field">
      <label htmlFor={id} style={{ display: 'flex', gap: 7, alignItems: 'flex-start', cursor: 'pointer' }}>
        <input
          id={id}
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          style={{ accentColor: 'var(--accent)', marginTop: 2 }}
        />
        <span>
          <span style={{ fontSize: 'var(--t-small)' }}>{label}</span>
          {hint && <span className="note" style={{ display: 'block' }}>{hint}</span>}
        </span>
      </label>
    </div>
  )
}

/** Parallelism degrees are powers of two in every framework worth modelling. */
function powersOfTwo(max: number): number[] {
  const out: number[] = []
  for (let n = 1; n <= max; n *= 2) out.push(n)
  return out
}


import { useEffect, useRef, useState } from 'react'
import type { AppState } from '../app/state'
import { normalise, shardingOptions } from '../app/state'
import type { Topology } from '../core/hardware'
import { PRESETS } from '../data/presets'
import type { ResolvedModel } from '../hub/useHubModel'

/**
 * The scenario, as a sentence you can click.
 *
 * The previous build put ~20 inputs on the first screen, which told the user
 * they had to understand twenty things before the tool would answer them. The
 * configuration is now a readable summary — "Llama 3.1 70B on 8 × H100, batch
 * 32, 2k context" — and the controls live in a sheet you open when you want to
 * change something.
 */

export interface ScenarioProps {
  readonly state: AppState
  readonly topology: Topology
  readonly resolved: ResolvedModel | undefined
  readonly onEdit: () => void
}

const compact = (n: number): string =>
  n >= 1024 ? `${Math.round(n / 1024)}k` : String(n)

export function ScenarioBar({ state, topology, resolved, onEdit }: ScenarioProps) {
  const name = resolved?.spec.name ?? state.modelId.split('/').pop() ?? state.modelId
  const gpus = state.tensorParallel * state.pipelineParallel * state.dataParallel

  return (
    <div className="scenario">
      <p className="summary" style={{ margin: 0 }}>
        <b>{name}</b> on <b>{gpus} × {topology.accelerator.name.replace(/ \d+GB$/, '')}</b>
        {state.phase === 'inference' ? (
          <>
            {/* The generated length is in the summary because it is in the KV
                cache: a 2k prompt with 4k of output was described as "2k
                context" while occupying three times that. */}
            , batch <b>{state.batch}</b>, <b>{compact(state.promptLength)}</b> in,{' '}
            <b>{compact(state.generateLength)}</b> out
          </>
        ) : (
          <>
            , <b>{compact(state.sequenceLength)}</b> sequence, <b>{state.gradientAccumulation}</b>{' '}
            microbatches
          </>
        )}
      </p>
      <button type="button" className="ghost" onClick={onEdit}>
        Change this
      </button>
    </div>
  )
}

/**
 * The settings sheet.
 *
 * A real dialog: focus moves in, Escape closes it, the backdrop closes it, and
 * on a phone it sits at the bottom where a thumb can reach. Controls that
 * changed the model silently from a 28px unlabelled row are gone.
 */
export interface SettingsProps {
  readonly open: boolean
  readonly onClose: () => void
  readonly state: AppState
  readonly onChange: (next: AppState) => void
  readonly topology: Topology
  readonly modelSlot: React.ReactNode
  readonly onReset: () => void
}

export function Settings({
  open,
  onClose,
  state,
  onChange,
  topology,
  modelSlot,
  onReset,
}: SettingsProps) {
  const ref = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (open && !dialog.open) dialog.showModal()
    if (!open && dialog.open) dialog.close()
  }, [open])

  const set = <K extends keyof AppState>(key: K, value: AppState[K]) =>
    onChange({ ...state, [key]: value })

  const splits = shardingOptions(topology.gpus)
  const currentSplit = `${state.tensorParallel}-${state.pipelineParallel}-${state.dataParallel}`

  return (
    <dialog
      ref={ref}
      className="sheet"
      onClose={onClose}
      // A native <dialog> does NOT close on a backdrop click by itself. The
      // backdrop is part of the dialog's own box, so a click landing outside
      // the inner content is the signal.
      onClick={(e) => {
        if (e.target === ref.current) onClose()
      }}
    >
      <div className="sheet-head">
        <h2 style={{ margin: 0 }}>Scenario</h2>
        <button type="button" className="ghost" onClick={onClose} aria-label="Close settings">
          Done
        </button>
      </div>

      <div className="sheet-body">
        <div className="field">
          <span className="label">Workload</span>
          <div className="segmented">
            {(['inference', 'training'] as const).map((phase) => (
              <button
                key={phase}
                type="button"
                aria-pressed={state.phase === phase}
                onClick={() => set('phase', phase)}
              >
                {phase === 'inference' ? 'Inference' : 'Training'}
              </button>
            ))}
          </div>
        </div>

        {modelSlot}

        <div className="field">
          <label className="label" htmlFor="topology">
            Hardware
          </label>
          <select
            id="topology"
            value={state.topologyId}
            // Routed through the same normaliser the URL uses, so a split is
            // valid for the hardware by construction rather than by two copies
            // of the clamping rule agreeing with each other.
            onChange={(e) => onChange(normalise({ ...state, topologyId: e.target.value }))}
          >
            <option value="h100-x1">1 × H100 SXM5</option>
            <option value="h100-x8">8 × H100 SXM5 (NVSwitch)</option>
          </select>
        </div>

        {state.phase === 'inference' ? (
          <>
            <NumberField
              id="batch"
              label="Batch — concurrent sequences"
              value={state.batch}
              min={1}
              max={512}
              onChange={(v) => set('batch', v)}
            />
            <NumberField
              id="prompt"
              label="Prompt length"
              value={state.promptLength}
              min={128}
              max={32768}
              step={128}
              onChange={(v) => set('promptLength', v)}
            />
            <NumberField
              id="generate"
              label="Tokens to generate"
              value={state.generateLength}
              min={1}
              max={4096}
              onChange={(v) => set('generateLength', v)}
            />
          </>
        ) : (
          <>
            <NumberField
              id="seq"
              label="Sequence length"
              value={state.sequenceLength}
              min={512}
              max={32768}
              step={512}
              onChange={(v) => set('sequenceLength', v)}
            />
            <NumberField
              id="accum"
              label="Microbatches per step"
              value={state.gradientAccumulation}
              min={1}
              max={256}
              onChange={(v) => set('gradientAccumulation', v)}
            />
          </>
        )}

        <details className="disclosure" style={{ boxShadow: 'none' }}>
          <summary>
            <span className="title">Precision and sharding</span>
            <span className="preview">
              {state.weightDtype.toUpperCase()} weights · TP {state.tensorParallel}
            </span>
          </summary>
          <div className="body">
            <div className="field">
              <label className="label" htmlFor="wd">
                Weights
              </label>
              <select
                id="wd"
                value={state.weightDtype}
                onChange={(e) => set('weightDtype', e.target.value as AppState['weightDtype'])}
              >
                <option value="bf16">BF16 — 2 bytes</option>
                <option value="fp8">FP8 — 1 byte</option>
                <option value="int8">INT8 weight-only</option>
                <option value="int4">INT4 weight-only</option>
              </select>
            </div>
            {/* Inference only. Training has no KV cache to keep, so a control
                for its precision changed nothing at all. */}
            {state.phase === 'inference' && (
              <div className="field">
                <label className="label" htmlFor="kd">
                  KV cache
                </label>
                <select
                  id="kd"
                  value={state.kvDtype}
                  onChange={(e) => set('kvDtype', e.target.value as AppState['kvDtype'])}
                >
                  <option value="bf16">BF16</option>
                  <option value="fp8">FP8</option>
                  <option value="int8">INT8</option>
                </select>
              </div>
            )}
            <div className="field">
              <label className="label" htmlFor="split">
                Sharding
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
                {splits.map((s) => (
                  <option
                    key={`${s.tensor}-${s.pipeline}-${s.data}`}
                    value={`${s.tensor}-${s.pipeline}-${s.data}`}
                  >
                    TP {s.tensor} · PP {s.pipeline} · DP {s.data}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </details>

        {/* Every one of these changed a reported number while having no control
            anywhere in the interface, which made the answer depend on state the
            user could neither see nor set. The training block was the worst of
            it: ZeRO stage and recomputation decide whether a run fits at all. */}
        {state.phase === 'training' && (
          <details className="disclosure" style={{ boxShadow: 'none' }}>
            <summary>
              <span className="title">Optimizer and memory</span>
              <span className="preview">
                ZeRO-{state.zeroStage} · {state.recompute} recompute
              </span>
            </summary>
            <div className="body">
              <div className="field">
                <label className="label" htmlFor="opt">
                  Optimizer
                </label>
                <select
                  id="opt"
                  value={state.optimizer}
                  onChange={(e) => set('optimizer', e.target.value as AppState['optimizer'])}
                >
                  {/* Bytes of OPTIMIZER STATE per parameter, matching
                      OPTIMIZER_BYTES exactly. The earlier numbers were neither
                      the state nor the total, and two of them were simply
                      wrong. */}
                  <option value="adam-mixed">Adam, mixed precision — 12 bytes of state</option>
                  <option value="adam-fp32-grads">
                    Adam, FP32 gradients — 12 bytes of state, 4-byte grads
                  </option>
                  <option value="adam-8bit">Adam, 8-bit states — 6 bytes of state</option>
                  <option value="sgd-momentum">SGD with momentum — 4 bytes of state</option>
                </select>
              </div>
              <div className="field">
                <label className="label" htmlFor="zero">
                  ZeRO stage
                </label>
                <select
                  id="zero"
                  value={state.zeroStage}
                  onChange={(e) =>
                    set('zeroStage', Number(e.target.value) as AppState['zeroStage'])
                  }
                >
                  <option value={0}>0 — replicate everything</option>
                  <option value={1}>1 — shard optimizer states</option>
                  <option value={2}>2 — shard gradients too</option>
                  <option value={3}>3 — shard parameters too</option>
                </select>
                {/* ZeRO shards along the data-parallel dimension, so at DP 1
                    there is nothing to shard and the setting is inert. Leaving
                    that unsaid makes a control that changes no number look
                    broken. */}
                {state.dataParallel === 1 && (
                  <p className="small muted" style={{ margin: '0.375rem 0 0' }}>
                    Shards across data-parallel ranks. At DP 1 there is nothing to shard, so this
                    changes nothing until you reshard.
                  </p>
                )}
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
                  <option value="none">None — store every activation</option>
                  <option value="selective">Selective — recompute attention only</option>
                  <option value="full">Full — recompute every layer</option>
                </select>
              </div>
              <Toggle
                id="sp"
                label="Sequence parallelism"
                {...(state.tensorParallel === 1
                  ? { hint: 'Shards what tensor parallelism leaves replicated. At TP 1 there is nothing to shard.' }
                  : { hint: 'Shards the layer norms and dropouts that tensor parallelism replicates.' })}
                checked={state.sequenceParallel}
                onChange={(v) => set('sequenceParallel', v)}
              />
            </div>
          </details>
        )}

        {/* Inference only, all of it. The cost figure is a serving metric and
            neither runtime switch is read by the training path — three controls
            that moved nothing. */}
        {state.phase === 'inference' && (
          <details className="disclosure" style={{ boxShadow: 'none' }}>
            <summary>
              <span className="title">Runtime and cost</span>
              <span className="preview">
                ${state.gpuHourCost.toFixed(2)} per GPU-hour
                {state.cudaGraph ? ' · CUDA graphs' : ''}
              </span>
            </summary>
            <div className="body">
              <NumberField
                id="cost"
                label="Cost per GPU-hour, in dollars"
                value={state.gpuHourCost}
                min={0}
                max={100}
                step={0.05}
                onChange={(v) => set('gpuHourCost', v)}
              />
              <Toggle
                id="cg"
                label="CUDA graph capture"
                hint="Removes most per-kernel launch cost."
                checked={state.cudaGraph}
                onChange={(v) => set('cudaGraph', v)}
              />
              <Toggle
                id="fa"
                label="FlashAttention"
                hint="Keeps the attention matrix off HBM."
                checked={state.flashAttention}
                onChange={(v) => set('flashAttention', v)}
              />
            </div>
          </details>
        )}

        <button
          type="button"
          className="ghost"
          style={{ marginTop: '0.5rem' }}
          onClick={onReset}
        >
          Reset everything
        </button>
      </div>
    </dialog>
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
      <label className="toggle" htmlFor={id}>
        <input
          id={id}
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span>
          {label}
          {hint && <em className="small muted">{hint}</em>}
        </span>
      </label>
    </div>
  )
}

/**
 * A number input rather than a range slider.
 *
 * The sliders mapped 1–512 onto a 340px track on a phone: one pixel was 1.5
 * units and a precise value was unreachable by touch. Typing 64 works
 * everywhere.
 */
function NumberField({
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
  const [draft, setDraft] = useState<string>()

  return (
    <div className="field">
      <label className="label" htmlFor={id}>
        {label}{' '}
        <span style={{ fontWeight: 400, opacity: 0.8 }}>
          ({min.toLocaleString('en-US')}–{max.toLocaleString('en-US')})
        </span>
      </label>
      <input
        id={id}
        type="number"
        inputMode="numeric"
        // Held locally while being typed. Committing only in-range values to a
        // strictly controlled input meant backspacing "32" produced "" -> 0,
        // which failed `>= min`, so nothing was committed and the field snapped
        // straight back to 32. The value could not be cleared and therefore
        // could not be replaced by typing — only by selecting and overwriting.
        value={draft ?? String(value)}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          setDraft(e.target.value)
          const next = Number(e.target.value)
          if (e.target.value !== '' && Number.isFinite(next) && next >= min && next <= max) {
            onChange(next)
          }
        }}
        onBlur={() => setDraft(undefined)}
      />
    </div>
  )
}

export const PRESET_COUNT = PRESETS.length

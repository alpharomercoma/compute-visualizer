import { useEffect, useRef } from 'react'
import type { AppState } from '../app/state'
import { shardingOptions } from '../app/state'
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
            , batch <b>{state.batch}</b>, <b>{compact(state.promptLength)}</b> context
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
}

export function Settings({
  open,
  onClose,
  state,
  onChange,
  topology,
  modelSlot,
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
            onChange={(e) => {
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
            {state.phase === 'inference' && (
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
            )}
          </div>
        </details>
      </div>
    </dialog>
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
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const next = Number(e.target.value)
          if (Number.isFinite(next) && next >= min && next <= max) onChange(next)
        }}
      />
    </div>
  )
}

export const PRESET_COUNT = PRESETS.length

import { useMemo, useState } from 'react'
import type { AppState } from '../app/state'
import type { InferenceConfig } from '../core/inference'
import { rankSharding } from '../core/search'
import { fitCollective, parseNcclTests, compareToPrior, profileFromFit } from '../core/calibration'
import { type CollectiveProfile, defaultProfiles } from '../core/collectives'
import { KiB, MiB } from '../core/units'

/**
 * Two things that only make sense once the numbers exist: the best way to shard
 * what you have, and a way to replace our guesses with your measurements.
 */

export interface AdvancedProps {
  readonly state: AppState
  readonly onChange: (next: AppState) => void
  readonly baseConfig: InferenceConfig | undefined
  readonly worldSize: number
  /** Publishes a measured collective profile, replacing the shipped priors. */
  readonly onCalibrated?: (profiles: readonly CollectiveProfile[] | undefined) => void
  readonly calibrated?: boolean
}

export function Advanced({
  state,
  onChange,
  baseConfig,
  worldSize,
  onCalibrated,
  calibrated,
}: AdvancedProps) {
  const [tab, setTab] = useState<'sharding' | 'calibrate'>('sharding')

  return (
    <section className="panel">
      <div className="segmented" style={{ marginBottom: 12 }}>
        {(['sharding', 'calibrate'] as const).map((key) => (
          <button key={key} type="button" aria-pressed={tab === key} onClick={() => setTab(key)}>
            {key === 'sharding' ? 'Best split' : calibrated ? 'Calibrated' : 'Calibrate'}
          </button>
        ))}
      </div>

      {tab === 'sharding' ? (
        <Sharding
          state={state}
          onChange={onChange}
          baseConfig={baseConfig}
          worldSize={worldSize}
        />
      ) : (
        <Calibrate
          worldSize={worldSize}
          {...(onCalibrated ? { onCalibrated } : {})}
          {...(calibrated !== undefined ? { calibrated } : {})}
        />
      )}
    </section>
  )
}

function Sharding({ state, onChange, baseConfig, worldSize }: AdvancedProps) {
  const ranked = useMemo(() => {
    if (!baseConfig || state.phase !== 'inference') return []
    return rankSharding(baseConfig, worldSize, 'throughput')
  }, [baseConfig, worldSize, state.phase])

  if (state.phase !== 'inference') {
    return <p className="small muted">Sharding search runs on inference configurations.</p>
  }
  if (ranked.length === 0) {
    return <p className="small muted">No valid split of {worldSize} GPUs for this model.</p>
  }

  const best = ranked[0]!
  const current = `${state.tensorParallel}-${state.pipelineParallel}-${state.dataParallel}`

  return (
    <div>
      <p className="small muted" style={{ marginBottom: 10 }}>
        Every way to split {worldSize} GPUs, ranked by aggregate throughput. The search is
        exhaustive — there are only {ranked.length} of them.
      </p>

      {ranked.map((candidate) => {
        const key = `${candidate.tensor}-${candidate.pipeline}-${candidate.data}`
        const isCurrent = key === current
        const relative = candidate.feasible
          ? (candidate.tokensPerSecond ?? 0) / (best.tokensPerSecond ?? 1)
          : 0

        return (
          <button
            key={key}
            type="button"
            onClick={() =>
              onChange({
                ...state,
                tensorParallel: candidate.tensor,
                pipelineParallel: candidate.pipeline,
                dataParallel: candidate.data,
              })
            }
            aria-current={isCurrent}
            style={{
              display: 'block',
              width: '100%',
              textAlign: 'left',
              // A 6%-white wash marks the current row on a dark panel and is
              // invisible on a white one. Light is the default theme, so the
              // marker was missing exactly where most people would look for it.
              background: isCurrent ? 'var(--panel-sunk)' : 'transparent',
              border: 0,
              borderLeft: `2px solid ${isCurrent ? 'var(--accent)' : 'transparent'}`,
              padding: '0.625rem 0.75rem',
              minHeight: 'var(--tap)',
              borderRadius: '6px',
              font: 'inherit',
              color: 'inherit',
              cursor: 'pointer',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'baseline',
                gap: '1rem',
              }}
            >
              <span style={{ color: candidate.feasible ? 'var(--ink)' : 'var(--ink-2)' }}>
                TP {candidate.tensor} · PP {candidate.pipeline} · DP {candidate.data}
              </span>
              <span className="num" style={{ color: 'var(--ink-2)' }}>
                {candidate.feasible
                  ? `${Math.round(candidate.tokensPerSecond ?? 0).toLocaleString('en-US')} tok/s`
                  : 'will not fit'}
              </span>
            </div>
            {candidate.feasible && (
              <div
                aria-hidden="true"
                style={{
                  height: 2,
                  marginTop: 2,
                  width: `${Math.max(2, relative * 100)}%`,
                  background: isCurrent ? 'var(--accent)' : 'var(--rule-strong)',
                }}
              />
            )}
          </button>
        )
      })}
    </div>
  )
}

function Calibrate({
  worldSize,
  onCalibrated,
  calibrated,
}: {
  worldSize: number
  onCalibrated?: (profiles: readonly CollectiveProfile[] | undefined) => void
  calibrated?: boolean
}) {
  const [text, setText] = useState('')

  const fit = useMemo(() => {
    if (!text.trim()) return undefined
    const { measurements, ranks } = parseNcclTests(text)
    return fitCollective(measurements, ranks)
  }, [text])

  const prior = defaultProfiles(worldSize, true).find((p) => p.algorithm === 'ring')!
  const delta = fit ? compareToPrior(fit, prior) : undefined

  return (
    <div>
      <p className="small muted" style={{ marginBottom: 8 }}>
        Every collective constant here is a fitted guess, not a vendor specification. Measure
        yours and paste the output:
      </p>
      <code
        className="small muted"
        style={{
          display: 'block',
          background: 'var(--surface-lift)',
          padding: '6px 7px',
          borderRadius: 'var(--radius)',
          marginBottom: 8,
          userSelect: 'all',
        }}
      >
        ./build/all_reduce_perf -b 1K -e 1G -f 2 -g {worldSize} -n 100
      </code>

      <label className="label" htmlFor="ncclout">
        nccl-tests output
      </label>
      <textarea
        id="ncclout"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={5}
        placeholder="#  size  count  type  redop  root  time  algbw  busbw ..."
        style={{
          width: '100%',
          background: 'var(--surface-lift)',
          color: 'var(--ink)',
          border: '1px solid var(--rule-strong)',
          borderRadius: 'var(--radius)',
          padding: '6px 7px',
          font: 'inherit',
          fontSize: 'var(--t-micro)',
          resize: 'vertical',
        }}
      />

      {fit ? (
        <>
          <p className="label" style={{ marginTop: 12 }}>
            Fitted{' '}
            {calibrated ? (
              <span className="tag calibrated">in use</span>
            ) : (
              <span className="tag">not applied</span>
            )}
          </p>
          <dl className="rows">
            <Row label="Measurements" value={`${fit.measurements.length}`} />
            <Row label="Startup α" value={`${(fit.alpha * 1e6).toFixed(2)} µs`} />
            <Row label="algbw" value={`${(fit.algbw / 1e9).toFixed(0)} GB/s`} />
            <Row
              label="Latency knee"
              value={
                fit.kneeBytes >= MiB(1)
                  ? `${(fit.kneeBytes / MiB(1)).toFixed(2)} MiB`
                  : `${(fit.kneeBytes / KiB(1)).toFixed(0)} KiB`
              }
            />
          </dl>
          {delta && (
            <p className="small muted" style={{ marginTop: 8 }}>
              Against the shipped prior: α is {delta.alphaRatio.toFixed(2)}× and bandwidth{' '}
              {delta.algbwRatio.toFixed(2)}×. If those are far from 1, every other number here
              deserves the same scepticism.
            </p>
          )}
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: 10 }}>
            <button
              type="button"
              className="ghost"
              onClick={() => onCalibrated?.([profileFromFit(fit)])}
            >
              Use these numbers
            </button>
            {calibrated && (
              <button type="button" className="ghost" onClick={() => onCalibrated?.(undefined)}>
                Back to priors
              </button>
            )}
          </div>
        </>
      ) : (
        text.trim() && (
          <p className="small muted" style={{ marginTop: 8 }}>
            No measurements found. Expecting the standard table: size, count, type, redop, root,
            then time, algbw and busbw.
          </p>
        )
      )}
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

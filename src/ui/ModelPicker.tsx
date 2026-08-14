import { useEffect, useRef, useState } from 'react'
import { type ModelSummary, searchModels } from '../hub/client'
import type { ResolvedModel } from '../hub/useHubModel'
import { PRESETS } from '../data/presets'

/**
 * Choosing a model.
 *
 * The bundled catalog answers instantly and offline; the Hub answers for
 * everything else. Both are in the same control, because from the user's side
 * there is one question — which model — and where its shapes came from is our
 * problem, not theirs.
 */

export interface ModelPickerProps {
  readonly modelId: string
  readonly onChange: (repo: string) => void
  readonly resolved: ResolvedModel | undefined
  readonly loading: boolean
  readonly error: string | undefined
}

export function ModelPicker({ modelId, onChange, resolved, loading, error }: ModelPickerProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<ModelSummary[]>([])
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState<string>()
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([])
      setSearchError(undefined)
      return
    }

    // Debounced: the Hub's anonymous quota is per IP and shared with everyone
    // behind the same address, so a request per keystroke is rude.
    clearTimeout(timer.current)
    const controller = new AbortController()
    timer.current = setTimeout(() => {
      setSearching(true)
      setSearchError(undefined)
      searchModels(query.trim(), { limit: 12, signal: controller.signal })
        .then((found) => setResults(found))
        .catch((e: Error) => {
          if (!controller.signal.aborted) setSearchError(e.message)
        })
        .finally(() => setSearching(false))
    }, 350)

    return () => {
      clearTimeout(timer.current)
      controller.abort()
    }
  }, [query])

  const grouped = PRESETS.reduce<Record<string, Array<(typeof PRESETS)[number]>>>(
    (acc, preset) => {
      ;(acc[preset.family] ??= []).push(preset)
      return acc
    },
    {},
  )

  return (
    <div className="field">
      <label className="label" htmlFor="model" style={{ position: 'absolute', left: -9999 }}>
        Model
      </label>
      <select id="model" value={modelId} onChange={(e) => onChange(e.target.value)}>
        {!PRESETS.some((p) => p.id === modelId) && <option value={modelId}>{modelId}</option>}
        {Object.entries(grouped).map(([family, group]) => (
          <optgroup key={family} label={family}>
            {group.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>

      <label className="label" htmlFor="hubsearch" style={{ marginTop: 10 }}>
        Or search Hugging Face
      </label>
      <input
        id="hubsearch"
        type="text"
        value={query}
        placeholder="mistralai/Mistral-Small"
        onChange={(e) => setQuery(e.target.value)}
        autoComplete="off"
        spellCheck={false}
      />

      {searching && <p className="small muted">Searching…</p>}
      {searchError && <p className="small muted">Search failed: {searchError}</p>}

      {results.length > 0 && (
        <div
          style={{
            maxHeight: 168,
            overflowY: 'auto',
            border: '1px solid var(--divider)',
            borderRadius: 'var(--radius)',
            marginTop: 6,
          }}
        >
          {results.map((model) => (
            <button
              key={model.id}
              type="button"
              onClick={() => {
                onChange(model.id)
                setQuery('')
                setResults([])
              }}
              style={{
                display: 'block',
                width: '100%',
                textAlign: 'left',
                background: 'transparent',
                border: 0,
                borderBottom: '1px solid var(--divider)',
                padding: '0.5rem 0.625rem',
                minHeight: 'var(--tap)',
                font: 'inherit',
                fontSize: 'var(--t-label)',
                color: 'var(--ink-2)',
                cursor: 'pointer',
              }}
            >
              <span style={{ display: 'block', color: 'var(--ink)' }}>{model.id}</span>
              <span className="small muted">
                {model.parameterCount ? `${(model.parameterCount / 1e9).toFixed(1)}B · ` : ''}
                {model.downloads.toLocaleString()} downloads
                {model.gated ? ' · gated' : ''}
              </span>
            </button>
          ))}
        </div>
      )}

      {loading && <p className="small muted">Reading config.json…</p>}
      {error && (
        <p className="small muted" style={{ color: 'var(--c-net-lat)' }}>
          {error}
        </p>
      )}
      {resolved?.servedBy && resolved.servedBy !== modelId && (
        <p className="small muted">
          Gated. Shapes read from <strong>{resolved.servedBy}</strong>, an ungated mirror with the
          same architecture.
        </p>
      )}
      {resolved?.source === 'hub' && !resolved.servedBy && (
        <p className="small muted">Shapes read from the Hub.</p>
      )}
    </div>
  )
}

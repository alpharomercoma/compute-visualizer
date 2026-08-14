import { useEffect, useState } from 'react'
import { type HuggingFaceConfig, type TransformerSpec, fromHuggingFaceConfig } from '../core/modelspec'
import { HubError, fetchConfig, fetchSafetensors } from './client'
import { presetById } from '../data/presets'

/**
 * Resolves a repository id to a spec, from the bundled catalog or the Hub.
 *
 * Presets answer instantly and work offline; anything else is fetched. Results
 * are cached in localStorage so exploring a dozen models does not spend a dozen
 * requests against an anonymous rate limit shared with everyone behind the same
 * address.
 */

export interface ResolvedModel {
  readonly spec: TransformerSpec
  readonly source: 'preset' | 'hub'
  /** The repo that actually answered, when a gated one was mirrored. */
  readonly servedBy?: string
  readonly storedParameters?: number
}

export interface HubModelState {
  readonly model?: ResolvedModel
  readonly loading: boolean
  readonly error?: string
}

/** The part of a repo id worth showing in a header. */
const shortName = (repo: string): string => repo.split('/').pop() ?? repo

const CACHE_PREFIX = 'cv:config:'
const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000

interface CacheEntry {
  config: HuggingFaceConfig
  servedBy: string
  storedParameters?: number
  at: number
}

function readCache(repo: string): CacheEntry | undefined {
  try {
    const raw = localStorage.getItem(CACHE_PREFIX + repo)
    if (!raw) return undefined
    const entry = JSON.parse(raw) as CacheEntry
    if (Date.now() - entry.at > CACHE_TTL_MS) return undefined
    return entry
  } catch {
    // A corrupt or unavailable cache is not worth failing over; refetch.
    return undefined
  }
}

function writeCache(repo: string, entry: CacheEntry): void {
  try {
    localStorage.setItem(CACHE_PREFIX + repo, JSON.stringify(entry))
  } catch {
    // Private browsing and full quotas both land here. Caching is an
    // optimisation, so losing it silently is correct.
  }
}

export function useHubModel(repo: string, token?: string): HubModelState {
  const [state, setState] = useState<HubModelState>({ loading: false })

  useEffect(() => {
    const preset = presetById(repo)
    if (preset) {
      try {
        setState({
          loading: false,
          model: {
            spec: fromHuggingFaceConfig(preset.config, { id: preset.id, name: preset.label }),
            source: 'preset',
            ...(preset.storedParameters !== undefined
              ? { storedParameters: preset.storedParameters }
              : {}),
          },
        })
      } catch (error) {
        setState({ loading: false, error: (error as Error).message })
      }
      return
    }

    const cached = readCache(repo)
    if (cached) {
      try {
        setState({
          loading: false,
          model: {
            spec: fromHuggingFaceConfig(cached.config, { id: repo, name: shortName(repo) }),
            source: 'hub',
            servedBy: cached.servedBy,
            ...(cached.storedParameters !== undefined
              ? { storedParameters: cached.storedParameters }
              : {}),
          },
        })
        return
      } catch {
        // Fall through and refetch: a cached config we can no longer parse is
        // worse than none.
      }
    }

    const controller = new AbortController()
    setState({ loading: true })

    const options = token ? { token, signal: controller.signal } : { signal: controller.signal }

    void (async () => {
      try {
        const { config, servedBy, mirrored } = await fetchConfig(repo, options)
        // Best-effort: the stored count is useful but not worth failing for.
        let storedParameters: number | undefined
        try {
          storedParameters = (await fetchSafetensors(repo, options))?.total
        } catch {
          storedParameters = undefined
        }

        writeCache(repo, {
          config,
          servedBy,
          at: Date.now(),
          ...(storedParameters !== undefined ? { storedParameters } : {}),
        })

        setState({
          loading: false,
          model: {
            spec: fromHuggingFaceConfig(config, { id: repo, name: shortName(repo) }),
            source: 'hub',
            ...(mirrored ? { servedBy } : {}),
            ...(storedParameters !== undefined ? { storedParameters } : {}),
          },
        })
      } catch (error) {
        if (controller.signal.aborted) return
        const message =
          error instanceof HubError && error.status === 401
            ? `${repo} is gated. Accept its licence on Hugging Face and paste a token, or pick an ungated mirror.`
            : (error as Error).message
        setState({ loading: false, error: message })
      }
    })()

    return () => controller.abort()
  }, [repo, token])

  return state
}

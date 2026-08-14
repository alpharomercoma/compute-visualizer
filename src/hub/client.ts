/**
 * Hugging Face Hub client, running entirely in the browser.
 *
 * No proxy and no server: huggingface.co reflects the request Origin in
 * `Access-Control-Allow-Origin` (with `Vary: Origin`) and sets no
 * `Allow-Credentials`, so a default `credentials: 'omit'` fetch from a static
 * GitHub Pages origin is allowed. This was verified against the live API before
 * the architecture was committed to, because if it had failed the whole design
 * would have needed a build-time catalog instead.
 *
 * Anonymous rate limits are per-IP over a five-minute fixed window: 500 calls to
 * `/api/*` but 3,000 to the resolver. HF's own guidance is to prefer the
 * resolver, so `config.json` is fetched from `/resolve/` and `/api/models` is
 * used only for search and safetensors metadata.
 */

import type { HuggingFaceConfig } from '../core/modelspec'

const HUB = 'https://huggingface.co'

export interface ModelSummary {
  readonly id: string
  readonly downloads: number
  readonly likes: number
  readonly gated: boolean
  readonly parameterCount?: number
  readonly pipelineTag?: string
}

export interface SafetensorsInfo {
  /** Element counts keyed by dtype, as stored in the checkpoint. */
  readonly parameters: Readonly<Record<string, number>>
  readonly total: number
}

export class HubError extends Error {
  readonly status: number
  readonly code: string | undefined

  constructor(message: string, status: number, code?: string) {
    super(message)
    this.name = 'HubError'
    this.status = status
    this.code = code
  }
}

/**
 * Ungated stand-ins for repositories that require accepting a licence.
 *
 * Every `meta-llama/*` repo returns 401 GatedRepo for `config.json` when
 * anonymous, which in practice blocks the most-requested models in the tool.
 * The community mirrors carry byte-identical configs, so a lookup here keeps
 * those models usable without asking anyone for a token.
 */
const UNGATED_MIRRORS: Readonly<Record<string, string>> = {
  'meta-llama/Meta-Llama-3-8B': 'NousResearch/Meta-Llama-3-8B',
  'meta-llama/Meta-Llama-3-8B-Instruct': 'NousResearch/Meta-Llama-3-8B-Instruct',
  'meta-llama/Meta-Llama-3-70B': 'NousResearch/Meta-Llama-3-70B',
  'meta-llama/Meta-Llama-3-70B-Instruct': 'NousResearch/Meta-Llama-3-70B-Instruct',
  'meta-llama/Llama-3.1-8B': 'NousResearch/Meta-Llama-3.1-8B',
  'meta-llama/Llama-3.1-8B-Instruct': 'NousResearch/Meta-Llama-3.1-8B-Instruct',
  'meta-llama/Llama-3.1-70B': 'NousResearch/Meta-Llama-3.1-70B',
  'meta-llama/Llama-3.1-70B-Instruct': 'NousResearch/Meta-Llama-3.1-70B-Instruct',
  'meta-llama/Llama-3.1-405B': 'NousResearch/Meta-Llama-3.1-405B',
  'meta-llama/Llama-2-7b-hf': 'NousResearch/Llama-2-7b-hf',
  'meta-llama/Llama-2-13b-hf': 'NousResearch/Llama-2-13b-hf',
  'meta-llama/Llama-2-70b-hf': 'NousResearch/Llama-2-70b-hf',
  'google/gemma-2-9b': 'unsloth/gemma-2-9b',
  'google/gemma-2-27b': 'unsloth/gemma-2-27b',
}

export const mirrorFor = (repo: string): string | undefined => UNGATED_MIRRORS[repo]

/** Auth header, when the user has supplied a token. */
export interface HubOptions {
  readonly token?: string
  readonly signal?: AbortSignal
}

function headers(options: HubOptions | undefined): HeadersInit | undefined {
  return options?.token ? { Authorization: `Bearer ${options.token}` } : undefined
}

async function request(url: string, options?: HubOptions): Promise<Response> {
  const init: RequestInit = { credentials: 'omit' }
  const h = headers(options)
  if (h) init.headers = h
  if (options?.signal) init.signal = options.signal

  const response = await fetch(url, init)

  if (!response.ok) {
    // HF exposes its error detail through CORS, so the reason a request failed
    // can be shown to the user rather than guessed at.
    const code = response.headers.get('x-error-code') ?? undefined
    const detail = response.headers.get('x-error-message') ?? response.statusText

    if (response.status === 429) {
      const limit = response.headers.get('ratelimit') ?? ''
      const resetMatch = /t=(\d+)/.exec(limit)
      const wait = resetMatch ? ` Try again in ${resetMatch[1]}s.` : ''
      throw new HubError(`Hugging Face rate limit reached.${wait}`, 429, code)
    }
    throw new HubError(detail, response.status, code)
  }

  return response
}

/** Searches the Hub for text-generation models. */
export async function searchModels(
  query: string,
  options?: HubOptions & { limit?: number },
): Promise<ModelSummary[]> {
  const params = new URLSearchParams({
    search: query,
    filter: 'text-generation',
    sort: 'downloads',
    direction: '-1',
    limit: String(options?.limit ?? 20),
  })
  // Asking for expand[] on the LIST endpoint REPLACES the default field set, so
  // every field wanted has to be named explicitly.
  for (const field of ['downloads', 'likes', 'gated', 'safetensors', 'pipeline_tag']) {
    params.append('expand[]', field)
  }

  const response = await request(`${HUB}/api/models?${params}`, options)
  const raw = (await response.json()) as ReadonlyArray<{
    id: string
    downloads?: number
    likes?: number
    gated?: boolean | string
    pipeline_tag?: string
    safetensors?: { total?: number }
  }>

  return raw.map((model) => ({
    id: model.id,
    downloads: model.downloads ?? 0,
    likes: model.likes ?? 0,
    // `gated` is false, "auto" or "manual" -- anything truthy needs acceptance.
    gated: Boolean(model.gated),
    ...(model.safetensors?.total !== undefined
      ? { parameterCount: model.safetensors.total }
      : {}),
    ...(model.pipeline_tag !== undefined ? { pipelineTag: model.pipeline_tag } : {}),
  }))
}

/**
 * Fetches a model's config.json, falling back to an ungated mirror.
 *
 * Returns which repository actually answered, so the UI can say so rather than
 * silently substituting a different model.
 */
export async function fetchConfig(
  repo: string,
  options?: HubOptions,
): Promise<{ config: HuggingFaceConfig; servedBy: string; mirrored: boolean }> {
  try {
    const response = await request(`${HUB}/${repo}/resolve/main/config.json`, options)
    return { config: (await response.json()) as HuggingFaceConfig, servedBy: repo, mirrored: false }
  } catch (error) {
    const gated = error instanceof HubError && (error.status === 401 || error.status === 403)
    const mirror = gated ? mirrorFor(repo) : undefined
    if (!mirror) throw error

    const response = await request(`${HUB}/${mirror}/resolve/main/config.json`, options)
    return { config: (await response.json()) as HuggingFaceConfig, servedBy: mirror, mirrored: true }
  }
}

/**
 * Fetches safetensors metadata: exact stored element counts per dtype.
 *
 * Works anonymously even for gated repositories, which is why it is worth a
 * separate request: it gives a real parameter count for models whose config is
 * otherwise unreachable.
 */
export async function fetchSafetensors(
  repo: string,
  options?: HubOptions,
): Promise<SafetensorsInfo | undefined> {
  const response = await request(
    `${HUB}/api/models/${repo}?expand[]=safetensors`,
    options,
  )
  const body = (await response.json()) as { safetensors?: SafetensorsInfo }
  return body.safetensors
}

/**
 * Reads a safetensors header with two HTTP range requests.
 *
 * The header is a little-endian u64 length followed by that many bytes of JSON
 * describing every tensor's dtype and shape. Both hops survive CORS preflight,
 * so exact shapes are readable from a browser without downloading a multi-
 * gigabyte checkpoint. The OpenWeights Android app does the same trick against
 * GGUF headers to size a model before committing a user's mobile data to it.
 */
export async function fetchTensorShapes(
  repo: string,
  file = 'model.safetensors',
  options?: HubOptions,
): Promise<Record<string, { dtype: string; shape: number[] }>> {
  const url = `${HUB}/${repo}/resolve/main/${file}`

  const lengthInit: RequestInit = { credentials: 'omit', headers: { Range: 'bytes=0-7' } }
  if (options?.token) {
    lengthInit.headers = { ...lengthInit.headers, Authorization: `Bearer ${options.token}` }
  }
  const lengthResponse = await fetch(url, lengthInit)
  if (!lengthResponse.ok) {
    throw new HubError('could not read safetensors header length', lengthResponse.status)
  }

  const headerLength = new DataView(await lengthResponse.arrayBuffer()).getBigUint64(0, true)
  // A header far larger than any real one means we are not looking at a
  // safetensors file, and reading it would be a large pointless download.
  if (headerLength > 100_000_000n) {
    throw new HubError('safetensors header is implausibly large', 0)
  }

  const jsonInit: RequestInit = {
    credentials: 'omit',
    headers: { Range: `bytes=8-${8n + headerLength - 1n}` },
  }
  if (options?.token) {
    jsonInit.headers = { ...jsonInit.headers, Authorization: `Bearer ${options.token}` }
  }
  const jsonResponse = await fetch(url, jsonInit)
  if (!jsonResponse.ok) {
    throw new HubError('could not read safetensors header', jsonResponse.status)
  }

  const header = (await jsonResponse.json()) as Record<
    string,
    { dtype: string; shape: number[] } | unknown
  >
  const shapes: Record<string, { dtype: string; shape: number[] }> = {}
  for (const [name, value] of Object.entries(header)) {
    if (name === '__metadata__') continue
    const entry = value as { dtype?: string; shape?: number[] }
    if (entry.dtype && entry.shape) shapes[name] = { dtype: entry.dtype, shape: entry.shape }
  }
  return shapes
}

import type { ReactNode } from 'react'

/**
 * A collapsed section, named by the question it answers.
 *
 * Two rules, both from watching the previous version fail:
 *
 * 1. The title is a user question ("Why this diagnosis?"), never an
 *    implementation category ("Advanced", "Collectives", "Display"). Nobody
 *    opens "Advanced" because nobody knows what is behind it.
 * 2. Every closed row carries a PREVIEW proving there is something worth
 *    opening — "Best split: TP 2 · DP 4, +13% throughput". A stack of
 *    anonymous chevrons is just quieter overload, and collapsed content that
 *    nobody opens has been deleted with extra steps.
 *
 * Built on <details>, so it works without JavaScript, is keyboard-operable and
 * screen-reader-announced for free, and browser find-in-page can reach inside.
 */

export interface DisclosureProps {
  readonly title: string
  /** One line proving there is something inside. */
  readonly preview?: string
  readonly defaultOpen?: boolean
  readonly children: ReactNode
}

export function Disclosure({ title, preview, defaultOpen = false, children }: DisclosureProps) {
  return (
    <details className="disclosure" open={defaultOpen}>
      <summary>
        <span className="title">{title}</span>
        {preview && <span className="preview">{preview}</span>}
      </summary>
      <div className="body">{children}</div>
    </details>
  )
}

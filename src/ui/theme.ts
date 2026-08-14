import { useCallback, useEffect, useState } from 'react'

/**
 * Theme, as three states rather than two.
 *
 * "System" is a real choice and the honest default for a first visit; light and
 * dark are explicit overrides that persist. The actual DOM flag is set by an
 * inline script in index.html BEFORE first paint — without that, the page
 * renders light and then snaps to dark, which is worse than having no toggle.
 */

export type ThemeChoice = 'light' | 'dark' | 'system'

const STORAGE_KEY = 'cv:theme'

const systemPrefersDark = (): boolean =>
  typeof window !== 'undefined' &&
  window.matchMedia('(prefers-color-scheme: dark)').matches

export function resolveTheme(choice: ThemeChoice): 'light' | 'dark' {
  return choice === 'system' ? (systemPrefersDark() ? 'dark' : 'light') : choice
}

function readStored(): ThemeChoice {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw === 'light' || raw === 'dark' || raw === 'system') return raw
  } catch {
    // Private browsing. A theme preference is not worth failing over.
  }
  return 'light'
}

export function useTheme(): {
  choice: ThemeChoice
  resolved: 'light' | 'dark'
  setChoice: (next: ThemeChoice) => void
} {
  const [choice, setChoiceState] = useState<ThemeChoice>(readStored)
  const [resolved, setResolved] = useState<'light' | 'dark'>(() => resolveTheme(readStored()))

  const apply = useCallback((next: ThemeChoice) => {
    const effective = resolveTheme(next)
    document.documentElement.setAttribute('data-theme', effective)
    setResolved(effective)
  }, [])

  useEffect(() => {
    apply(choice)
  }, [choice, apply])

  // Following the system means following it as it changes, not only at load.
  useEffect(() => {
    if (choice !== 'system') return
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => apply('system')
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [choice, apply])

  const setChoice = useCallback((next: ThemeChoice) => {
    setChoiceState(next)
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // As above.
    }
  }, [])

  return { choice, resolved, setChoice }
}

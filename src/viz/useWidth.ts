/**
 * The element's own width, in CSS pixels.
 *
 * Charts that carry text cannot use a fixed viewBox with a stretched aspect
 * ratio: `preserveAspectRatio="none"` scales the glyphs with the box, so an 11px
 * label becomes 4px on a phone and 19px on a wide monitor. Nor can they letterbox
 * — that was the bug that rendered the old time bar at 36% of its height.
 *
 * Measuring instead lets one SVG unit be one CSS pixel, which makes the type
 * exactly the size the stylesheet says and makes "does this label fit inside its
 * block" an honest question rather than a guess.
 */

import { useEffect, useRef, useState } from 'react'

export function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)

  useEffect(() => {
    const element = ref.current
    if (!element) return

    // Set once up front: the observer fires asynchronously, and a first paint at
    // zero width would collapse every block to its minimum and flash.
    setWidth(element.clientWidth)

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) setWidth(Math.round(entry.contentRect.width))
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  return [ref, width]
}

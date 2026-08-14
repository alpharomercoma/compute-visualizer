/**
 * The hardware in three dimensions, on request only.
 *
 * The previous version rotated on `clock.elapsedTime` and tracked the cursor
 * across the entire viewport with an undamped ±10° parallax, so the board
 * swung whenever you reached for a control on the other side of the screen.
 * That was the "moving and bobbing unintelligently". It also claimed in a
 * comment to be orbitable and never was — there was no OrbitControls and no
 * drei dependency.
 *
 * This one does not move unless you move it. Drag to orbit, and the reset
 * button puts it back. Nothing is bound to wall-clock time.
 */

import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useMemo, useRef, useState } from 'react'
import * as THREE from 'three'

export interface SceneProps {
  readonly gpus: number
  readonly accent: string
  /** Fraction of each GPU's memory in use, 0..1. */
  readonly occupancy: number
  readonly onClose: () => void
}

function resolveColor(value: string, fallback = '#1b94ae'): string {
  if (typeof window === 'undefined') return fallback
  const match = /var\((--[\w-]+)\)/.exec(value)
  if (!match) return value
  return getComputedStyle(document.documentElement).getPropertyValue(match[1]!).trim() || fallback
}

const PACKAGE = 0.68
const PITCH = { x: 1.05, z: 1.2 }

function positions(count: number): Array<[number, number, number]> {
  if (count === 1) return [[0, 0, 0]]
  const columns = Math.ceil(count / 2)
  return Array.from({ length: count }, (_, i) => [
    ((i % columns) - (columns - 1) / 2) * PITCH.x,
    0,
    (Math.floor(i / columns) - 0.5) * PITCH.z,
  ])
}

function Gpu({
  position,
  occupancy,
  accent,
  edge,
}: {
  position: [number, number, number]
  occupancy: number
  accent: string
  edge: string
}) {
  const height = 0.46
  const fill = Math.max(0, Math.min(1, occupancy)) * height
  const outline = useMemo(() => new THREE.BoxGeometry(PACKAGE, height, PACKAGE), [])

  return (
    <group position={position}>
      <lineSegments>
        <edgesGeometry args={[outline]} />
        <lineBasicMaterial color={edge} />
      </lineSegments>
      {fill > 0 && (
        <mesh position={[0, -height / 2 + fill / 2, 0]}>
          <boxGeometry args={[PACKAGE * 0.86, fill, PACKAGE * 0.86]} />
          <meshBasicMaterial color={accent} transparent opacity={0.65} />
        </mesh>
      )}
    </group>
  )
}

/**
 * Drag to orbit. Deliberately not drei's OrbitControls: this needs two axes and
 * a reset, which is thirty lines, against another dependency in a bundle that
 * is already the heaviest thing here.
 */
function Orbit({
  target,
  children,
}: {
  target: { yaw: number; pitch: number }
  children: React.ReactNode
}) {
  const ref = useRef<THREE.Group>(null)
  const { invalidate } = useThree()

  useFrame(() => {
    const group = ref.current
    if (!group) return

    const dy = target.yaw - group.rotation.y
    const dx = target.pitch - group.rotation.x

    // Under frameloop="demand" a frame is only drawn when something asks for
    // one. Easing toward a target therefore has to keep asking, or the board
    // stops partway there — which is what "Reset view" would have looked like.
    if (Math.abs(dy) < 0.0005 && Math.abs(dx) < 0.0005) {
      group.rotation.y = target.yaw
      group.rotation.x = target.pitch
      return
    }

    group.rotation.y += dy * 0.2
    group.rotation.x += dx * 0.2
    invalidate()
  })

  return <group ref={ref}>{children}</group>
}

function Resizer() {
  const { camera, size } = useThree()
  const perspective = camera as THREE.PerspectiveCamera
  perspective.aspect = size.width / Math.max(1, size.height)
  perspective.updateProjectionMatrix()
  return null
}

const HOME = { yaw: 0.6, pitch: 0.5 }

export default function Scene({ gpus, accent, occupancy, onClose }: SceneProps) {
  const [rotation, setRotation] = useState(HOME)
  const dragging = useRef<{ x: number; y: number } | null>(null)
  const points = useMemo(() => positions(gpus), [gpus])

  const color = resolveColor(accent)
  const edge = resolveColor('var(--control-border)', '#868f9d')

  return (
    <div>
      <div
        style={{
          height: 260,
          borderRadius: 8,
          border: '1px solid var(--divider)',
          background: 'var(--panel-sunk)',
          touchAction: 'none',
          cursor: dragging.current ? 'grabbing' : 'grab',
        }}
        onPointerDown={(e) => {
          dragging.current = { x: e.clientX, y: e.clientY }
          e.currentTarget.setPointerCapture(e.pointerId)
        }}
        onPointerMove={(e) => {
          const from = dragging.current
          if (!from) return
          setRotation((prev) => ({
            yaw: prev.yaw + (e.clientX - from.x) * 0.008,
            // Clamped so the board can never tip past its own poles.
            pitch: Math.max(
              -0.2,
              Math.min(1.3, prev.pitch + (e.clientY - from.y) * 0.006),
            ),
          }))
          dragging.current = { x: e.clientX, y: e.clientY }
        }}
        onPointerUp={(e) => {
          dragging.current = null
          e.currentTarget.releasePointerCapture(e.pointerId)
        }}
      >
        <Canvas
          camera={{ position: [0, 2.2, 5.2], fov: 32 }}
          gl={{ antialias: true, alpha: true }}
          dpr={[1, 2]}
          // Renders only when something actually changes, rather than holding a
          // 60fps loop open forever on battery.
          frameloop="demand"
          onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}
          aria-hidden="true"
        >
          <Resizer />
          <Orbit target={rotation}>
            {points.map((position, i) => (
              <Gpu
                key={i}
                position={position}
                occupancy={occupancy}
                accent={color}
                edge={edge}
              />
            ))}
          </Orbit>
        </Canvas>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem' }}>
        <button type="button" className="ghost" onClick={() => setRotation(HOME)}>
          Reset view
        </button>
        <button type="button" className="ghost" onClick={onClose}>
          Hide 3D
        </button>
      </div>
      <p className="small muted" style={{ marginTop: '0.5rem' }}>
        Drag to orbit. Each outline is one GPU; the solid block is its memory in use.
      </p>
    </div>
  )
}

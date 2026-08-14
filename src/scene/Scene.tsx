/**
 * The hardware, as a readout rather than a picture.
 *
 * Every property of this scene is bound to a computed number: each GPU's HBM
 * stacks fill to its actual memory occupancy, the NVLink edges brighten and
 * thicken with the share of the step spent in collectives, and the whole scene
 * takes the hue of the current diagnosis. If it were decorative it would not be
 * worth the ~115 KB three.js costs; because it is an instrument, it is.
 *
 * Loaded lazily so the numbers and the roofline paint first.
 */

import { Canvas, useFrame } from '@react-three/fiber'
import { useMemo, useRef, useState } from 'react'
import * as THREE from 'three'

export interface SceneProps {
  readonly gpus: number
  readonly accent: string
  readonly occupancy: { weights: number; working: number; other: number }
  readonly linkActivity: number
  readonly infeasible: boolean
}

/** Resolves a CSS custom property to something three.js can use. */
function resolveColor(value: string, fallback = '#1b94ae'): string {
  if (typeof window === 'undefined') return fallback
  const match = /var\((--[\w-]+)\)/.exec(value)
  if (!match) return value
  const resolved = getComputedStyle(document.documentElement)
    .getPropertyValue(match[1]!)
    .trim()
  return resolved || fallback
}

/** Package footprint and the pitch between them on the baseboard. */
const PACKAGE = 0.62
const PITCH = { x: 1.02, z: 1.15 }

function positions(count: number): Array<[number, number, number]> {
  if (count === 1) return [[0, 0, 0]]
  // The HGX baseboard layout: two rows of four, which is how NVIDIA draws the
  // NVSwitch fabric too.
  const columns = Math.ceil(count / 2)
  const out: Array<[number, number, number]> = []
  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / columns)
    const column = i % columns
    out.push([(column - (columns - 1) / 2) * PITCH.x, 0, (row - 0.5) * PITCH.z])
  }
  return out
}

function Gpu({
  position,
  occupancy,
  accent,
  infeasible,
}: {
  position: [number, number, number]
  occupancy: SceneProps['occupancy']
  accent: string
  infeasible: boolean
}) {
  const height = 0.5
  const filled = Math.min(1, occupancy.weights + occupancy.working + occupancy.other)

  // Kept faint on purpose. This sits behind a dense readout, and a scene bright
  // enough to compete with the numbers would be working against them.
  const segments = [
    { fraction: occupancy.weights, opacity: 0.18, label: 'weights' },
    { fraction: occupancy.working, opacity: 0.34, label: 'working set' },
  ]

  const outline = useMemo(
    () => new THREE.BoxGeometry(PACKAGE, height, PACKAGE),
    [height],
  )

  let base = 0

  return (
    <group position={position}>
      {/* The package is an outline, not a solid: the fill inside carries the
          quantity and the chassis stays recessive behind the interface. */}
      <lineSegments>
        <edgesGeometry args={[outline]} />
        <lineBasicMaterial
          color={infeasible ? '#cc5490' : '#2a313b'}
          transparent
          opacity={infeasible ? 0.85 : 0.55}
        />
      </lineSegments>

      {/* HBM occupancy, filling from the base. */}
      {segments.map((segment) => {
        const h = Math.max(0, Math.min(segment.fraction, 1)) * height
        const y = -height / 2 + base + h / 2
        base += h
        if (h <= 0) return null
        return (
          <mesh key={segment.label} position={[0, y, 0]}>
            <boxGeometry args={[PACKAGE * 0.9, h, PACKAGE * 0.9]} />
            <meshBasicMaterial color={accent} transparent opacity={segment.opacity} />
          </mesh>
        )
      })}

      {/* A hairline at the fill level: comfortable versus about-to-fail is worth
          seeing without reading a number. Kept faint, because the readout is
          where the actual figure lives. */}
      <mesh position={[0, -height / 2 + filled * height, 0]}>
        <boxGeometry args={[PACKAGE * 0.94, 0.004, PACKAGE * 0.94]} />
        <meshBasicMaterial
          color={filled > 0.9 ? '#cc5490' : '#5a6472'}
          transparent
          opacity={filled > 0.9 ? 0.9 : 0.5}
        />
      </mesh>
    </group>
  )
}

function Links({
  points,
  accent,
  activity,
}: {
  points: Array<[number, number, number]>
  accent: string
  activity: number
}) {
  const ref = useRef<THREE.LineSegments>(null)

  const geometry = useMemo(() => {
    const vertices: number[] = []
    // Every GPU reaches every other through the switch, which is the property
    // that makes an 8-way all-reduce cheap and is worth showing literally.
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        vertices.push(...points[i]!, ...points[j]!)
      }
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
    return geo
  }, [points])

  useFrame(({ clock }) => {
    const material = ref.current?.material as THREE.LineBasicMaterial | undefined
    if (!material) return
    // A slow pulse whose depth tracks how much of the step is communication.
    const pulse = 0.5 + 0.5 * Math.sin(clock.elapsedTime * 1.6)
    material.opacity = 0.08 + activity * (0.25 + 0.35 * pulse)
  })

  if (points.length < 2) return null

  return (
    <lineSegments ref={ref} geometry={geometry}>
      <lineBasicMaterial color={accent} transparent opacity={0.15} />
    </lineSegments>
  )
}

function Rig({ children }: { children: React.ReactNode }) {
  const ref = useRef<THREE.Group>(null)
  const reduced =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches

  useFrame(({ clock, pointer }) => {
    if (!ref.current) return
    // Slow drift plus a little parallax. Deliberately gentle: this sits behind a
    // dense readout, and a scene that demands attention would be competing with
    // the numbers it exists to illustrate.
    const drift = reduced ? 0.6 : clock.elapsedTime * 0.05 + 0.6
    ref.current.rotation.y = drift + pointer.x * 0.18
    ref.current.rotation.x = 0.5 + pointer.y * 0.06
  })
  return <group ref={ref}>{children}</group>
}

export default function Scene({ gpus, accent, occupancy, linkActivity, infeasible }: SceneProps) {
  const [failed, setFailed] = useState(false)
  const points = useMemo(() => positions(gpus), [gpus])
  const color = resolveColor(accent)

  // Without WebGL the tool still works; only the scene is missing. Gating the
  // whole interface on a GPU would be absurd for a page about GPUs.
  if (failed) return null

  return (
    <Canvas
      // Far enough back that the board reads as an object on a bench rather than
      // as a wall behind the interface. The numbers are the subject; this is the
      // thing they describe.
      camera={{ position: [0, 3.6, 9.5], fov: 26 }}
      gl={{ antialias: true, alpha: true }}
      dpr={[1, 2]}
      onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}
      // eslint-disable-next-line react/no-unknown-property
      onError={() => setFailed(true)}
      aria-hidden="true"
    >
      <Rig>
        {points.map((position, index) => (
          <Gpu
            key={index}
            position={position}
            occupancy={occupancy}
            accent={color}
            infeasible={infeasible}
          />
        ))}
        <Links points={points} accent={color} activity={linkActivity} />
      </Rig>
    </Canvas>
  )
}

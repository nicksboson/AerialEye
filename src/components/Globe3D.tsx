import { useRef, useMemo, Suspense, Component, type ReactNode } from 'react'
import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { Sphere, OrbitControls, Stars } from '@react-three/drei'
import * as THREE from 'three'
import { TextureLoader } from 'three'

const EARTH_TEXTURE = 'https://unpkg.com/three-globe@2.31.1/example/img/earth-blue-marble.jpg'
const EARTH_BUMP = 'https://unpkg.com/three-globe@2.31.1/example/img/earth-topology.png'
const EARTH_CLOUDS = 'https://unpkg.com/three-globe@2.31.1/example/img/earth-water.png'

function Earth() {
  const earthRef = useRef<THREE.Mesh>(null)
  const cloudsRef = useRef<THREE.Mesh>(null)

  const [earthTexture, bumpMap, cloudsTexture] = useLoader(TextureLoader, [
    EARTH_TEXTURE,
    EARTH_BUMP,
    EARTH_CLOUDS,
  ])

  earthTexture.colorSpace = THREE.SRGBColorSpace
  cloudsTexture.colorSpace = THREE.SRGBColorSpace

  useFrame(({ clock }) => {
    const elapsed = clock.getElapsedTime()
    if (earthRef.current) earthRef.current.rotation.y = elapsed * 0.05
    if (cloudsRef.current) cloudsRef.current.rotation.y = elapsed * 0.06
  })

  const cityMarkers = useMemo(() => {
    const cities = [
      { lat: 28.6139, lng: 77.209 },
      { lat: 19.076, lng: 72.8777 },
      { lat: 13.0827, lng: 80.2707 },
      { lat: 22.5726, lng: 88.3639 },
      { lat: 12.9716, lng: 77.5946 },
    ]

    return cities.map((city) => {
      const phi = (90 - city.lat) * (Math.PI / 180)
      const theta = (city.lng + 180) * (Math.PI / 180)
      const radius = 2.02

      return new THREE.Vector3(
        -(radius * Math.sin(phi) * Math.cos(theta)),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.sin(theta),
      )
    })
  }, [])

  return (
    <group>
      <Sphere ref={earthRef} args={[2, 64, 64]}>
        <meshStandardMaterial
          map={earthTexture}
          bumpMap={bumpMap}
          bumpScale={0.06}
          roughness={0.7}
          metalness={0.08}
        />
      </Sphere>

      <Sphere args={[2.1, 64, 64]}>
        <meshBasicMaterial
          color="#86c5da"
          transparent
          opacity={0.08}
          side={THREE.BackSide}
        />
      </Sphere>

      <Sphere ref={cloudsRef} args={[2.02, 64, 64]}>
        <meshStandardMaterial
          alphaMap={cloudsTexture}
          transparent
          opacity={0.28}
          color="#ffffff"
          depthWrite={false}
        />
      </Sphere>

      {cityMarkers.map((position, index) => (
        <group key={index} position={position}>
          <mesh>
            <sphereGeometry args={[0.02, 16, 16]} />
            <meshBasicMaterial color="#0d9488" />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.03, 0.05, 32]} />
            <meshBasicMaterial
              color="#0d9488"
              transparent
              opacity={0.5}
              side={THREE.DoubleSide}
            />
          </mesh>
        </group>
      ))}
    </group>
  )
}

function FallbackGlobe() {
  const globeRef = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    if (globeRef.current) globeRef.current.rotation.y = clock.getElapsedTime() * 0.1
  })

  return (
    <group>
      <Sphere ref={globeRef} args={[2, 64, 64]}>
        <meshStandardMaterial color="#1e3a5f" roughness={0.6} metalness={0.2} />
      </Sphere>
      <Sphere args={[2.01, 32, 32]}>
        <meshBasicMaterial color="#22c55e" wireframe transparent opacity={0.3} />
      </Sphere>
      <Sphere args={[2.15, 32, 32]}>
        <meshBasicMaterial color="#93c5fd" transparent opacity={0.1} side={THREE.BackSide} />
      </Sphere>
    </group>
  )
}

class GlobeErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean }> {
  state = { hasError: false }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  render() {
    return this.state.hasError ? <FallbackGlobe /> : this.props.children
  }
}

function Scene() {
  return (
    <>
      <ambientLight intensity={0.55} />
      <directionalLight position={[5, 3, 5]} intensity={1.15} color="#ffffff" />
      <pointLight position={[-5, -3, -5]} intensity={0.25} color="#93c5fd" />

      <GlobeErrorBoundary>
        <Suspense fallback={<FallbackGlobe />}>
          <Earth />
        </Suspense>
      </GlobeErrorBoundary>

      <Stars radius={80} depth={40} count={420} factor={2} saturation={0} fade speed={0.35} />

      <OrbitControls
        enableZoom={false}
        enablePan={false}
        autoRotate={false}
        minPolarAngle={Math.PI / 3}
        maxPolarAngle={Math.PI / 1.5}
      />
    </>
  )
}

export default function Globe3D() {
  return (
    <div className="w-full h-full relative">
      <div className="absolute inset-0 -z-10 rounded-full bg-linear-to-br from-teal-50 via-sky-50 to-stone-50 blur-2xl opacity-70 pointer-events-none" />
      <Canvas
        className="relative z-10"
        camera={{ position: [0, 0, 5.5], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'transparent' }}
        dpr={[1, 1.6]}
      >
        <Scene />
      </Canvas>
    </div>
  )
}

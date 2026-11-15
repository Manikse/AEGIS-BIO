"use client";

import React, { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, RoundedBox, Cylinder, Sphere, Torus, Line } from "@react-three/drei";
import * as THREE from "three";

export type SystemId = "brain" | "heart" | "radiation" | "bones";
export type EnvironmentMood = "calm" | "co2_hazard" | "solar_storm" | "blackout" | "recovery";

interface Astronaut3DProps {
  selectedSystem: SystemId;
  onSelectSystem: (s: SystemId) => void;
  statuses: Record<SystemId, "ok" | "warn" | "crit">;
  xrayMode: boolean;
  lbnpActive: boolean;
  scrubberActive: boolean;
  shieldActive: boolean;
  heartRate: number;
  icp: number;
  effFlux: number;
  boneZ: number;
  unconscious?: boolean;
  mood?: EnvironmentMood;
}

const STATUS_COLOR = {
  ok: "#e2e8f0",
  warn: "#f59e0b",
  crit: "#f43f5e",
};

function DeepSpaceLunarEnvironment({ mood = "calm" }: { mood?: EnvironmentMood }) {
  const farStarsRef = useRef<THREE.Points>(null);
  const midStarsRef = useRef<THREE.Points>(null);
  const stormRef = useRef<THREE.Points>(null);
  const rimRef = useRef<THREE.MeshBasicMaterial>(null);

  const farCount = 1200;
  const midCount = 280;
  const stormCount = 220;

  const farPositions = useMemo(() => {
    const pos = new Float32Array(farCount * 3);
    for (let i = 0; i < farCount; i++) {
      const r = 14 + Math.random() * 24;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = -5 - Math.abs(r * Math.cos(phi));
    }
    return pos;
  }, []);

  const midPositions = useMemo(() => {
    const pos = new Float32Array(midCount * 3);
    for (let i = 0; i < midCount; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 28;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 18;
      pos[i * 3 + 2] = -6 - Math.random() * 12;
    }
    return pos;
  }, []);

  const stormPositions = useMemo(() => {
    const pos = new Float32Array(stormCount * 3);
    for (let i = 0; i < stormCount; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 11;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 7;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 5;
    }
    return pos;
  }, []);

  useFrame((_, delta) => {
    if (farStarsRef.current) farStarsRef.current.rotation.y += delta * 0.005;
    if (midStarsRef.current) midStarsRef.current.rotation.y += delta * 0.012;

    if (stormRef.current && mood === "solar_storm") {
      const attr = stormRef.current.geometry.attributes.position as THREE.BufferAttribute;
      for (let i = 0; i < stormCount; i++) {
        attr.array[i * 3] -= delta * 9.0;
        attr.array[i * 3 + 1] -= delta * 2.2;
        if (attr.array[i * 3] < -5.5) {
          attr.array[i * 3] = 5.5;
          attr.array[i * 3 + 1] = (Math.random() - 0.5) * 7;
        }
      }
      attr.needsUpdate = true;
    }

    if (rimRef.current) {
      const targetCol =
        mood === "solar_storm"
          ? new THREE.Color("#f43f5e")
          : mood === "recovery"
          ? new THREE.Color("#38bdf8")
          : mood === "blackout"
          ? new THREE.Color("#3f3f46")
          : new THREE.Color("#a1a1aa");
      rimRef.current.color.lerp(targetCol, delta * 3.0);
    }
  });

  return (
    <group>
      <points ref={farStarsRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[farPositions, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.026} color="#94a3b8" transparent opacity={0.45} />
      </points>

      <points ref={midStarsRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[midPositions, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.042} color="#f8fafc" transparent opacity={0.68} />
      </points>

      {mood === "solar_storm" && (
        <points ref={stormRef}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[stormPositions, 3]} />
          </bufferGeometry>
          <pointsMaterial size={0.075} color="#fb7185" transparent opacity={0.85} />
        </points>
      )}

      <mesh position={[0, -9.6, -6.2]} rotation={[0, 0, 0.1]}>
        <sphereGeometry args={[8.1, 64, 64]} />
        <meshStandardMaterial
          color={mood === "blackout" ? "#18181b" : "#27272a"}
          emissive="#09090b"
          roughness={0.92}
          metalness={0.08}
        />
      </mesh>

      <mesh position={[0, -9.58, -6.2]} rotation={[0.2, 0.4, 0]}>
        <sphereGeometry args={[8.12, 36, 24]} />
        <meshBasicMaterial color="#52525b" wireframe transparent opacity={0.12} />
      </mesh>

      <mesh position={[0, -9.5, -6.5]}>
        <ringGeometry args={[8.08, 8.26, 96]} />
        <meshBasicMaterial
          ref={rimRef}
          color="#a1a1aa"
          side={THREE.DoubleSide}
          transparent
          opacity={mood === "recovery" ? 0.45 : 0.25}
        />
      </mesh>
    </group>
  );
}

function RealisticAnatomicalAstronaut({
  onSelectSystem,
  statuses,
  xrayMode,
  lbnpActive,
  scrubberActive,
  shieldActive,
  heartRate,
  unconscious,
}: Astronaut3DProps) {
  const groupRef = useRef<THREE.Group>(null);
  const heartRef = useRef<THREE.Group>(null);
  const leftLungRef = useRef<THREE.Mesh>(null);
  const rightLungRef = useRef<THREE.Mesh>(null);
  const o2FlowRef = useRef<THREE.Group>(null);
  const lbnpWaveRef = useRef<THREE.Group>(null);

  useFrame((state, delta) => {
    const t = state.clock.getElapsedTime();

    // Zero-G body floating + unconscious tilt
    if (groupRef.current) {
      const targetRotX = unconscious ? -0.44 : Math.sin(t * 0.5) * 0.04;
      const targetRotZ = unconscious ? 0.26 : Math.cos(t * 0.4) * 0.03;
      const targetY = Math.sin(t * 0.85) * 0.06 - (unconscious ? 0.14 : 0);

      groupRef.current.position.y = THREE.MathUtils.lerp(groupRef.current.position.y, targetY, delta * 3);
      groupRef.current.rotation.x = THREE.MathUtils.lerp(groupRef.current.rotation.x, targetRotX, delta * 3);
      groupRef.current.rotation.z = THREE.MathUtils.lerp(groupRef.current.rotation.z, targetRotZ, delta * 3);
      groupRef.current.rotation.y = Math.sin(t * 0.25) * 0.34;
    }

    // Realistic cardiac systole/diastole contraction
    if (heartRef.current) {
      const freq = (heartRate / 60) * Math.PI * 2;
      const systole = 1 + Math.pow(Math.max(0, Math.sin(t * freq)), 4) * 0.26;
      heartRef.current.scale.set(systole, systole, systole);
    }

    // Realistic bilateral lung respiration (faster when CO2/Heart is stressed)
    const respFreq = statuses.heart === "crit" ? 2.8 : 1.15;
    const breath = 1 + Math.sin(t * respFreq) * 0.085;
    if (leftLungRef.current && rightLungRef.current) {
      leftLungRef.current.scale.set(breath, 1.35 * breath, 0.95 * breath);
      rightLungRef.current.scale.set(breath, 1.35 * breath, 0.95 * breath);
    }

    // Helmet O2 laminar flow ring when scrubber purge is active
    if (o2FlowRef.current && scrubberActive) {
      o2FlowRef.current.rotation.y += delta * 3.5;
    }

    // Sequential Chibis LBNP negative pressure peristaltic wave down the legs
    if (lbnpWaveRef.current && lbnpActive) {
      lbnpWaveRef.current.position.y = -((t * 0.75) % 0.88);
    }
  });

  // When X-Ray is ON, suit shell becomes a subtle technical ghost so internal organs & skeleton stand out clearly
  const suitShellOpacity = xrayMode ? 0.16 : 0.88;
  const organOpacity = xrayMode ? 0.95 : 0.35;

  const brainCol = STATUS_COLOR[statuses.brain];
  const heartCol = STATUS_COLOR[statuses.heart];
  const radCol = STATUS_COLOR[statuses.radiation];
  const boneCol = STATUS_COLOR[statuses.bones];

  return (
    <group ref={groupRef} position={[0, 0.08, 0]}>
      {/* ===================================================================== */}
      {/* 1. HEAD, HELMET, BILATERAL BRAIN HEMISPHERES & OPTIC NERVES (SANS)    */}
      {/* ===================================================================== */}
      <group
        position={[0, 1.44, 0]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectSystem("brain");
        }}
      >
        {/* Outer xEMU Pressure Helmet */}
        <Sphere args={[0.31, 32, 32]}>
          <meshStandardMaterial
            color="#f8fafc"
            roughness={0.2}
            metalness={0.15}
            transparent
            opacity={suitShellOpacity}
          />
        </Sphere>

        {/* Helmet Visor */}
        <Sphere
          args={[0.29, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.52]}
          rotation={[Math.PI * 0.46, 0, 0]}
        >
          <meshStandardMaterial
            color={unconscious ? "#e11d48" : scrubberActive ? "#bae6fd" : "#d4d4d8"}
            roughness={0.1}
            metalness={0.85}
            transparent
            opacity={xrayMode ? 0.14 : 0.82}
          />
        </Sphere>

        {/* Neck Disconnect Ring */}
        <Torus args={[0.21, 0.026, 16, 48]} position={[0, -0.25, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <meshStandardMaterial color="#71717a" metalness={0.7} roughness={0.25} />
        </Torus>

        {/* Helmet Valving O2 Purge Stream (when LiOH scrubber is active) */}
        {scrubberActive && (
          <group ref={o2FlowRef} position={[0, -0.18, 0]}>
            <Torus args={[0.23, 0.009, 12, 48]} rotation={[Math.PI / 2, 0.15, 0]}>
              <meshBasicMaterial color="#38bdf8" />
            </Torus>
          </group>
        )}

        {/* ANATOMY: Left & Right Cerebral Hemispheres */}
        <Sphere args={[0.105, 20, 20]} position={[-0.055, 0.05, -0.01]} scale={[0.92, 1.05, 1.25]}>
          <meshStandardMaterial
            color={brainCol}
            emissive={brainCol}
            emissiveIntensity={statuses.brain === "crit" ? 1.8 : 0.45}
            wireframe
            transparent
            opacity={organOpacity}
          />
        </Sphere>
        <Sphere args={[0.105, 20, 20]} position={[0.055, 0.05, -0.01]} scale={[0.92, 1.05, 1.25]}>
          <meshStandardMaterial
            color={brainCol}
            emissive={brainCol}
            emissiveIntensity={statuses.brain === "crit" ? 1.8 : 0.45}
            wireframe
            transparent
            opacity={organOpacity}
          />
        </Sphere>

        {/* ANATOMY: Brainstem & Cervical Spinal Cord */}
        <Cylinder args={[0.025, 0.018, 0.24, 12]} position={[0, -0.12, -0.03]}>
          <meshStandardMaterial color={brainCol} transparent opacity={organOpacity} />
        </Cylinder>

        {/* ANATOMY: Bilateral Eyes & Optic Nerve Sheaths (Direct SANS target!) */}
        {[-0.065, 0.065].map((ex, i) => (
          <group key={i} position={[ex, 0.01, 0.14]}>
            <Sphere args={[0.028, 14, 14]}>
              <meshStandardMaterial
                color={statuses.brain === "crit" ? "#f43f5e" : "#ffffff"}
                emissive={statuses.brain === "crit" ? "#f43f5e" : "#38bdf8"}
                emissiveIntensity={statuses.brain === "crit" ? 2.0 : 0.6}
              />
            </Sphere>
            {/* Optic Nerve Sheath connecting Eye to Brain */}
            <Cylinder
              args={[0.009, 0.009, 0.1, 8]}
              position={[0, 0.01, -0.05]}
              rotation={[Math.PI / 2, 0, 0]}
            >
              <meshBasicMaterial color={brainCol} />
            </Cylinder>
          </group>
        ))}
      </group>

      {/* ===================================================================== */}
      {/* 2. THORAX, RIBCAGE, LUNGS, HEART, LIVER, KIDNEYS & ASTRORAD VEST      */}
      {/* ===================================================================== */}
      <group
        position={[0, 0.62, 0]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectSystem("heart");
        }}
      >
        {/* Outer Hard Upper Torso (HUT) Suit Shell */}
        <RoundedBox args={[0.74, 0.94, 0.44]} radius={0.14} smoothness={4}>
          <meshStandardMaterial
            color="#f4f4f5"
            roughness={0.35}
            metalness={0.2}
            transparent
            opacity={suitShellOpacity}
          />
        </RoundedBox>

        {/* REAL NASA ARTEMIS ASTRORAD VEST (High-Density Polyethylene Contoured Armor) */}
        {shieldActive && (
          <group position={[0, -0.04, 0]}>
            {/* Main Contoured HDPE Chest/Abdomen Core Shield */}
            <RoundedBox args={[0.77, 0.78, 0.47]} radius={0.07} smoothness={4}>
              <meshStandardMaterial
                color="#1e293b"
                emissive="#0284c7"
                emissiveIntensity={0.25}
                roughness={0.45}
                metalness={0.35}
                transparent
                opacity={0.86}
              />
            </RoundedBox>
            {/* Segmented Polyethylene Marrow Shielding Plates across Sternum & Pelvis */}
            {[0.2, 0.02, -0.16, -0.32].map((py, idx) => (
              <RoundedBox
                key={idx}
                args={[0.66, 0.12, 0.49]}
                radius={0.02}
                position={[0, py, 0]}
              >
                <meshStandardMaterial color="#334155" metalness={0.4} roughness={0.3} />
              </RoundedBox>
            ))}
            {/* Shoulder Yoke Straps */}
            {[-0.24, 0.24].map((sx, idx) => (
              <RoundedBox
                key={idx}
                args={[0.14, 0.22, 0.46]}
                radius={0.03}
                position={[sx, 0.38, 0]}
              >
                <meshStandardMaterial color="#1e293b" metalness={0.4} roughness={0.4} />
              </RoundedBox>
            ))}
          </group>
        )}

        {/* PLSS Life-Support Backpack */}
        <RoundedBox
          args={[0.66, 0.86, 0.28]}
          radius={0.05}
          position={[0, 0.04, -0.32]}
          onClick={(e) => {
            e.stopPropagation();
            onSelectSystem("radiation");
          }}
        >
          <meshStandardMaterial
            color="#d4d4d8"
            metalness={0.35}
            roughness={0.4}
            transparent
            opacity={suitShellOpacity}
          />
        </RoundedBox>

        {/* ================= INTERNAL THORACIC & ABDOMINAL ORGANS ================= */}

        {/* ANATOMY: Trachea & Primary Bronchi */}
        <Cylinder args={[0.022, 0.022, 0.28, 12]} position={[0, 0.38, 0.02]}>
          <meshStandardMaterial color="#94a3b8" transparent opacity={organOpacity} />
        </Cylinder>

        {/* ANATOMY: Breathing Left & Right Lungs (Alveolar O2/CO2 exchange) */}
        <mesh ref={rightLungRef} position={[-0.14, 0.16, 0.02]}>
          <sphereGeometry args={[0.135, 20, 20]} />
          <meshStandardMaterial
            color={statuses.heart === "crit" ? "#fb7185" : scrubberActive ? "#7dd3fc" : "#cbd5e1"}
            emissive={statuses.heart === "crit" ? "#f43f5e" : "#38bdf8"}
            emissiveIntensity={statuses.heart === "crit" ? 0.9 : 0.25}
            transparent
            opacity={organOpacity * 0.72}
          />
        </mesh>
        <mesh ref={leftLungRef} position={[0.14, 0.16, 0.02]}>
          <sphereGeometry args={[0.125, 20, 20]} />
          <meshStandardMaterial
            color={statuses.heart === "crit" ? "#fb7185" : scrubberActive ? "#7dd3fc" : "#cbd5e1"}
            emissive={statuses.heart === "crit" ? "#f43f5e" : "#38bdf8"}
            emissiveIntensity={statuses.heart === "crit" ? 0.9 : 0.25}
            transparent
            opacity={organOpacity * 0.72}
          />
        </mesh>

        {/* ANATOMY: Anatomical Heart + Aortic Arch + Jugular/Carotid Vessels */}
        <group ref={heartRef} position={[0.04, 0.12, 0.08]}>
          {/* Left & Right Ventricles */}
          <Sphere args={[0.085, 20, 20]} scale={[1.0, 1.25, 0.95]} rotation={[0, 0, -0.35]}>
            <meshStandardMaterial
              color={heartCol}
              emissive={heartCol}
              emissiveIntensity={statuses.heart === "crit" ? 2.2 : 1.1}
            />
          </Sphere>
          {/* Aortic Arch */}
          <Torus
            args={[0.055, 0.018, 12, 24, Math.PI]}
            position={[0, 0.08, 0]}
            rotation={[0, 0, 0]}
          >
            <meshStandardMaterial color={heartCol} emissive={heartCol} emissiveIntensity={0.8} />
          </Torus>
        </group>

        {/* Carotid Arteries & Jugular Veins (Cephalad Fluid Shift Pathway to Brain) */}
        <Line
          points={[
            [-0.04, 0.2, 0.06],
            [-0.05, 0.78, 0.02],
          ]}
          color={statuses.brain === "crit" ? "#f43f5e" : "#94a3b8"}
          lineWidth={2.2}
        />
        <Line
          points={[
            [0.04, 0.2, 0.06],
            [0.05, 0.78, 0.02],
          ]}
          color={statuses.brain === "crit" ? "#f43f5e" : "#94a3b8"}
          lineWidth={2.2}
        />

        {/* ANATOMY: Liver (Right Upper Quadrant — high radiosensitivity organ) */}
        <Sphere
          args={[0.13, 18, 18]}
          position={[-0.11, -0.1, 0.04]}
          scale={[1.25, 0.72, 0.85]}
          rotation={[0, 0, 0.2]}
        >
          <meshStandardMaterial
            color={radCol}
            emissive={radCol}
            emissiveIntensity={statuses.radiation === "crit" ? 1.4 : 0.3}
            transparent
            opacity={organOpacity * 0.78}
          />
        </Sphere>

        {/* ANATOMY: Stomach (Left Upper Quadrant) */}
        <Sphere
          args={[0.085, 16, 16]}
          position={[0.11, -0.11, 0.05]}
          scale={[1.0, 0.8, 0.75]}
          rotation={[0, 0, -0.3]}
        >
          <meshStandardMaterial
            color="#a1a1aa"
            transparent
            opacity={organOpacity * 0.65}
          />
        </Sphere>

        {/* ANATOMY: Bilateral Kidneys (Renal Calcium Stone Risk from 0G Bone Resorption) */}
        {[-0.1, 0.1].map((kx, idx) => (
          <Sphere
            key={idx}
            args={[0.048, 14, 14]}
            position={[kx, -0.24, -0.05]}
            scale={[0.75, 1.25, 0.65]}
          >
            <meshStandardMaterial
              color={boneCol}
              emissive={boneCol}
              emissiveIntensity={statuses.bones === "crit" ? 1.5 : 0.4}
              transparent
              opacity={organOpacity * 0.9}
            />
          </Sphere>
        ))}

        {/* ANATOMY: Thoracic Ribcage & Sternum */}
        <Cylinder args={[0.022, 0.022, 0.42, 8]} position={[0, 0.16, 0.18]}>
          <meshStandardMaterial color={boneCol} transparent opacity={organOpacity * 0.8} />
        </Cylinder>
        {[0.3, 0.21, 0.12, 0.03, -0.05].map((ry, rIdx) => (
          <Torus
            key={rIdx}
            args={[0.24 - rIdx * 0.01, 0.01, 8, 32, Math.PI * 1.65]}
            position={[0, ry, 0.01]}
            rotation={[Math.PI / 2 + 0.15, 0, -Math.PI * 0.82]}
          >
            <meshStandardMaterial color={boneCol} transparent opacity={organOpacity * 0.65} />
          </Torus>
        ))}

        {/* ANATOMY: Segmented Vertebral Spine (T1 - L5) */}
        {[0.38, 0.28, 0.18, 0.08, -0.02, -0.12, -0.22, -0.32, -0.42].map((vy, vIdx) => (
          <Cylinder
            key={vIdx}
            args={[0.036, 0.038, 0.065, 12]}
            position={[0, vy, -0.12]}
          >
            <meshStandardMaterial
              color={boneCol}
              emissive={boneCol}
              emissiveIntensity={statuses.bones === "crit" ? 1.2 : 0.25}
              transparent
              opacity={organOpacity * 0.9}
            />
          </Cylinder>
        ))}
      </group>

      {/* ===================================================================== */}
      {/* 3. ARTICULATED UPPER LIMBS (HUMERUS, ELBOW, RADIUS/ULNA)              */}
      {/* ===================================================================== */}
      {[-1, 1].map((dir) => (
        <group
          key={dir}
          position={[dir * 0.48, 0.72, 0]}
          rotation={[unconscious ? -0.35 : 0.08, 0, dir * (unconscious ? -0.42 : -0.12)]}
        >
          {/* Shoulder Bearing */}
          <Sphere args={[0.13, 18, 18]} position={[0, 0.22, 0]}>
            <meshStandardMaterial color="#e4e4e7" transparent opacity={suitShellOpacity} />
          </Sphere>
          {/* Upper Arm + Internal Humerus Bone */}
          <Cylinder args={[0.105, 0.095, 0.42, 20]} position={[0, -0.02, 0]}>
            <meshStandardMaterial color="#fafafa" transparent opacity={suitShellOpacity} />
          </Cylinder>
          <Cylinder args={[0.022, 0.018, 0.72, 10]} position={[0, -0.18, 0]}>
            <meshStandardMaterial color={boneCol} transparent opacity={organOpacity * 0.8} />
          </Cylinder>
          {/* Forearm */}
          <Cylinder args={[0.092, 0.082, 0.38, 20]} position={[0, -0.4, 0.03]} rotation={[0.15, 0, 0]}>
            <meshStandardMaterial color="#f4f4f5" transparent opacity={suitShellOpacity} />
          </Cylinder>
          {/* EVA Glove */}
          <Sphere args={[0.095, 16, 16]} position={[0, -0.64, 0.06]}>
            <meshStandardMaterial color="#71717a" transparent opacity={suitShellOpacity + 0.2} />
          </Sphere>
        </group>
      ))}

      {/* ===================================================================== */}
      {/* 4. PELVIC GIRDLE (MARROW CORE), FEMURS, KNEES & CHIBIS LBNP PANTS     */}
      {/* ===================================================================== */}
      <group
        position={[0, -0.02, 0]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectSystem("bones");
        }}
      >
        {/* Outer Suit Brief / Waist Bearing */}
        <Cylinder args={[0.35, 0.33, 0.28, 28]} position={[0, 0, 0]}>
          <meshStandardMaterial
            color="#e4e4e7"
            transparent
            opacity={suitShellOpacity}
          />
        </Cylinder>

        {/* ANATOMY: Iliac Crest / Pelvic Bone Marrow Ring (Primary Hematopoietic Site) */}
        <Torus args={[0.19, 0.038, 14, 36]} position={[0, 0.02, -0.02]} rotation={[Math.PI / 2.3, 0, 0]}>
          <meshStandardMaterial
            color={statuses.radiation === "crit" ? "#f43f5e" : boneCol}
            emissive={statuses.radiation === "crit" ? "#f43f5e" : boneCol}
            emissiveIntensity={statuses.radiation === "crit" || statuses.bones === "crit" ? 1.4 : 0.35}
            transparent
            opacity={organOpacity * 0.9}
          />
        </Torus>

        {/* Left & Right Lower Limbs (Femur, Knee Joint, Tibia/Fibula + Chibis LBNP) */}
        {[-0.19, 0.19].map((lx, i) => (
          <group key={i} position={[lx, -0.82, 0]}>
            {/* Outer Suit Leg Assembly */}
            <Cylinder args={[0.145, 0.118, 1.32, 24]}>
              <meshStandardMaterial
                color={lbnpActive ? "#0284c7" : "#fafafa"}
                transparent
                opacity={lbnpActive ? 0.38 : suitShellOpacity}
              />
            </Cylinder>

            {/* ANATOMY: Internal Femur, Patella (Knee), and Tibia Bones */}
            <Cylinder args={[0.03, 0.024, 0.58, 12]} position={[0, 0.32, 0]}>
              <meshStandardMaterial
                color={boneCol}
                emissive={boneCol}
                emissiveIntensity={statuses.bones === "crit" ? 1.3 : 0.3}
                transparent
                opacity={organOpacity * 0.92}
              />
            </Cylinder>
            <Sphere args={[0.036, 12, 12]} position={[0, 0.0, 0.01]}>
              <meshStandardMaterial color={boneCol} transparent opacity={organOpacity * 0.92} />
            </Sphere>
            <Cylinder args={[0.024, 0.019, 0.58, 12]} position={[0, -0.32, 0]}>
              <meshStandardMaterial
                color={boneCol}
                emissive={boneCol}
                emissiveIntensity={statuses.bones === "crit" ? 1.3 : 0.3}
                transparent
                opacity={organOpacity * 0.92}
              />
            </Cylinder>

            {/* Femoral Artery & Saphenous Vein (Blood pooled by Chibis LBNP) */}
            <Line
              points={[
                [0, 0.65, 0.03],
                [0, -0.55, 0.03],
              ]}
              color={lbnpActive ? "#38bdf8" : "#f43f5e"}
              lineWidth={2.0}
            />

            {/* REAL NASA / ISS CHIBIS-M LBNP CORRUGATED VACUUM CUFFS */}
            {lbnpActive && (
              <group>
                {[-0.45, -0.2, 0.05, 0.3, 0.5].map((ry, rIdx) => (
                  <Torus
                    key={rIdx}
                    args={[0.152, 0.015, 12, 32]}
                    position={[0, ry, 0]}
                    rotation={[Math.PI / 2, 0, 0]}
                  >
                    <meshStandardMaterial color="#38bdf8" emissive="#0284c7" emissiveIntensity={0.8} />
                  </Torus>
                ))}
                <group ref={i === 0 ? lbnpWaveRef : undefined}>
                  <Torus
                    args={[0.162, 0.02, 12, 32]}
                    position={[0, 0.45, 0]}
                    rotation={[Math.PI / 2, 0, 0]}
                  >
                    <meshBasicMaterial color="#e0f2fe" />
                  </Torus>
                </group>
              </group>
            )}

            {/* Lunar EVA Boots */}
            <RoundedBox args={[0.22, 0.15, 0.34]} radius={0.04} position={[0, -0.72, 0.04]}>
              <meshStandardMaterial color="#52525b" metalness={0.4} roughness={0.5} />
            </RoundedBox>
          </group>
        ))}
      </group>
    </group>
  );
}

export default function BioTwin3D(props: Astronaut3DProps) {
  return (
    <div className="w-full h-full relative select-none">
      <Canvas camera={{ position: [0, 0.18, 4.35], fov: 44 }}>
        <color attach="background" args={["#040508"]} />
        <fog attach="fog" args={["#040508", 10, 30]} />
        <ambientLight intensity={1.05} />
        <directionalLight position={[5, 7, 5]} intensity={1.55} color="#ffffff" />
        <directionalLight
          position={[-5, -4, -3]}
          intensity={props.mood === "solar_storm" ? 1.4 : 0.5}
          color={props.mood === "solar_storm" ? "#f43f5e" : "#a1a1aa"}
        />
        <DeepSpaceLunarEnvironment mood={props.mood} />
        <RealisticAnatomicalAstronaut {...props} />
        <OrbitControls enableZoom={true} enablePan={false} minDistance={2.6} maxDistance={6.0} />
      </Canvas>
    </div>
  );
}
<template>
  <div ref="container" class="viewer-shell" />
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'

const props = defineProps<{ points: number[][] }>()
const container = ref<HTMLElement | null>(null)
let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let animation = 0

function draw() {
  if (!container.value) return
  cleanup()
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x111827)
  camera = new THREE.PerspectiveCamera(60, container.value.clientWidth / 360, 0.01, 1000)
  camera.position.set(2.5, 2.5, 2.5)
  camera.lookAt(0, 0, 0)
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(container.value.clientWidth, 360)
  container.value.innerHTML = ''
  container.value.appendChild(renderer.domElement)

  const flat = new Float32Array(props.points.flat())
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(flat, 3))
  geometry.computeBoundingSphere()
  const sphere = geometry.boundingSphere
  if (sphere) {
    camera.position.set(sphere.radius * 2.4, sphere.radius * 2.1, sphere.radius * 2.4)
    camera.lookAt(sphere.center)
  }
  const material = new THREE.PointsMaterial({ color: 0x5eead4, size: 0.025 })
  scene.add(new THREE.Points(geometry, material))
  scene.add(new THREE.AxesHelper(1))

  const renderLoop = () => {
    animation = window.requestAnimationFrame(renderLoop)
    scene?.rotation.set(0, (Date.now() / 7000) % (Math.PI * 2), 0)
    if (renderer && scene && camera) renderer.render(scene, camera)
  }
  renderLoop()
}

function cleanup() {
  if (animation) window.cancelAnimationFrame(animation)
  renderer?.dispose()
  renderer = null
  scene = null
  camera = null
}

onMounted(draw)
watch(() => props.points, draw, { deep: true })
onBeforeUnmount(cleanup)
</script>

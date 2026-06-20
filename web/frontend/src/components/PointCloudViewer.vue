<template>
  <div ref="container" class="viewer-shell">
    <el-empty v-if="points.length === 0" description="No points to display" style="height: 100%" />
  </div>
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
let resizeObserver: ResizeObserver | null = null
let mesh: THREE.Points | null = null

function draw() {
  if (!container.value || props.points.length === 0) return
  cleanup()
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0f172a)

  const width = container.value.clientWidth || 360
  const height = container.value.clientHeight || 480
  camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 1000)
  camera.position.set(2.5, 2.5, 2.5)
  camera.lookAt(0, 0, 0)

  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(width, height)
  container.value.innerHTML = ''
  container.value.appendChild(renderer.domElement)

  const flat = new Float32Array(props.points.flat())
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(flat, 3))
  geometry.computeBoundingSphere()

  const sphere = geometry.boundingSphere
  if (sphere && sphere.radius > 0) {
    camera.position.set(sphere.radius * 2.4, sphere.radius * 2.1, sphere.radius * 2.4)
    camera.lookAt(sphere.center)
  }

  const material = new THREE.PointsMaterial({ color: 0x5eead4, size: 0.025 })
  mesh = new THREE.Points(geometry, material)
  scene.add(mesh)
  scene.add(new THREE.AxesHelper(1))

  const renderLoop = () => {
    animation = window.requestAnimationFrame(renderLoop)
    scene?.rotation.set(0, (Date.now() / 7000) % (Math.PI * 2), 0)
    if (renderer && scene && camera) renderer.render(scene, camera)
  }
  renderLoop()

  resizeObserver = new ResizeObserver(() => {
    if (!container.value || !renderer || !camera) return
    const w = container.value.clientWidth
    const h = container.value.clientHeight || 480
    if (w === 0 || h === 0) return
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h)
  })
  resizeObserver.observe(container.value)
}

function cleanup() {
  if (animation) window.cancelAnimationFrame(animation)
  if (resizeObserver) resizeObserver.disconnect()
  if (mesh) {
    mesh.geometry.dispose()
    ;(mesh.material as THREE.Material).dispose()
  }
  renderer?.dispose()
  renderer = null
  scene = null
  camera = null
  mesh = null
}

onMounted(draw)
watch(() => props.points, draw, { deep: true })
onBeforeUnmount(cleanup)
</script>

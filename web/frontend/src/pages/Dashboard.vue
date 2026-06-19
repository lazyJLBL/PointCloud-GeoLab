<template>
  <section class="page">
    <div class="surface">
      <h1>Dashboard</h1>
      <p class="muted">
        Experimental reviewer console for the PointCloud-GeoLab geometry core:
        upload, preview, run API-backed tasks, and inspect reproducible artifacts.
      </p>
    </div>
    <div class="form-grid">
      <div class="surface">
        <h3>Datasets</h3>
        <p>{{ datasets.items.length }} uploaded</p>
        <el-button @click="$router.push('/datasets')">Open datasets</el-button>
      </div>
      <div class="surface">
        <h3>Tasks</h3>
        <p>{{ tasks.items.length }} recorded</p>
        <el-button @click="$router.push('/reports')">Open reports</el-button>
      </div>
      <div class="surface">
        <h3>Boundaries</h3>
        <p class="muted">
          Reviewer interface only. No full nonlinear GICP, SLAM backend, CUDA stack,
          PointNet training page, official KITTI benchmark, or production web platform.
        </p>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'

import { useDatasetStore } from '../stores/datasets'
import { useTaskStore } from '../stores/tasks'

const datasets = useDatasetStore()
const tasks = useTaskStore()

onMounted(() => {
  datasets.refresh()
  tasks.refresh()
})
</script>

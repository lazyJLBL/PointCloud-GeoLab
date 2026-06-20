<template>
  <section class="page">
    <div class="surface">
      <h1>Dashboard</h1>
      <p class="muted" style="max-width: 800px">
        Experimental reviewer console for the PointCloud-GeoLab geometry core.
        Upload datasets, preview models, run API-backed tasks, and inspect reproducible artifacts.
      </p>
    </div>

    <div class="form-grid">
      <div class="surface">
        <h3>Workflow</h3>
        <el-steps direction="vertical" :active="2" finish-status="success" style="margin-top: 16px">
          <el-step title="Upload" description="Import .ply, .pcd, .off, or KITTI-like formats." />
          <el-step title="Preview" description="Verify geometry and sampling in 3D viewer." />
          <el-step title="Run Task" description="Execute Registration, Segmentation, or Geometry tasks." />
          <el-step title="Review" description="Inspect metrics and download reproducible artifacts." />
        </el-steps>
      </div>

      <div class="surface" style="display: flex; flex-direction: column; gap: 16px">
        <div>
          <h3>Datasets</h3>
          <p class="muted">
            {{ datasets.items.length === 0 ? 'No datasets uploaded yet.' : `${datasets.items.length} datasets available for processing.` }}
          </p>
          <el-button type="primary" @click="$router.push('/datasets')">
            {{ datasets.items.length === 0 ? 'Upload Dataset' : 'Manage Datasets' }}
          </el-button>
        </div>

        <el-divider />

        <div>
          <h3>Tasks</h3>
          <p class="muted">
            {{ tasks.items.length === 0 ? 'No tasks recorded yet.' : `${tasks.items.length} tasks recorded in history.` }}
          </p>
          <el-button @click="$router.push('/reports')" :disabled="tasks.items.length === 0">
            Open Reports
          </el-button>
        </div>
      </div>

      <div class="surface">
        <h3>Experimental Boundaries</h3>
        <el-alert
          type="warning"
          :closable="false"
          show-icon
          title="Reviewer Interface Only"
          style="margin-bottom: 12px"
        />
        <ul class="muted" style="padding-left: 20px; margin-top: 0">
          <li>No full nonlinear GICP</li>
          <li>No SLAM backend</li>
          <li>No CUDA acceleration stack</li>
          <li>No PointNet training environment</li>
          <li>Not an official KITTI benchmark</li>
          <li>Not a production web platform</li>
        </ul>
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

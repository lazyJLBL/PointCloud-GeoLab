<template>
  <section class="page">
    <div class="surface">
      <h1>Segmentation</h1>
      <p class="muted">Run DBSCAN, Euclidean, region growing, or ground-object segmentation.</p>
    </div>
    <div class="surface form-grid">
      <el-select v-model="datasetId" placeholder="Dataset">
        <el-option v-for="item in datasets.items" :key="item.id" :label="item.filename" :value="item.id" />
      </el-select>
      <el-select v-model="method">
        <el-option label="DBSCAN" value="dbscan" />
        <el-option label="Euclidean" value="euclidean" />
        <el-option label="Region growing" value="region_growing" />
        <el-option label="Ground-object" value="ground-object" />
      </el-select>
      <el-input-number v-model="eps" :min="0.001" :step="0.01" />
      <el-input-number v-model="minPoints" :min="1" />
    </div>
    <el-alert v-if="!canRun" :title="selectionHint" type="info" show-icon :closable="false" />
    <el-button type="primary" :loading="loading" :disabled="!canRun" @click="run">
      Run segmentation
    </el-button>
    <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" />
    <MetricsPanel :metrics="metrics" />
    <ArtifactDownloads v-if="task" :task-id="task.id" :artifacts="task.artifacts" />
    <TaskResultJson :value="task?.result" />
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

import { errorMessage } from '../api/client'
import { createTask } from '../api/tasks'
import ArtifactDownloads from '../components/ArtifactDownloads.vue'
import MetricsPanel from '../components/MetricsPanel.vue'
import TaskResultJson from '../components/TaskResultJson.vue'
import { useDatasetStore } from '../stores/datasets'
import type { TaskRecord } from '../types'

const datasets = useDatasetStore()
const datasetId = ref('')
const method = ref('dbscan')
const eps = ref(0.05)
const minPoints = ref(5)
const loading = ref(false)
const error = ref('')
const task = ref<TaskRecord | null>(null)
const metrics = computed(() => task.value?.result?.metrics as Record<string, unknown> | undefined)
const canRun = computed(() => Boolean(datasetId.value))
const selectionHint = computed(() =>
  datasets.items.length === 0
    ? 'Upload a dataset before running segmentation.'
    : 'Choose a dataset before running segmentation.',
)

onMounted(datasets.refresh)

async function run() {
  if (!canRun.value) {
    error.value = selectionHint.value
    return
  }
  loading.value = true
  error.value = ''
  try {
    const path = method.value === 'ground-object' ? 'segmentation/ground-object' : 'segmentation'
    task.value = await createTask(path, {
      dataset_id: datasetId.value,
      parameters: { method: method.value, eps: eps.value, min_points: minPoints.value },
    })
  } catch (caught) {
    error.value = errorMessage(caught)
  } finally {
    loading.value = false
  }
}
</script>

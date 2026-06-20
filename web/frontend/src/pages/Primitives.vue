<template>
  <section class="page">
    <div class="surface">
      <h1>Primitives</h1>
      <p class="muted">Run plane segmentation, primitive fitting, or sequential extraction.</p>
    </div>

    <div class="surface">
      <h3 style="margin-top: 0">Algorithm Settings</h3>
      <div class="form-grid">
        <el-select v-model="datasetId" placeholder="Select Dataset" style="width: 100%">
          <el-option v-for="item in datasets.items" :key="item.id" :label="item.filename" :value="item.id" />
        </el-select>
        <el-select v-model="action" style="width: 100%">
          <el-option label="Plane Segmentation" value="primitives/plane" />
          <el-option label="Primitive Fitting" value="primitives/fit" />
          <el-option label="Sequential Extraction" value="primitives/extract" />
        </el-select>
        <el-select v-model="model" style="width: 100%">
          <el-option label="Plane" value="plane" />
          <el-option label="Sphere" value="sphere" />
          <el-option label="Cylinder" value="cylinder" />
        </el-select>
        <div>
          <span class="muted text-sm" style="display: block; margin-bottom: 4px">Distance Threshold:</span>
          <el-input-number v-model="threshold" :min="0.001" :step="0.01" style="width: 100%" />
        </div>
      </div>
    </div>

    <el-alert v-if="!canRun" :title="selectionHint" type="info" show-icon :closable="false" />
    <el-button type="primary" size="large" :loading="loading" :disabled="!canRun" @click="run">
      Run Primitive Task
    </el-button>
    <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" />

    <MetricsPanel v-if="task" :metrics="metrics" />
    <ArtifactDownloads v-if="task" :task-id="task.id" :artifacts="task.artifacts" />
    <TaskResultJson v-if="task" :value="task.result" />
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
const action = ref('primitives/plane')
const model = ref('plane')
const threshold = ref(0.02)
const loading = ref(false)
const error = ref('')
const task = ref<TaskRecord | null>(null)

const metrics = computed(() => task.value?.result?.metrics as Record<string, unknown> | undefined)
const canRun = computed(() => Boolean(datasetId.value))
const selectionHint = computed(() =>
  datasets.items.length === 0
    ? 'Upload a dataset before running primitive fitting.'
    : 'Choose a dataset before running primitive fitting.',
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
    task.value = await createTask(action.value, {
      dataset_id: datasetId.value,
      parameters: { model: model.value, threshold: threshold.value },
    })
  } catch (caught) {
    error.value = errorMessage(caught)
  } finally {
    loading.value = false
  }
}
</script>

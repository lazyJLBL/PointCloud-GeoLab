<template>
  <section class="page">
    <div class="surface">
      <h1>Primitives</h1>
      <p class="muted">Run plane segmentation, primitive fitting, or sequential extraction.</p>
    </div>
    <div class="surface form-grid">
      <el-select v-model="datasetId" placeholder="Dataset">
        <el-option v-for="item in datasets.items" :key="item.id" :label="item.filename" :value="item.id" />
      </el-select>
      <el-select v-model="action">
        <el-option label="Plane" value="primitives/plane" />
        <el-option label="Fit" value="primitives/fit" />
        <el-option label="Extract" value="primitives/extract" />
      </el-select>
      <el-select v-model="model">
        <el-option label="Plane" value="plane" />
        <el-option label="Sphere" value="sphere" />
        <el-option label="Cylinder" value="cylinder" />
      </el-select>
      <el-input-number v-model="threshold" :min="0.001" :step="0.01" />
    </div>
    <el-button type="primary" :loading="loading" @click="run">Run primitive task</el-button>
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
const action = ref('primitives/plane')
const model = ref('plane')
const threshold = ref(0.02)
const loading = ref(false)
const error = ref('')
const task = ref<TaskRecord | null>(null)
const metrics = computed(() => task.value?.result?.metrics as Record<string, unknown> | undefined)

onMounted(datasets.refresh)

async function run() {
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

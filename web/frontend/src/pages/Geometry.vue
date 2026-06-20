<template>
  <section class="page">
    <div class="surface">
      <h1>Geometry</h1>
      <p class="muted">Compute AABB, OBB, and PCA metrics for one dataset.</p>
    </div>

    <div class="surface">
      <h3 style="margin-top: 0">Algorithm Settings</h3>
      <div class="form-grid">
        <el-select v-model="datasetId" placeholder="Select Dataset" style="width: 100%">
          <el-option v-for="item in datasets.items" :key="item.id" :label="item.filename" :value="item.id" />
        </el-select>
      </div>
    </div>

    <el-alert v-if="!canRun" :title="selectionHint" type="info" show-icon :closable="false" />
    <el-button type="primary" size="large" :loading="loading" :disabled="!canRun" @click="run">
      Run Geometry Analysis Task
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
const loading = ref(false)
const error = ref('')
const task = ref<TaskRecord | null>(null)

const metrics = computed(() => task.value?.result?.metrics as Record<string, unknown> | undefined)
const canRun = computed(() => Boolean(datasetId.value))
const selectionHint = computed(() =>
  datasets.items.length === 0
    ? 'Upload a dataset before running geometry analysis.'
    : 'Choose a dataset before running geometry analysis.',
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
    task.value = await createTask('geometry', { dataset_id: datasetId.value, parameters: {} })
  } catch (caught) {
    error.value = errorMessage(caught)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="page">
    <div class="surface">
      <h1>Preprocessing</h1>
      <p class="muted">Run voxel downsampling, radius filtering, normalization, and sampling.</p>
    </div>
    <div class="surface form-grid">
      <el-select v-model="datasetId" placeholder="Dataset">
        <el-option v-for="item in datasets.items" :key="item.id" :label="item.filename" :value="item.id" />
      </el-select>
      <el-input-number v-model="params.voxel_size" :min="0" :step="0.01" placeholder="voxel_size" />
      <el-input-number v-model="params.radius" :min="0" :step="0.01" placeholder="radius" />
      <el-input-number v-model="params.sample_count" :min="0" :step="100" placeholder="sample_count" />
      <el-switch v-model="params.normalize" active-text="Normalize" />
    </div>
    <el-alert v-if="!canRun" :title="selectionHint" type="info" show-icon :closable="false" />
    <el-button type="primary" :loading="loading" :disabled="!canRun" @click="run">
      Run preprocessing
    </el-button>
    <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" />
    <MetricsPanel :metrics="metrics" />
    <TaskResultJson :value="task?.result" />
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'

import { errorMessage } from '../api/client'
import { createTask } from '../api/tasks'
import MetricsPanel from '../components/MetricsPanel.vue'
import TaskResultJson from '../components/TaskResultJson.vue'
import { useDatasetStore } from '../stores/datasets'
import type { TaskRecord } from '../types'

const datasets = useDatasetStore()
const datasetId = ref('')
const loading = ref(false)
const error = ref('')
const task = ref<TaskRecord | null>(null)
const params = reactive({ voxel_size: 0, radius: 0, sample_count: 0, normalize: false })
const metrics = computed(() => task.value?.result?.metrics as Record<string, unknown> | undefined)
const canRun = computed(() => Boolean(datasetId.value))
const selectionHint = computed(() =>
  datasets.items.length === 0
    ? 'Upload a dataset before running preprocessing.'
    : 'Choose a dataset before running preprocessing.',
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
    task.value = await createTask('preprocessing', {
      dataset_id: datasetId.value,
      parameters: { ...params, sample_count: params.sample_count || null },
    })
  } catch (caught) {
    error.value = errorMessage(caught)
  } finally {
    loading.value = false
  }
}
</script>

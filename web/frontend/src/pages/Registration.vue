<template>
  <section class="page">
    <div class="surface">
      <h1>Registration</h1>
      <p class="muted">Run ICP, robust ICP, or multiscale ICP between two uploaded datasets.</p>
    </div>

    <div class="surface">
      <h3 style="margin-top: 0">Algorithm Settings</h3>
      <div class="form-grid">
        <el-select v-model="sourceId" placeholder="Select Source Dataset" style="width: 100%">
          <el-option v-for="item in datasets.items" :key="item.id" :label="item.filename" :value="item.id" />
        </el-select>
        <el-select v-model="targetId" placeholder="Select Target Dataset" style="width: 100%">
          <el-option v-for="item in datasets.items" :key="item.id" :label="item.filename" :value="item.id" />
        </el-select>
        <el-select v-model="mode" style="width: 100%">
          <el-option label="ICP" value="registration/icp" />
          <el-option label="Robust ICP" value="registration/robust-icp" />
          <el-option label="Multiscale ICP" value="registration/multiscale-icp" />
        </el-select>
        <div>
          <span class="muted text-sm" style="display: block; margin-bottom: 4px">Max Iterations:</span>
          <el-input-number v-model="maxIterations" :min="1" :max="200" style="width: 100%" />
        </div>
      </div>
    </div>

    <el-alert v-if="!canRun" :title="selectionHint" type="info" show-icon :closable="false" />
    <el-button type="primary" size="large" :loading="loading" :disabled="!canRun" @click="run">
      Run Registration Task
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
const sourceId = ref('')
const targetId = ref('')
const mode = ref('registration/icp')
const maxIterations = ref(50)
const loading = ref(false)
const error = ref('')
const task = ref<TaskRecord | null>(null)

const metrics = computed(() => task.value?.result?.metrics as Record<string, unknown> | undefined)
const canRun = computed(() => Boolean(sourceId.value && targetId.value))
const selectionHint = computed(() =>
  datasets.items.length < 2
    ? 'Upload at least two datasets before running registration.'
    : 'Choose both source and target datasets before running registration.',
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
    task.value = await createTask(mode.value, {
      source_dataset_id: sourceId.value,
      target_dataset_id: targetId.value,
      parameters: { max_iterations: maxIterations.value, max_iterations_per_level: maxIterations.value },
    })
  } catch (caught) {
    error.value = errorMessage(caught)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="page">
    <div class="surface">
      <h1>Registration</h1>
      <p class="muted">Run ICP, robust ICP, or multiscale ICP between two uploaded datasets.</p>
    </div>
    <div class="surface form-grid">
      <el-select v-model="sourceId" placeholder="Source">
        <el-option v-for="item in datasets.items" :key="item.id" :label="item.filename" :value="item.id" />
      </el-select>
      <el-select v-model="targetId" placeholder="Target">
        <el-option v-for="item in datasets.items" :key="item.id" :label="item.filename" :value="item.id" />
      </el-select>
      <el-select v-model="mode">
        <el-option label="ICP" value="registration/icp" />
        <el-option label="Robust ICP" value="registration/robust-icp" />
        <el-option label="Multiscale ICP" value="registration/multiscale-icp" />
      </el-select>
      <el-input-number v-model="maxIterations" :min="1" :max="200" />
    </div>
    <el-button type="primary" :loading="loading" @click="run">Run registration</el-button>
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
const sourceId = ref('')
const targetId = ref('')
const mode = ref('registration/icp')
const maxIterations = ref(50)
const loading = ref(false)
const error = ref('')
const task = ref<TaskRecord | null>(null)
const metrics = computed(() => task.value?.result?.metrics as Record<string, unknown> | undefined)

onMounted(datasets.refresh)

async function run() {
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

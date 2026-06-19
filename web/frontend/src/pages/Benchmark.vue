<template>
  <section class="page">
    <div class="surface">
      <h1>Benchmark</h1>
      <p class="muted">
        Web benchmarks always use quick mode. Timing and memory metadata are local
        machine references, not portable performance claims.
      </p>
    </div>
    <div class="surface form-grid">
      <el-select v-model="suite">
        <el-option label="All" value="all" />
        <el-option label="KDTree" value="kdtree" />
        <el-option label="ICP" value="icp" />
        <el-option label="RANSAC" value="ransac" />
        <el-option label="Registration" value="registration" />
        <el-option label="GICP-style ICP" value="gicp" />
        <el-option label="Segmentation" value="segmentation" />
      </el-select>
      <el-input-number v-model="repeat" :min="1" :max="3" />
    </div>
    <el-button type="primary" :loading="loading" @click="run">Run quick benchmark</el-button>
    <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" />
    <MetricsPanel :metrics="metrics" />
    <ArtifactDownloads v-if="task" :task-id="task.id" :artifacts="task.artifacts" />
    <TaskResultJson :value="task?.result" />
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

import { errorMessage } from '../api/client'
import { createTask } from '../api/tasks'
import ArtifactDownloads from '../components/ArtifactDownloads.vue'
import MetricsPanel from '../components/MetricsPanel.vue'
import TaskResultJson from '../components/TaskResultJson.vue'
import type { TaskRecord } from '../types'

const suite = ref('all')
const repeat = ref(1)
const loading = ref(false)
const error = ref('')
const task = ref<TaskRecord | null>(null)
const metrics = computed(() => task.value?.result?.metrics as Record<string, unknown> | undefined)

async function run() {
  loading.value = true
  error.value = ''
  try {
    task.value = await createTask('benchmark', {
      parameters: { suite: suite.value, quick: true, repeat: repeat.value },
    })
  } catch (caught) {
    error.value = errorMessage(caught)
  } finally {
    loading.value = false
  }
}
</script>

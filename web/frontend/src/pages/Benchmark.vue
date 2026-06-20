<template>
  <section class="page">
    <div class="surface">
      <h1>Benchmark</h1>
      <p class="muted">
        Run algorithm benchmarks on synthetic datasets. Web benchmarks always use <strong>quick</strong> mode to prevent timeouts.
      </p>
      <el-alert
        type="warning"
        :closable="false"
        show-icon
        title="Local References Only"
        style="margin-top: 12px"
      >
        Timing and memory metadata are local machine references and do not represent portable performance claims or official KITTI results.
      </el-alert>
    </div>

    <div class="surface">
      <h3 style="margin-top: 0">Benchmark Configuration</h3>
      <div class="form-grid">
        <el-select v-model="suite" style="width: 100%">
          <el-option label="All Suites" value="all" />
          <el-option label="KDTree Suite" value="kdtree" />
          <el-option label="ICP Suite" value="icp" />
          <el-option label="RANSAC Suite" value="ransac" />
          <el-option label="Registration Suite" value="registration" />
          <el-option label="GICP-style ICP Suite" value="gicp" />
          <el-option label="Segmentation Suite" value="segmentation" />
        </el-select>
        <div>
          <span class="muted text-sm" style="display: block; margin-bottom: 4px">Repeat Count:</span>
          <el-input-number v-model="repeat" :min="1" :max="3" style="width: 100%" />
        </div>
      </div>
    </div>

    <el-button type="primary" size="large" :loading="loading" @click="run">
      Run Quick Benchmark
    </el-button>
    <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" />

    <MetricsPanel v-if="task" :metrics="metrics" />
    <ArtifactDownloads v-if="task" :task-id="task.id" :artifacts="task.artifacts" />
    <TaskResultJson v-if="task" :value="task.result" />
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

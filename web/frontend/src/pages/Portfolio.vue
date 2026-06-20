<template>
  <section class="page">
    <div class="surface">
      <h1>Portfolio</h1>
      <p class="muted" style="max-width: 800px">
        Run the existing portfolio verification workflow and collect generated reports under the
        task artifacts directory. Note that the generation task may take a longer time depending on the available compute.
      </p>
      <el-alert
        type="info"
        :closable="false"
        show-icon
        title="Long Running Task"
        style="margin-top: 12px"
      >
        Web tasks currently run synchronously and may block until the generation is completed.
      </el-alert>
    </div>

    <el-button type="primary" size="large" :loading="loading" @click="run">
      Generate Portfolio Report
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

const loading = ref(false)
const error = ref('')
const task = ref<TaskRecord | null>(null)

const metrics = computed(() => task.value?.result?.metrics as Record<string, unknown> | undefined)

async function run() {
  loading.value = true
  error.value = ''
  try {
    task.value = await createTask('portfolio', { parameters: { quick: true } })
  } catch (caught) {
    error.value = errorMessage(caught)
  } finally {
    loading.value = false
  }
}
</script>

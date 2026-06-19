<template>
  <section class="page">
    <div class="surface">
      <h1>Portfolio</h1>
      <p class="muted">
        Run the existing portfolio verification workflow and collect generated reports under the
        task artifacts directory.
      </p>
    </div>
    <el-button type="primary" :loading="loading" @click="run">Generate portfolio report</el-button>
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

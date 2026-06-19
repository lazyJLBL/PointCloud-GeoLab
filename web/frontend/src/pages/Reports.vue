<template>
  <section class="page">
    <div class="surface">
      <h1>Reports</h1>
      <p class="muted">Review task status, metrics summaries, artifacts, and raw TaskResult JSON.</p>
    </div>
    <el-button :loading="tasks.loading" @click="tasks.refresh()">Refresh</el-button>
    <el-alert v-if="tasks.error" :title="tasks.error" type="error" show-icon :closable="false" />
    <div v-for="task in tasks.items" :key="task.id" class="surface">
      <div class="toolbar">
        <strong>{{ task.task_type }}</strong>
        <TaskStatusBadge :status="task.status" />
        <span class="muted">{{ task.updated_at }}</span>
      </div>
      <MetricsPanel :metrics="task.result?.metrics as Record<string, unknown> | undefined" />
      <ArtifactDownloads :task-id="task.id" :artifacts="task.artifacts" />
      <TaskResultJson :value="task.result" />
    </div>
  </section>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'

import ArtifactDownloads from '../components/ArtifactDownloads.vue'
import MetricsPanel from '../components/MetricsPanel.vue'
import TaskResultJson from '../components/TaskResultJson.vue'
import TaskStatusBadge from '../components/TaskStatusBadge.vue'
import { useTaskStore } from '../stores/tasks'

const tasks = useTaskStore()

onMounted(tasks.refresh)
</script>

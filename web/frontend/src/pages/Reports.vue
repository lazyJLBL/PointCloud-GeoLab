<template>
  <section class="page">
    <div class="surface">
      <div class="toolbar" style="justify-content: space-between">
        <div>
          <h1>Reports</h1>
          <p class="muted" style="margin-top: 8px">Review task status, metrics summaries, artifacts, and raw TaskResult JSON.</p>
        </div>
        <el-button :loading="tasks.loading" @click="tasks.refresh()">Refresh Reports</el-button>
      </div>
    </div>

    <div class="surface" style="display: flex; gap: 16px; align-items: center">
      <strong class="muted">Filter Status:</strong>
      <el-radio-group v-model="filterStatus">
        <el-radio-button label="all">All</el-radio-button>
        <el-radio-button label="completed">Completed</el-radio-button>
        <el-radio-button label="failed">Failed</el-radio-button>
      </el-radio-group>
    </div>

    <el-alert v-if="tasks.error" :title="tasks.error" type="error" show-icon :closable="false" />

    <el-empty v-if="filteredTasks.length === 0" description="No tasks match the filter" class="surface" />

    <div v-for="task in filteredTasks" :key="task.id" class="surface">
      <div class="toolbar" style="margin-bottom: 16px">
        <strong style="font-size: 16px; color: #0f172a">{{ task.task_type }}</strong>
        <TaskStatusBadge :status="task.status" />
        <span class="muted text-sm">{{ task.updated_at }}</span>
      </div>
      <MetricsPanel :metrics="task.result?.metrics as Record<string, unknown> | undefined" />
      <ArtifactDownloads :task-id="task.id" :artifacts="task.artifacts" />
      <TaskResultJson :value="task.result" />
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

import ArtifactDownloads from '../components/ArtifactDownloads.vue'
import MetricsPanel from '../components/MetricsPanel.vue'
import TaskResultJson from '../components/TaskResultJson.vue'
import TaskStatusBadge from '../components/TaskStatusBadge.vue'
import { useTaskStore } from '../stores/tasks'

const tasks = useTaskStore()
const filterStatus = ref('all')

const filteredTasks = computed(() => {
  if (filterStatus.value === 'all') return tasks.items
  return tasks.items.filter((t) => t.status === filterStatus.value)
})

onMounted(tasks.refresh)
</script>

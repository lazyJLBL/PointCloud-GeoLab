<template>
  <div class="surface">
    <h3>Artifacts</h3>
    <el-empty v-if="items.length === 0" description="No artifacts yet" />
    <div v-else class="toolbar">
      <el-button
        v-for="[label, name] in items"
        :key="label"
        tag="a"
        :href="url(name)"
        target="_blank"
        type="default"
      >
        Download {{ label }}
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import { artifactUrl } from '../api/tasks'

const props = defineProps<{ taskId: string; artifacts?: Record<string, string> }>()

const items = computed(() => Object.entries(props.artifacts ?? {}))

function url(name: string) {
  return artifactUrl(props.taskId, name)
}
</script>

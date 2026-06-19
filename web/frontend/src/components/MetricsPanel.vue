<template>
  <div class="surface">
    <h3>Metrics</h3>
    <el-empty v-if="entries.length === 0" description="No metrics yet" />
    <el-descriptions v-else :column="2" border>
      <el-descriptions-item v-for="[key, value] in entries" :key="key" :label="key">
        {{ value }}
      </el-descriptions-item>
    </el-descriptions>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ metrics?: Record<string, unknown> | null }>()

const entries = computed(() => Object.entries(props.metrics ?? {}))
</script>

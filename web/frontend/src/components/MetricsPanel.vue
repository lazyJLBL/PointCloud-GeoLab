<template>
  <div class="surface">
    <h3>Metrics</h3>
    <el-empty v-if="entries.length === 0" description="No metrics yet" />
    <el-descriptions v-else :column="2" border>
      <el-descriptions-item v-for="[key, value] in entries" :key="key" :label="key">
        {{ formatValue(value) }}
      </el-descriptions-item>
    </el-descriptions>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ metrics?: Record<string, unknown> | null }>()

const entries = computed(() => Object.entries(props.metrics ?? {}))

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'object') {
    if (Array.isArray(value)) {
      return `[${value.map((v) => (typeof v === 'number' ? v.toFixed(4) : String(v))).join(', ')}]`
    }
    return JSON.stringify(value)
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(6)
  }
  return String(value)
}
</script>

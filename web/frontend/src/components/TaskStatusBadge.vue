<template>
  <el-tag :type="tagType" effect="light">{{ status }}</el-tag>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import type { TaskStatus } from '../types'

const props = defineProps<{ status: TaskStatus }>()

const tagType = computed(() => {
  if (props.status === 'completed') return 'success'
  if (props.status === 'failed') return 'danger'
  if (props.status === 'running') return 'warning'
  return 'info'
})
</script>

<template>
  <div class="surface">
    <div class="toolbar">
      <input ref="input" type="file" accept=".ply,.pcd,.xyz,.txt,.bin,.off" @change="pickFile" />
      <el-button type="primary" :loading="loading" :disabled="!file" @click="submit">
        Upload
      </el-button>
    </div>
    <el-alert
      v-if="error"
      :title="error"
      type="error"
      show-icon
      :closable="false"
      style="margin-top: 12px"
    />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

import { errorMessage } from '../api/client'
import { uploadDataset } from '../api/datasets'
import type { DatasetRecord } from '../types'

const emit = defineEmits<{ uploaded: [record: DatasetRecord] }>()
const input = ref<HTMLInputElement | null>(null)
const file = ref<File | null>(null)
const loading = ref(false)
const error = ref('')

function pickFile(event: Event) {
  const target = event.target as HTMLInputElement
  file.value = target.files?.[0] ?? null
  error.value = ''
}

async function submit() {
  if (!file.value) return
  loading.value = true
  error.value = ''
  try {
    emit('uploaded', await uploadDataset(file.value))
    file.value = null
    if (input.value) input.value.value = ''
  } catch (caught) {
    error.value = errorMessage(caught)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="page">
    <div class="surface">
      <h1>Datasets</h1>
      <p class="muted">
        Upload .ply, .pcd, .xyz, .txt, KITTI-like .bin, or ModelNet-like .off files.
      </p>
    </div>
    <FileUploader @uploaded="afterUpload" />
    <el-alert v-if="store.error" :title="store.error" type="error" show-icon :closable="false" />
    <div class="surface">
      <el-table v-loading="store.loading" :data="store.items">
        <el-table-column prop="filename" label="File" />
        <el-table-column prop="extension" label="Format" width="110" />
        <el-table-column prop="size_bytes" label="Bytes" width="120" />
        <el-table-column label="Actions" width="220">
          <template #default="{ row }">
            <el-button size="small" @click="store.loadPreview(row.id)">Preview</el-button>
            <el-button size="small" type="danger" @click="store.remove(row.id)">Delete</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <PointCloudViewer v-if="store.preview" :points="store.preview.points" />
  </section>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'

import FileUploader from '../components/FileUploader.vue'
import PointCloudViewer from '../components/PointCloudViewer.vue'
import { useDatasetStore } from '../stores/datasets'
import type { DatasetRecord } from '../types'

const store = useDatasetStore()

onMounted(store.refresh)

async function afterUpload(record: DatasetRecord) {
  await store.refresh()
  await store.loadPreview(record.id)
}
</script>

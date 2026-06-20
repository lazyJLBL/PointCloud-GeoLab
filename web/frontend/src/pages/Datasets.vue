<template>
  <section class="page">
    <div class="surface">
      <h1>Datasets</h1>
      <p class="muted">
        Manage PointCloud-GeoLab datasets. Uploaded files are processed by the backend and cached for tasks.
      </p>
    </div>
    <FileUploader @uploaded="afterUpload" />
    <el-alert v-if="store.error" :title="store.error" type="error" show-icon :closable="false" />
    <div class="surface">
      <el-table v-loading="store.loading" :data="store.items" empty-text="No datasets uploaded yet">
        <el-table-column prop="filename" label="File" />
        <el-table-column prop="extension" label="Format" width="110" />
        <el-table-column prop="size_bytes" label="Bytes" width="120" />
        <el-table-column label="Actions" width="220">
          <template #default="{ row }">
            <el-button size="small" @click="store.loadPreview(row.id)">Preview</el-button>
            <el-popconfirm title="Are you sure you want to delete this dataset?" @confirm="store.remove(row.id)">
              <template #reference>
                <el-button size="small" type="danger">Delete</el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div v-if="store.preview" class="surface">
      <div class="toolbar" style="margin-bottom: 16px; justify-content: space-between">
        <h3>Preview</h3>
        <span class="muted text-sm">
          Displaying {{ store.preview.points.length }} points
        </span>
      </div>
      <PointCloudViewer :points="store.preview.points" />
    </div>
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

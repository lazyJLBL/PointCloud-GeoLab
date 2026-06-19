import { defineStore } from 'pinia'

import { errorMessage } from '../api/client'
import { deleteDataset, getDatasetPreview, listDatasets } from '../api/datasets'
import type { DatasetPreview, DatasetRecord } from '../types'

export const useDatasetStore = defineStore('datasets', {
  state: () => ({
    items: [] as DatasetRecord[],
    preview: null as DatasetPreview | null,
    loading: false,
    error: '',
  }),
  actions: {
    async refresh() {
      this.loading = true
      this.error = ''
      try {
        this.items = await listDatasets()
      } catch (error) {
        this.error = errorMessage(error)
      } finally {
        this.loading = false
      }
    },
    async loadPreview(datasetId: string) {
      this.loading = true
      this.error = ''
      try {
        this.preview = await getDatasetPreview(datasetId)
      } catch (error) {
        this.error = errorMessage(error)
      } finally {
        this.loading = false
      }
    },
    async remove(datasetId: string) {
      await deleteDataset(datasetId)
      await this.refresh()
      if (this.preview?.dataset_id === datasetId) this.preview = null
    },
  },
})

import { defineStore } from 'pinia'

import { errorMessage } from '../api/client'
import { listTasks } from '../api/tasks'
import type { TaskRecord } from '../types'

export const useTaskStore = defineStore('tasks', {
  state: () => ({
    items: [] as TaskRecord[],
    loading: false,
    error: '',
  }),
  actions: {
    async refresh() {
      this.loading = true
      this.error = ''
      try {
        this.items = await listTasks()
      } catch (error) {
        this.error = errorMessage(error)
      } finally {
        this.loading = false
      }
    },
  },
})

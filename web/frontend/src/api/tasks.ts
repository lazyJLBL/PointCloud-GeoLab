import type { TaskRecord, TaskRequest } from '../types'
import { apiClient } from './client'

export async function createTask(path: string, payload: TaskRequest): Promise<TaskRecord> {
  const response = await apiClient.post<TaskRecord>(`/tasks/${path}`, payload)
  return response.data
}

export async function listTasks(): Promise<TaskRecord[]> {
  const response = await apiClient.get<TaskRecord[]>('/tasks')
  return response.data
}

export function artifactUrl(taskId: string, artifactName: string): string {
  const base = apiClient.defaults.baseURL ?? '/api'
  return `${base}/artifacts/${encodeURIComponent(taskId)}/${encodeURIComponent(artifactName)}`
}

import type { DatasetPreview, DatasetRecord } from '../types'
import { apiClient } from './client'

export async function uploadDataset(file: File): Promise<DatasetRecord> {
  const body = new FormData()
  body.append('file', file)
  const response = await apiClient.post<DatasetRecord>('/datasets/upload', body)
  return response.data
}

export async function listDatasets(): Promise<DatasetRecord[]> {
  const response = await apiClient.get<DatasetRecord[]>('/datasets')
  return response.data
}

export async function getDatasetPreview(datasetId: string): Promise<DatasetPreview> {
  const response = await apiClient.get<DatasetPreview>(`/datasets/${datasetId}/preview`)
  return response.data
}

export async function deleteDataset(datasetId: string): Promise<void> {
  await apiClient.delete(`/datasets/${datasetId}`)
}

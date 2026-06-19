export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface DatasetRecord {
  id: string
  filename: string
  original_filename: string
  path: string
  size_bytes: number
  extension: string
  created_at: string
}

export interface DatasetPreview {
  dataset_id: string
  point_count: number
  sampled_count: number
  points: number[][]
}

export interface TaskRequest {
  dataset_id?: string | null
  source_dataset_id?: string | null
  target_dataset_id?: string | null
  parameters: Record<string, unknown>
}

export interface TaskRecord {
  id: string
  task_type: string
  status: TaskStatus
  request: TaskRequest
  result: Record<string, unknown> | null
  error: string | null
  artifacts: Record<string, string>
  created_at: string
  updated_at: string
}

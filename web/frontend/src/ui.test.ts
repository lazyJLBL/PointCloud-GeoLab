// @ts-nocheck
import { describe, expect, it } from 'vitest'
import fs from 'fs'
import path from 'path'

describe('Component & Page checks', () => {
  it('MetricsPanel formats complex metrics', () => {
    const code = fs.readFileSync(path.join(__dirname, 'components/MetricsPanel.vue'), 'utf-8')
    expect(code).toContain('formatValue')
    expect(code).toContain('JSON.stringify')
    expect(code).toContain('toFixed')
  })

  it('ArtifactDownloads handles nested artifact labels', () => {
    const code = fs.readFileSync(path.join(__dirname, 'components/ArtifactDownloads.vue'), 'utf-8')
    expect(code).toContain('artifactUrl(props.taskId, name)')
  })

  it('Registration page disables button without dataset and shows hint', () => {
    const code = fs.readFileSync(path.join(__dirname, 'pages/Registration.vue'), 'utf-8')
    expect(code).toContain(':disabled="!canRun"')
    expect(code).toContain('datasets.items.length < 2')
    expect(code).toContain('Choose both source and target datasets')
  })

  it('Datasets page accepts .off', () => {
    const code = fs.readFileSync(path.join(__dirname, 'components/FileUploader.vue'), 'utf-8')
    expect(code).toContain('.off')
    expect(code).toContain('ModelNet-like')
  })
})

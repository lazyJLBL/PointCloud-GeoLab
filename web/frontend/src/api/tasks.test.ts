import { describe, expect, it } from 'vitest'

import { artifactUrl } from './tasks'

describe('artifactUrl', () => {
  it('preserves nested artifact paths while encoding path parts', () => {
    expect(artifactUrl('task 1', 'artifacts/figures/a plot.png')).toBe(
      '/api/artifacts/task%201/artifacts/figures/a%20plot.png',
    )
  })
})

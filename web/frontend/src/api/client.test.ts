import { describe, expect, it } from 'vitest'
import { AxiosError } from 'axios'

import { errorMessage } from './client'

function axiosError(detail: unknown): AxiosError {
  return new AxiosError('request failed', '400', undefined, undefined, {
    data: { detail },
    status: 400,
    statusText: 'Bad Request',
    headers: {},
    config: {} as never,
  })
}

describe('errorMessage', () => {
  it('uses FastAPI string detail', () => {
    expect(errorMessage(axiosError('bad upload'))).toBe('bad upload')
  })

  it('renders FastAPI validation arrays', () => {
    expect(errorMessage(axiosError([{ msg: 'field required' }, 'bad type']))).toBe(
      'field required; bad type',
    )
  })

  it('renders object detail without losing context', () => {
    expect(errorMessage(axiosError({ reason: 'too large' }))).toContain('too large')
  })
})

/**
 * API client service for RepoMind backend
 */

import type {
  IndexRepositoryRequest,
  IndexRepositoryResponse,
  JobStatusResponse,
  QueryRepositoryRequest,
  SSEEvent,
} from '../types/api';

const API_BASE_URL = '/api';

export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string
  ) {
    super(message);
    this.name = 'APIError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new APIError(
      error.error || 'Request failed',
      response.status,
      error.detail
    );
  }
  return response.json();
}

export const api = {
  /**
   * Check API health status
   */
  async healthCheck(): Promise<{ status: string }> {
    const response = await fetch('/health');
    return handleResponse(response);
  },

  /**
   * Start indexing a repository
   */
  async indexRepository(request: IndexRepositoryRequest): Promise<IndexRepositoryResponse> {
    const response = await fetch(`${API_BASE_URL}/repos/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse(response);
  },

  /**
   * Get repository indexing status
   */
  async getRepositoryStatus(repoId: string): Promise<JobStatusResponse> {
    const response = await fetch(`${API_BASE_URL}/repos/${repoId}/status`);
    return handleResponse(response);
  },

  /**
   * Query a repository with SSE streaming
   */
  streamQuery(
    repoId: string,
    request: QueryRepositoryRequest,
    onEvent: (event: SSEEvent) => void,
    onError: (error: Error) => void
  ): () => void {
    const abortController = new AbortController();

    fetch(`${API_BASE_URL}/repos/${repoId}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal: abortController.signal,
    })
      .then((response) => {
        if (!response.ok) {
          throw new APIError('Query failed', response.status);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          throw new Error('Response body is not readable');
        }

        const activeReader = reader;
        let buffer = '';

        function read() {
          activeReader.read().then(({ done, value }) => {
            if (done) {
              return;
            }

            // Accumulate raw text — a single read() chunk may contain
            // partial lines, or multiple complete SSE blocks.
            buffer += decoder.decode(value, { stream: true });

            // SSE blocks are separated by a blank line (\n\n or \r\n\r\n).
            // Process every complete block and keep the trailing
            // partial text in `buffer` for the next read().
            const blocks = buffer.split(/\r?\n\r?\n/);
            buffer = blocks.pop() || '';

            for (const block of blocks) {
              let eventType = 'message';
              let dataStr = '';

              for (const line of block.split(/\r?\n/)) {
                if (line.startsWith('event:')) {
                  eventType = line.substring(6).trim();
                } else if (line.startsWith('data:')) {
                  dataStr += line.substring(5).trim();
                }
                // ignore comments (lines starting with ':')
              }

              if (!dataStr) continue;

              try {
                const data = JSON.parse(dataStr);
                onEvent({ type: eventType, data } as SSEEvent);
              } catch (e) {
                console.error('Failed to parse SSE data:', e);
              }
            }

            read();
          }).catch(onError);
        }

        read();
      })
      .catch(onError);

    return () => abortController.abort();
  },
};

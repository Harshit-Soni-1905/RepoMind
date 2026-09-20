/**
 * Repository status polling component
 */

import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type { JobStatusResponse } from '../types/api';

interface StatusPanelProps {
  repoId: string;
  onStatusChange: (status: JobStatusResponse) => void;
}

/** Whether the job is still in an active-processing state. */
function isInProgress(status: string): boolean {
  return status !== 'ready' && status !== 'failed';
}

export function StatusPanel({ repoId, onStatusChange }: StatusPanelProps) {
  const [status, setStatus] = useState<JobStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let intervalId: number;

    const pollStatus = async () => {
      try {
        const response = await api.getRepositoryStatus(repoId);
        setStatus(response);
        onStatusChange(response);

        // Stop polling if terminal state reached
        if (response.status === 'ready' || response.status === 'failed') {
          clearInterval(intervalId);
        }
      } catch (err) {
        setError('Failed to fetch status');
      }
    };

    // Initial fetch
    pollStatus();

    // Poll every 2 seconds
    intervalId = window.setInterval(pollStatus, 2000);

    return () => clearInterval(intervalId);
  }, [repoId, onStatusChange]);

  if (error) {
    return (
      <div className="error-card" role="alert">
        <span className="error-icon" aria-hidden="true">⚠</span>
        <div className="error-content">
          <div className="error-message">{error}</div>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="loading-status" aria-live="polite">
        <span className="thinking-spinner" aria-hidden="true" />
        Loading status…
      </div>
    );
  }

  const inProgress = isInProgress(status.status);

  return (
    <div className="status-card">
      {/* Badge + status label */}
      <div className="status-header-row">
        <span className={`status-badge status-${status.status}`}>
          <span className="status-badge-dot" aria-hidden="true" />
          {status.status.replace('_', ' ')}
        </span>
      </div>

      {/* Current message */}
      <p className="status-message">{status.message}</p>

      {/* Progress bar */}
      <div className="progress-container">
        <div className="progress-bar" role="progressbar" aria-valuenow={status.progress} aria-valuemin={0} aria-valuemax={100}>
          <div
            className={`progress-bar-fill${inProgress ? ' animating' : ''}`}
            style={{ width: `${status.progress}%` }}
          />
        </div>
        <div className="progress-label">
          <span>{status.progress}% complete</span>
          {inProgress && <span>Processing…</span>}
        </div>
      </div>

      {/* Error state */}
      {status.error && (
        <div className="error-card" role="alert" style={{ marginTop: '0.75rem' }}>
          <span className="error-icon" aria-hidden="true">⚠</span>
          <div className="error-content">
            <div className="error-message">{status.error}</div>
          </div>
        </div>
      )}

      {/* Success state with stats */}
      {status.status === 'ready' && (
        <>
          <div className="success-card" role="status">
            <span aria-hidden="true">✓</span>
            Repository indexed successfully
          </div>
          <div className="status-stats">
            <div className="status-stat">
              <span className="status-stat-value">{status.file_count || 0}</span>
              <span className="status-stat-label">Files</span>
            </div>
            <div className="status-stat">
              <span className="status-stat-value">{status.chunk_count || 0}</span>
              <span className="status-stat-label">Chunks</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

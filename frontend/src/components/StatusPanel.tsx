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
    return <div className="error">{error}</div>;
  }

  if (!status) {
    return <div className="loading">Loading status</div>;
  }

  return (
    <div className="status-card">
      <div style={{ marginBottom: '1rem' }}>
        <span className={`status-badge status-${status.status}`}>
          {status.status}
        </span>
      </div>

      <p style={{ marginBottom: '1rem' }}>{status.message}</p>

      <div className="progress-bar">
        <div
          className="progress-bar-fill"
          style={{ width: `${status.progress}%` }}
        />
      </div>

      <div style={{ fontSize: '0.875rem', color: '#666' }}>
        {status.progress}% complete
      </div>

      {status.error && (
        <div className="error" style={{ marginTop: '1rem' }}>
          <strong>Error:</strong> {status.error}
        </div>
      )}

      {status.status === 'ready' && (
        <div className="success" style={{ marginTop: '1rem' }}>
          <strong>Ready!</strong> Indexed {status.file_count || 0} files,{' '}
          {status.chunk_count || 0} code chunks
        </div>
      )}
    </div>
  );
}

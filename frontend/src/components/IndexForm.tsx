/**
 * Repository indexing form component
 */

import { useState, FormEvent } from 'react';
import { api, APIError } from '../services/api';
import type { JobStatusResponse } from '../types/api';

interface IndexFormProps {
  onIndexStarted: (status: JobStatusResponse) => void;
}

export function IndexForm({ onIndexStarted }: IndexFormProps) {
  const [repoUrl, setRepoUrl] = useState('');
  const [branch, setBranch] = useState('main');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const response = await api.indexRepository({
        repo_url: repoUrl,
        branch: branch || 'main',
      });

      // Poll for initial status
      const status = await api.getRepositoryStatus(response.repo_id);
      onIndexStarted(status);
    } catch (err) {
      if (err instanceof APIError) {
        setError(`${err.message}${err.detail ? ': ' + err.detail : ''}`);
      } else {
        setError('Failed to start indexing');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {error && (
        <div className="error-card" role="alert">
          <span className="error-icon" aria-hidden="true">⚠</span>
          <div className="error-content">
            <div className="error-message">{error}</div>
          </div>
        </div>
      )}

      <div className="form-group">
        <label htmlFor="repo-url">Repository URL</label>
        <div className="input-wrapper">
          <span className="input-icon" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path fillRule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
          </span>
          <input
            id="repo-url"
            type="url"
            className="has-icon"
            value={repoUrl}
            onChange={(e) => setRepoUrl(e.target.value)}
            placeholder="https://github.com/user/repo"
            required
            disabled={loading}
          />
        </div>
        <small>Public GitHub or GitLab repository URL</small>
      </div>

      <div className="form-group">
        <label htmlFor="branch">Branch</label>
        <div className="input-wrapper">
          <span className="input-icon" aria-hidden="true">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
              <path fillRule="evenodd" d="M11.75 2.5a.75.75 0 100 1.5.75.75 0 000-1.5zm-2.25.75a2.25 2.25 0 113 2.122V6A2.5 2.5 0 0110 8.5H6a1 1 0 00-1 1v1.128a2.251 2.251 0 11-1.5 0V5.372a2.25 2.25 0 111.5 0v1.836A2.492 2.492 0 016 7h4a1 1 0 001-1v-.628A2.25 2.25 0 019.5 3.25zM4.25 12a.75.75 0 100 1.5.75.75 0 000-1.5zM3.5 3.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0z" />
            </svg>
          </span>
          <input
            id="branch"
            type="text"
            className="has-icon"
            value={branch}
            onChange={(e) => setBranch(e.target.value)}
            placeholder="main"
            disabled={loading}
          />
        </div>
        <small>Branch to index (default: main)</small>
      </div>

      <button type="submit" className="btn-primary" disabled={loading || !repoUrl}>
        {loading && <span className="btn-spinner" aria-hidden="true" />}
        {loading ? 'Starting…' : 'Index Repository'}
      </button>
    </form>
  );
}

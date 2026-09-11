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
      {error && <div className="error">{error}</div>}

      <div className="form-group">
        <label htmlFor="repo-url">Repository URL</label>
        <input
          id="repo-url"
          type="url"
          value={repoUrl}
          onChange={(e) => setRepoUrl(e.target.value)}
          placeholder="https://github.com/user/repo"
          required
          disabled={loading}
        />
        <small>GitHub or GitLab public repository URL</small>
      </div>

      <div className="form-group">
        <label htmlFor="branch">Branch</label>
        <input
          id="branch"
          type="text"
          value={branch}
          onChange={(e) => setBranch(e.target.value)}
          placeholder="main"
          disabled={loading}
        />
        <small>Branch to index (default: main)</small>
      </div>

      <button type="submit" disabled={loading || !repoUrl}>
        {loading ? 'Starting...' : 'Index Repository'}
      </button>
    </form>
  );
}

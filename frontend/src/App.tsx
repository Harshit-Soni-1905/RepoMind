/**
 * Main App component
 */

import { useState } from 'react';
import { IndexForm } from './components/IndexForm';
import { StatusPanel } from './components/StatusPanel';
import { QueryPanel } from './components/QueryPanel';
import type { JobStatusResponse } from './types/api';
import './App.css';

export function App() {
  const [currentRepo, setCurrentRepo] = useState<JobStatusResponse | null>(null);

  return (
    <div className="app">
      <header className="app-header">
        <h1>RepoMind</h1>
        <p>AI-Powered Codebase Understanding</p>
      </header>

      <main className="app-main">
        <section className="index-section">
          <h2>Index a Repository</h2>
          <IndexForm onIndexStarted={setCurrentRepo} />
        </section>

        {currentRepo && (
          <>
            <section className="status-section">
              <h2>Indexing Status</h2>
              <StatusPanel
                repoId={currentRepo.repo_id}
                onStatusChange={setCurrentRepo}
              />
            </section>

            {currentRepo.status === 'ready' && (
              <section className="query-section">
                <h2>Ask Questions</h2>
                <QueryPanel repoId={currentRepo.repo_id} />
              </section>
            )}
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>
          RepoMind - A demonstration of AST-based parsing, hybrid retrieval, and ReAct agents
        </p>
      </footer>
    </div>
  );
}

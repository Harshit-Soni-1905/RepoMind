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
      {/* Navigation Bar */}
      <nav className="nav-bar" aria-label="Main navigation">
        <div className="nav-brand">
          <div className="nav-logo" aria-hidden="true">R</div>
          <span className="nav-title">RepoMind</span>
          <span className="nav-subtitle">Code Intelligence</span>
        </div>
        <div className="nav-status" aria-label="AI Agent status: online">
          <span className="nav-status-dot" aria-hidden="true" />
          <span>AI Agent Online</span>
        </div>
      </nav>

      {/* Hero Header */}
      <header className="app-header">
        <div className="hero-badge">
          <span className="hero-badge-dot" aria-hidden="true" />
          AI-Powered Code Intelligence
        </div>
        <h1>Understand Any Codebase.</h1>
        <p>
          Explore repositories, trace dependencies, and ask questions about
          your codebase using AI-powered hybrid retrieval.
        </p>
      </header>

      <main className="app-main">
        {/* Index Section */}
        <section className="index-section" aria-labelledby="index-heading">
          <div className="section-header">
            <div className="section-icon" aria-hidden="true">📦</div>
            <h2 id="index-heading">Index a Repository</h2>
          </div>
          <p className="section-description">
            Connect a public GitHub or GitLab repository to begin analysis
          </p>
          <IndexForm onIndexStarted={setCurrentRepo} />
        </section>

        {currentRepo && (
          <>
            {/* Status Section */}
            <section className="status-section" aria-labelledby="status-heading">
              <div className="section-header">
                <div className="section-icon" aria-hidden="true">⚡</div>
                <h2 id="status-heading">Repository Status</h2>
              </div>
              <p className="section-description">
                Tracking indexing progress for your repository
              </p>
              <StatusPanel
                repoId={currentRepo.repo_id}
                onStatusChange={setCurrentRepo}
              />
            </section>

            {/* Query Section */}
            {currentRepo.status === 'ready' && (
              <section className="query-section" aria-labelledby="query-heading">
                <div className="section-header">
                  <div className="section-icon" aria-hidden="true">🔍</div>
                  <h2 id="query-heading">Ask Your Codebase</h2>
                </div>
                <p className="section-description">
                  Ask questions about architecture, dependencies, functions, files, and behavior
                </p>
                <QueryPanel repoId={currentRepo.repo_id} />
              </section>
            )}
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>
          RepoMind — AST-based parsing · hybrid retrieval · ReAct agent
        </p>
      </footer>
    </div>
  );
}

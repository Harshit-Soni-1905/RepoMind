/**
 * Repository query component with SSE streaming
 */

import { useState, useCallback, FormEvent, KeyboardEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../services/api';
import type { SSEEvent, ToolExecutionResponse } from '../types/api';

interface QueryPanelProps {
  repoId: string;
}

const SUGGESTED_QUESTIONS = [
  'What does this project do?',
  'What are the main entry points?',
  'How are imports structured?',
  'Show me the core data models',
];

/** Map tool names to a short icon character. */
function toolIcon(name: string): string {
  const lower = name.toLowerCase();
  if (lower.includes('search') || lower.includes('retriev')) return '🔍';
  if (lower.includes('graph')) return '🔗';
  if (lower.includes('read') || lower.includes('file')) return '📄';
  if (lower.includes('list')) return '📂';
  return '⚙';
}

export function QueryPanel({ repoId }: QueryPanelProps) {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState('');
  const [toolExecutions, setToolExecutions] = useState<ToolExecutionResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [activityOpen, setActivityOpen] = useState(true);

  const submitQuestion = useCallback(
    (q: string) => {
      if (!q.trim() || loading) return;

      setError(null);
      setAnswer('');
      setToolExecutions([]);
      setLoading(true);

      const toolExecs: ToolExecutionResponse[] = [];

      api.streamQuery(
        repoId,
        { question: q },
        (event: SSEEvent) => {
          if (event.type === 'tool_result') {
            toolExecs.push({
              tool: event.data.tool,
              arguments: {},
              summary: event.data.summary,
              order: event.data.order,
            });
            setToolExecutions([...toolExecs]);
          } else if (event.type === 'answer') {
            setAnswer(event.data.text);
          } else if (event.type === 'error') {
            setError(event.data.message);
            setLoading(false);
          } else if (event.type === 'done') {
            setLoading(false);
          }
        },
        (err: Error) => {
          setError(err.message);
          setLoading(false);
        },
      );
    },
    [repoId, loading],
  );

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    submitQuestion(question);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Ctrl/Cmd + Enter submits
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      submitQuestion(question);
    }
  };

  return (
    <div className="query-form">
      {/* Suggested questions */}
      <div className="query-suggestions" role="list" aria-label="Suggested questions">
        {SUGGESTED_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            role="listitem"
            className="query-suggestion"
            onClick={() => {
              setQuestion(q);
              submitQuestion(q);
            }}
            disabled={loading}
          >
            {q}
          </button>
        ))}
      </div>

      {/* Error display */}
      {error && (
        <div className="error-card" role="alert">
          <span className="error-icon" aria-hidden="true">⚠</span>
          <div className="error-content">
            <div className="error-message">{error}</div>
          </div>
        </div>
      )}

      {/* Input form */}
      <form onSubmit={handleSubmit}>
        <div className="query-input-wrapper">
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about the codebase…"
            required
            disabled={loading}
            aria-label="Ask a question about the codebase"
          />
          <div className="query-actions">
            <button
              type="submit"
              className="btn-primary"
              disabled={loading || !question.trim()}
              aria-label="Submit question"
            >
              {loading && <span className="btn-spinner" aria-hidden="true" />}
              {loading ? 'Thinking…' : 'Ask'}
              {!loading && <kbd className="btn-kbd" aria-hidden="true">⌘↵</kbd>}
            </button>
          </div>
        </div>
      </form>

      {/* Agent activity (tool trace) */}
      {toolExecutions.length > 0 && (
        <div className="agent-activity">
          <button
            type="button"
            className="agent-activity-header"
            onClick={() => setActivityOpen((p) => !p)}
            aria-expanded={activityOpen}
            aria-controls="agent-activity-list"
          >
            <span className="agent-activity-title">
              <span className="agent-activity-title-icon" aria-hidden="true">⚡</span>
              Agent Activity
              <span className="agent-activity-count">{toolExecutions.length}</span>
            </span>
            <span
              className={`agent-activity-chevron${activityOpen ? ' expanded' : ''}`}
              aria-hidden="true"
            >
              ▼
            </span>
          </button>

          {activityOpen && (
            <div className="agent-activity-list" id="agent-activity-list" role="list">
              {toolExecutions.map((exec, idx) => (
                <div key={idx} className="tool-execution" role="listitem" style={{ animationDelay: `${idx * 50}ms` }}>
                  <span className="tool-icon complete" aria-hidden="true">
                    {toolIcon(exec.tool)}
                  </span>
                  <span className="tool-name">{exec.tool}</span>
                  <span className="tool-separator" aria-hidden="true">→</span>
                  <span className="tool-summary">{exec.summary}</span>
                </div>
              ))}
              {loading && (
                <div className="tool-execution" role="listitem">
                  <span className="tool-icon running" aria-hidden="true">●</span>
                  <span className="tool-name">working</span>
                  <span className="tool-separator" aria-hidden="true">→</span>
                  <span className="tool-summary">Processing…</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Answer */}
      {answer && (
        <div className="answer-box">
          <div className="answer-header">
            <span className="answer-avatar" aria-hidden="true">R</span>
            <span className="answer-label">RepoMind AI</span>
          </div>
          <div className="answer-body">
            <div className="markdown-content">
              <ReactMarkdown
                components={{
                  a: ({ node, ...props }) => (
                    <a {...props} target="_blank" rel="noopener noreferrer" />
                  ),
                }}
              >
                {answer}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      )}

      {/* Thinking indicator */}
      {loading && !answer && toolExecutions.length === 0 && (
        <div className="thinking-indicator" aria-live="polite">
          <span className="thinking-spinner" aria-hidden="true" />
          <span className="thinking-text">
            Agent is analyzing the codebase<span className="thinking-dots" />
          </span>
        </div>
      )}
    </div>
  );
}

/**
 * Repository query component with SSE streaming
 */

import { useState, FormEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../services/api';
import type { SSEEvent, ToolExecutionResponse } from '../types/api';

interface QueryPanelProps {
  repoId: string;
}

export function QueryPanel({ repoId }: QueryPanelProps) {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState('');
  const [toolExecutions, setToolExecutions] = useState<ToolExecutionResponse[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setAnswer('');
    setToolExecutions([]);
    setLoading(true);

    const toolExecs: ToolExecutionResponse[] = [];

    const cancel = api.streamQuery(
      repoId,
      { question },
      (event: SSEEvent) => {
        console.log('SSE event:', event);

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
      }
    );

    // Store cancel function for cleanup
    return () => cancel();
  };

  return (
    <div className="query-form">
      {error && <div className="error">{error}</div>}

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="question">Ask a question about the codebase</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="How does authentication work?"
            required
            disabled={loading}
          />
        </div>

        <button type="submit" disabled={loading || !question}>
          {loading ? 'Thinking...' : 'Ask Question'}
        </button>
      </form>

      {toolExecutions.length > 0 && (
        <div className="tool-trace">
          <h4>Agent Tool Trace</h4>
          {toolExecutions.map((exec, idx) => (
            <div key={idx} className="tool-execution">
              <strong>{exec.tool}</strong>: {exec.summary}
            </div>
          ))}
        </div>
      )}

      {answer && (
        <div className="answer-box">
          <h3>Answer</h3>
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
      )}

      {loading && !answer && (
        <div className="loading">Agent is analyzing the codebase</div>
      )}
    </div>
  );
}

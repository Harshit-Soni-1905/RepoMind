/**
 * TypeScript type definitions for RepoMind API
 */

export enum JobStatus {
  PENDING = 'pending',
  CLONING = 'cloning',
  SCANNING = 'scanning',
  PARSING = 'parsing',
  CHUNKING = 'chunking',
  EMBEDDING = 'embedding',
  GRAPH_BUILDING = 'graph_building',
  READY = 'ready',
  FAILED = 'failed',
}

export interface IndexRepositoryRequest {
  repo_url: string;
  branch?: string;
}

export interface IndexRepositoryResponse {
  repo_id: string;
  job_id: string;
  status: string;
  message: string;
}

export interface JobStatusResponse {
  repo_id: string;
  job_id: string;
  status: JobStatus;
  progress: number;
  message: string;
  error?: string;
  created_at: string;
  updated_at: string;
  file_count?: number;
  chunk_count?: number;
}

export interface QueryRepositoryRequest {
  question: string;
  provider?: string;
  model?: string;
  top_k?: number;
  depth?: number;
  max_iterations?: number;
}

export interface ToolExecutionResponse {
  tool: string;
  arguments: Record<string, any>;
  summary: string;
  order: number;
}

export interface QueryRepositoryResponse {
  answer: string;
  tool_executions: ToolExecutionResponse[];
  success: boolean;
  error?: string;
}

export interface SSEEvent {
  type: 'start' | 'tool_start' | 'tool_result' | 'answer' | 'done' | 'error';
  data?: any;
}

export interface ErrorResponse {
  error: string;
  detail?: string;
  request_id?: string;
}

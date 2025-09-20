import axios from 'axios';

const configuredBase = import.meta.env.VITE_API_BASE_URL as string | undefined;
const normalizedBase = (() => {
  const fallback = 'http://localhost:8000';
  const base = (configuredBase && configuredBase.trim().length > 0)
    ? configuredBase.trim()
    : fallback;
  const sanitized = base.endsWith('/') ? base.slice(0, -1) : base;
  return sanitized.endsWith('/api') ? sanitized : `${sanitized}/api`;
})();

const apiClient = axios.create({
  baseURL: normalizedBase

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api'


});

export interface Metric {
  name: string;
  value: number;
  details: Record<string, unknown>;
}

export interface Prediction {
  uid: string;
  inputs: Record<string, unknown>;
  expected_output: unknown;
  predicted_output: unknown;
  metadata: Record<string, unknown>;
}

export interface RunSummary {
  id: number;
  name: string;
  task: string;
  config_name: string;
  config_path: string;
  started_at: string;
  completed_at: string;
  duration: number;
  metrics: Metric[];
}

export interface RunDetail extends RunSummary {
  predictions: Prediction[];
}

export interface ConfigInfo {
  name: string;
  path: string;
  description?: string | null;
}

export async function fetchConfigs(): Promise<ConfigInfo[]> {
  const { data } = await apiClient.get<ConfigInfo[]>('/configs');
  return data;
}

export async function fetchRuns(): Promise<RunSummary[]> {
  const { data } = await apiClient.get<RunSummary[]>('/runs');
  return data;
}

export async function fetchRun(runId: number): Promise<RunDetail> {
  const { data } = await apiClient.get<RunDetail>(`/runs/${runId}`);
  return data;
}

export async function triggerRun(config: string, savePredictions?: boolean): Promise<RunDetail> {
  const payload: Record<string, unknown> = { config };
  if (typeof savePredictions === 'boolean') {
    payload.save_predictions = savePredictions;
  }
  const { data } = await apiClient.post<RunDetail>('/runs', payload);
  return data;
}

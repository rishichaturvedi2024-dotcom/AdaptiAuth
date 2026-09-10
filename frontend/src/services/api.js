/**
 * AdaptiAuth — API Client Service
 *
 * Centralized API client for communicating with the FastAPI backend.
 * All API calls go through here for consistent error handling.
 */

const API_BASE = '/api';

class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  // Add auth token if available
  const token = localStorage.getItem('adaptiauth_token');
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(url, config);

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(
      data.detail || `API error: ${response.status}`,
      response.status,
      data
    );
  }

  return response.json();
}

// ── Auth ────────────────────────────────────────────────────
export async function register(username, password) {
  return request('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
}

export async function login(username, password) {
  const data = await request('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
  localStorage.setItem('adaptiauth_token', data.access_token);
  localStorage.setItem('adaptiauth_session', data.session_id);
  return data;
}

export function logout() {
  localStorage.removeItem('adaptiauth_token');
  localStorage.removeItem('adaptiauth_session');
}

// ── Trust ───────────────────────────────────────────────────
export async function getTrustScore(sessionId) {
  return request(`/trust/${sessionId}/score`);
}

export async function triggerRescore(sessionId) {
  return request(`/trust/${sessionId}/rescore`, { method: 'POST' });
}

export async function getTrustExplanation(sessionId) {
  return request(`/trust/${sessionId}/explain`);
}

// ── Sessions ────────────────────────────────────────────────
export async function getSession(sessionId) {
  return request(`/sessions/${sessionId}`);
}

export async function listSessions(status = null) {
  const query = status ? `?status_filter=${status}` : '';
  return request(`/sessions/${query}`);
}

// ── Dashboard ───────────────────────────────────────────────
export async function getDashboardSummary() {
  return request('/dashboard/summary');
}

export async function getAlerts(limit = 50) {
  return request(`/dashboard/alerts?limit=${limit}`);
}

export async function acknowledgeAlert(alertId) {
  return request(`/dashboard/alerts/${alertId}/acknowledge`, {
    method: 'POST',
  });
}

export async function getTrustEvents(sessionId) {
  return request(`/dashboard/events/${sessionId}`);
}

export { ApiError };

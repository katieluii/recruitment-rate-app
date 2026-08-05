/* API client — swap RAILWAY_URL for your deployed Railway backend URL */
const RAILWAY_URL = 'https://web-production-e6859b.up.railway.app';

const API_BASE = (() => {
  const { hostname, origin } = window.location;
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    // Same origin the page was served from. The FastAPI app serves both the
    // frontend and the API, so hardcoding a port breaks the moment the server
    // runs anywhere other than 8000 — and silently, because the request then
    // hits whatever else happens to hold that port and 404s.
    return `${origin}/api`;
  }
  // When embedded in Bolt or any external page, call Railway directly
  return `${RAILWAY_URL}/api`;
})();

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  getPhases: () => apiFetch('/phases'),
  getTherapeuticAreas: () => apiFetch('/therapeutic-areas'),
  getEndpointArchetypes: () => apiFetch('/endpoint-archetypes').then(r => r.archetypes),
  predict: (payload) =>
    apiFetch('/predict', { method: 'POST', body: JSON.stringify(payload) }),
  getAnalytics: (phase) => apiFetch(`/analytics?phase=${phase}`),
  getInputRanges: (phase, ta) =>
    apiFetch(`/input-ranges?phase=${encodeURIComponent(phase)}` +
             (ta ? `&therapeutic_area=${encodeURIComponent(ta)}` : '')),
};

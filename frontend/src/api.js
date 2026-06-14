// All backend calls live here. In dev Vite proxies `/api` to the backend (see vite.config.js).
// In production the static site must call the backend directly — use VITE_API_BASE at build time.
const BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function handle(res) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export const health = () => fetch(`${BASE}/health`).then(handle);

export const ingestText = (text, source = "pasted-text") =>
  fetch(`${BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source }),
  }).then(handle);

export const ingestFile = (file) => {
  const form = new FormData();
  form.append("file", file);
  return fetch(`${BASE}/ingest-file`, { method: "POST", body: form }).then(handle);
};

export const chat = (message) =>
  fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  }).then(handle);

export const reset = () => fetch(`${BASE}/reset`, { method: "POST" }).then(handle);

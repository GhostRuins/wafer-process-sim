/** Base URL for the FastAPI backend (no trailing slash). */
export const API_BASE_URL = (
  (import.meta.env.VITE_API_URL as string | undefined)?.trim() ||
  `${window.location.protocol}//${window.location.hostname}:8000`
).replace(/\/$/, '')

export function apiUrl(path: string): string {
  const p = path.startsWith('/') ? path : `/${path}`
  return `${API_BASE_URL}${p}`
}

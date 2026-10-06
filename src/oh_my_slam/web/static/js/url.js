// A page's state in its URL (http_server.md "Every page has a stable URL"): the query of the hash
// route (#/page?key=value&…), changed in place (history.replaceState: no new page, no reload).

// Set (or, for null / '', remove) these keys of the current route's query.
export function setQuery(changes) {
  const raw = location.hash || '#/';
  const i = raw.indexOf('?');
  const path = i < 0 ? raw : raw.slice(0, i);
  const q = new URLSearchParams(i < 0 ? '' : raw.slice(i + 1));
  for (const [k, v] of Object.entries(changes)) {
    if (v === null || v === undefined || v === '') q.delete(k); else q.set(k, String(v));
  }
  const s = q.toString();
  const next = path + (s ? `?${s}` : '');
  if (next !== location.hash) history.replaceState(null, '', next);
}

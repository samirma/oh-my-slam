// A page's state in its URL (http_server.md "Every page has a stable URL"): the query of the hash
// route (#/page?key=value&…), changed in place (history.replaceState: no new page, no reload).

// The route a hash names: its path segments (decoded where they can be), the path they make and
// its query.
export function parseRoute(hash) {
  const raw = hash.replace(/^#\/?/, '');
  const [path, query = ''] = raw.split('?');
  const parts = path.split('/').filter(Boolean).map((p) => { try { return decodeURIComponent(p); } catch { return p; } });
  return { parts, path: parts.join('/'), query: new URLSearchParams(query) };
}

// `hash` (the current route, '#/' when empty) with these keys of its query set (or, for null /
// '', removed).
export function withQuery(hash, changes) {
  const raw = hash || '#/';
  const i = raw.indexOf('?');
  const path = i < 0 ? raw : raw.slice(0, i);
  const q = new URLSearchParams(i < 0 ? '' : raw.slice(i + 1));
  for (const [k, v] of Object.entries(changes)) {
    if (v === null || v === undefined || v === '') q.delete(k); else q.set(k, String(v));
  }
  const s = q.toString();
  return path + (s ? `?${s}` : '');
}

// Set (or, for null / '', remove) these keys of the current route's query.
export function setQuery(changes) {
  const next = withQuery(location.hash, changes);
  if (next !== location.hash) history.replaceState(null, '', next);
}

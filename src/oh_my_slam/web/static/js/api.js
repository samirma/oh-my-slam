// The public API of server.sh (http_server.md "API"), the only thing the web application talks to.
// Errors come back as ApiError with the command's message and the code of its exit status, as the
// service sends them ({error: {code, message, http_status, problems?, by_parameter?}}).

export class ApiError extends Error {
  constructor(status, body) {
    const e = (body && body.error) || {};
    super(typeof e === 'string' ? e : (e.message || `HTTP ${status}`));
    this.status = status;
    this.code = e.code || null;
    this.problems = e.problems || [];
    this.byParameter = e.by_parameter || {};
  }
}

async function bodyOf(res) {
  const text = await res.text();
  try { return text ? JSON.parse(text) : null; } catch { return { error: { message: text } }; }
}

export async function getJson(url, signal) {
  const res = await fetch(url, { signal, headers: { Accept: 'application/json' } });
  const body = await bodyOf(res);
  if (!res.ok) throw new ApiError(res.status, body);
  return body;
}

export async function postJson(url, data = {}) {
  const res = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
  });
  const body = await bodyOf(res);
  if (!res.ok) throw new ApiError(res.status, body);
  return body;
}

// Upload one file as the raw request body (its own media type, else octet-stream); `onProgress(f)`
// gets the fraction sent. Resolves to the upload ({id, name, size, path}).
export function upload(file, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/uploads?name=${encodeURIComponent(file.name)}`);
    const type = file.type && !/^(text\/plain|application\/x-www-form-urlencoded|multipart\/form-data)/.test(file.type)
      ? file.type : 'application/octet-stream';
    xhr.setRequestHeader('Content-Type', type);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { body = { error: { message: xhr.responseText } }; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new ApiError(xhr.status, body));
    };
    xhr.onerror = () => reject(new ApiError(0, { error: { message: 'the upload was interrupted; try again' } }));
    xhr.send(file);
  });
}

export async function discardUpload(id) {
  try { await fetch(`/api/uploads/${encodeURIComponent(id)}`, { method: 'DELETE' }); } catch { /* gone */ }
}

export const enc = encodeURIComponent;

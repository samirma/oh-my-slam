// The public API of server.sh (http_server.md "API"), the only thing the web application talks to.
// Errors come back as ApiError with the command's message and the code of its exit status, as the
// service sends them ({error: {code, message, http_status, problems?, by_parameter?}}).

export const enc = encodeURIComponent;

export class ApiError extends Error {
  constructor(status, body) {
    const e = (body && body.error) || {};
    super(typeof e === 'string' ? e : (e.message || (status ? `HTTP ${status}` : 'the service did not answer')));
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
  let res;
  try {
    res = await fetch(url, { signal, headers: { Accept: 'application/json' }, cache: 'no-store' });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new ApiError(0, { error: { message: 'the service did not answer; is server.sh running?' } });
  }
  const body = await bodyOf(res);
  if (!res.ok) throw new ApiError(res.status, body);
  return body;
}

export async function postJson(url, data = {}, signal) {
  const res = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data), signal,
  });
  const body = await bodyOf(res);
  if (!res.ok) throw new ApiError(res.status, body);
  return body;
}

// The stages of a Server-Timing header (`name;dur=<ms>, …`): [{name, ms}] in its order.
export function parseServerTiming(value) {
  if (!value) return [];
  return value.split(',').map((part) => {
    const [name, ...params] = part.trim().split(';');
    const dur = params.map((p) => p.trim()).find((p) => p.startsWith('dur='));
    return { name: name.trim(), ms: dur ? Number(dur.slice(4)) : null };
  }).filter((s) => s.name);
}

// Run an operation within this request (POST /api/ops/<op>): resolves when the command ends, to
// {blob, mediaType, stages} on success; throws ApiError with the command's error otherwise.
// Aborting `signal` closes the connection, which interrupts the command (http_server.md
// "Requests").
export async function runOperation(opId, values, signal) {
  let res;
  try {
    res = await fetch(`/api/ops/${enc(opId)}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values), signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new ApiError(0, { error: { message: 'the connection to the service was lost before the answer: the command was interrupted. Check that server.sh still runs, then run it again.' } });
  }
  if (!res.ok) throw new ApiError(res.status, await bodyOf(res));
  const blob = await res.blob();
  return {
    blob,
    mediaType: (res.headers.get('Content-Type') || 'application/octet-stream').split(';')[0].trim(),
    stages: parseServerTiming(res.headers.get('Server-Timing')),
  };
}

// Upload one file as the raw request body (its own media type, else octet-stream); `onProgress(f)`
// gets the fraction sent. Resolves to the upload ({id, name, size, path}); `abort()` on the
// returned promise stops it (the service deletes an interrupted upload at once).
export function upload(file, onProgress = () => {}) {
  const xhr = new XMLHttpRequest();
  const promise = new Promise((resolve, reject) => {
    xhr.open('POST', `/api/uploads?name=${enc(file.name)}`);
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
    xhr.onerror = () => reject(new ApiError(0, { error: { message: 'the upload was interrupted; choose the file again' } }));
    xhr.onabort = () => reject(new ApiError(0, { error: { message: 'the upload was cancelled' } }));
    xhr.send(file);
  });
  promise.abort = () => xhr.abort();
  return promise;
}

export async function discardUpload(id) {
  try { await fetch(`/api/uploads/${enc(id)}`, { method: 'DELETE', keepalive: true }); } catch { /* gone */ }
}

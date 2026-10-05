// Reading the 3D scene viewer's files (http_server.md "3D scene viewer"): a PLY's header first (the
// first bytes of a file from disk, or a Range request to the service), then the whole file parsed
// in a worker — or, for a job's PLY above the display budget, the service's budgeted cloud of it
// (/api/jobs/<id>/display-cloud: the shared voxel-grid selection, each point with its values,
// the header's comments kept). A disk file above the budget is refused: the browser would have to
// draw it whole.
import { plyHeader } from '/static/viewer/lib/ply.js';
import { parseCloudDocument } from '/static/viewer/lib/data.js';
import { DISPLAY_POINT_BUDGET } from '/static/viewer/lib/controls.js';

const HEADER_TRIES = [1 << 16, 1 << 20, 1 << 24];  // bytes read for a PLY header, until it ends
const fmtN = (n) => n.toLocaleString('en-US');

let worker = null;
let seq = 0;
const pending = new Map();
function inWorker(msg, transfer = []) {
  if (!worker) {
    worker = new Worker('/static/js/scene/worker.js', { type: 'module' });
    worker.onmessage = ({ data }) => {
      const p = pending.get(data.id);
      pending.delete(data.id);
      if (data.error) p.reject(new Error(data.error)); else p.resolve(data);
    };
    worker.onerror = (e) => { for (const p of pending.values()) p.reject(new Error(e.message || 'the file reader failed')); pending.clear(); };
  }
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...msg, id }, transfer);
  });
}

// A PLY's header from its first bytes: `range(n)` gives the first n bytes (fewer at the end).
export async function readHeader(range, size = Infinity) {
  let last = null;
  for (const n of HEADER_TRIES) {
    const bytes = await range(n);
    try { return plyHeader(bytes); } catch (err) {
      last = err;
      if (bytes.byteLength < n || n >= size || !/end_header/.test(err.message)) throw err;
    }
  }
  throw last;
}

// The service's URL of a job's PLY as the viewer draws it.
export function displayCloudUrl(url) {
  const m = /^\/api\/jobs\/([^/]+)\/(result|files\/(.+))$/.exec(url);
  if (!m) return null;
  const base = `/api/jobs/${m[1]}/display-cloud`;
  return m[3] ? `${base}?file=${encodeURIComponent(decodeURIComponent(m[3]))}` : base;
}

async function okBuffer(r) {
  if (!r.ok) {
    const text = await r.text();
    let msg = text.slice(0, 300);
    try { msg = JSON.parse(text).error.message; } catch { /* plain text */ }
    throw new Error(`the service answered ${r.status}: ${msg}`);
  }
  return r.arrayBuffer();
}

function tooLarge(name, count) {
  return new Error(`${name} has ${fmtN(count)} points, more than the viewer's display budget of `
    + `${fmtN(DISPLAY_POINT_BUDGET)} points (view.sh, §2.5). A file opened from disk is drawn whole by your browser, so it is refused. `
    + 'View that cloud as a map in its viewer (Maps), or open it as a job\'s file here, where the service shows a budgeted selection of its points.');
}

// A PLY from disk (a File): the cloud structure of lib/data.js, with header.comments.
export async function plyFromDisk(file) {
  const h = await readHeader(async (n) => file.slice(0, n).arrayBuffer(), file.size);
  if (h.count > DISPLAY_POINT_BUDGET) throw tooLarge(file.name, h.count);
  const buffer = await file.arrayBuffer();
  return (await inWorker({ kind: 'ply', buffer }, [buffer])).cloud;
}

// A job's PLY (its /api URL): whole when within the budget, else the service's budgeted cloud.
export async function plyFromService(url) {
  const h = await readHeader(async (n) => okBuffer(await fetch(url, { headers: { Range: `bytes=0-${n - 1}` } })));
  if (h.count > DISPLAY_POINT_BUDGET) {
    const display = displayCloudUrl(url);
    if (!display) throw tooLarge(url, h.count);
    const doc = parseCloudDocument(await okBuffer(await fetch(display)));
    return { header: { ...doc.header, comments: doc.header.comments || [] }, arrays: doc.arrays };
  }
  const buffer = await okBuffer(await fetch(url));
  return (await inWorker({ kind: 'ply', buffer }, [buffer])).cloud;
}

// A scene JSON's bytes: { doc, errors } (errors: every reason it is not a valid scene document).
export async function sceneFromBytes(buffer) {
  const r = await inWorker({ kind: 'json', buffer }, [buffer]);
  return { doc: r.doc, errors: r.errors };
}

export const fetchBuffer = async (url) => okBuffer(await fetch(url));

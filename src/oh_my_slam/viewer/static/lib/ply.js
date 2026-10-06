// PLY files read in the browser, as the oh-my-slam writers emit them (core/ply.py): `format
// binary_little_endian 1.0` or `ascii 1.0`, the first element `vertex`, with the properties x y z,
// nx ny nz, red green blue, label; others are read and ignored. A page that holds a PLY's bytes (the
// server.sh web application's point-cloud results) draws it with this viewer's rendering through
// cloudview.js. The result is the cloud structure of data.js, plus the header's format and comments.
//
// Display budget (spec §2.5, controls.js DISPLAY_POINT_BUDGET): above `budget` vertices only
// `budget` of them are read, evenly spaced in the file's order, each with exactly its values, and
// header.step says so (controls.js budgetNote). The bytes themselves are never changed.
// Throws an Error saying why bytes are not a PLY this viewer can draw.

const TYPES = {
  char: ['getInt8', 1], int8: ['getInt8', 1], uchar: ['getUint8', 1], uint8: ['getUint8', 1],
  short: ['getInt16', 2], int16: ['getInt16', 2], ushort: ['getUint16', 2], uint16: ['getUint16', 2],
  int: ['getInt32', 4], int32: ['getInt32', 4], uint: ['getUint32', 4], uint32: ['getUint32', 4],
  float: ['getFloat32', 4], float32: ['getFloat32', 4], double: ['getFloat64', 8], float64: ['getFloat64', 8],
};
// one reader per DataView getter, little-endian (monomorphic calls in the vertex loop)
const READERS = {
  getInt8: (v, o) => v.getInt8(o), getUint8: (v, o) => v.getUint8(o),
  getInt16: (v, o) => v.getInt16(o, true), getUint16: (v, o) => v.getUint16(o, true),
  getInt32: (v, o) => v.getInt32(o, true), getUint32: (v, o) => v.getUint32(o, true),
  getFloat32: (v, o) => v.getFloat32(o, true), getFloat64: (v, o) => v.getFloat64(o, true),
};
const GROUPS = { position: ['x', 'y', 'z'], normal: ['nx', 'ny', 'nz'], color: ['red', 'green', 'blue'] };
const END = 'end_header';

function parseHeader(bytes) {
  // the header is ASCII and ends with a line "end_header"
  const limit = Math.min(bytes.length, 1 << 24);
  let text = '';
  let end = -1;
  for (let at = 0; at < limit && end < 0; at += 1 << 16) {
    text += new TextDecoder('latin1').decode(bytes.subarray(at, Math.min(limit, at + (1 << 16))));
    const m = /(^|\n)end_header\r?\n/.exec(text);
    if (m) end = m.index + m[0].length;
  }
  if (!text.startsWith('ply\n') && !text.startsWith('ply\r\n')) throw new Error('not a PLY file (it does not start with "ply")');
  if (end < 0) throw new Error(`not a PLY file (no "${END}" line)`);
  const lines = text.slice(0, end).split(/\r?\n/).filter((l) => l.length);
  let format = null, count = 0, element = null;
  const comments = [], props = [];
  for (const line of lines.slice(1)) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'format') format = parts.slice(1).join(' ');
    else if (parts[0] === 'comment') comments.push(line.slice(line.indexOf('comment') + 8));
    else if (parts[0] === 'element') {
      if (element === null && parts[1] !== 'vertex') throw new Error('the first PLY element must be "vertex"');
      element = parts[1];
      if (element === 'vertex') count = Number(parts[2]);
    } else if (parts[0] === 'property' && element === 'vertex') {
      if (parts[1] === 'list') throw new Error('list properties on vertices are not supported');
      if (!TYPES[parts[1]]) throw new Error(`unknown PLY property type "${parts[1]}"`);
      props.push({ name: parts[2], type: parts[1] });
    }
  }
  if (format !== 'binary_little_endian 1.0' && format !== 'ascii 1.0') {
    throw new Error(`unsupported PLY format "${format}" (binary_little_endian 1.0 or ascii 1.0)`);
  }
  if (element === null) throw new Error('the PLY file has no vertex element');
  if (!Number.isInteger(count) || count < 0) throw new Error('bad vertex count in the PLY header');
  const names = new Set(props.map((p) => p.name));
  if (!GROUPS.position.every((n) => names.has(n))) throw new Error('the PLY vertices have no x y z');
  return { format, count, comments, props, body: end };
}

// The vertices of `n` drawn within `budget`: { count, step }; vertex floor(i * step) is the i-th one
// drawn (i = 0 … count - 1). All of them (step 1) when they fit, else `budget` of them evenly spaced.
export function budgetSelection(n, budget = Infinity) {
  const count = Math.min(n, Math.max(1, Math.floor(budget)));
  return { count, step: count < n ? n / count : 1 };
}

// The cloud of a PLY file (ArrayBuffer), within `budget` points: { header: { count, total, voxel: 0,
// step, attrs, format, comments }, arrays: { position, normal?, color?, label? } }. `total` is the
// file's vertex count, `count` the points kept (vertex floor(i * step) for the i-th). `attrs` is what
// the writer recorded (the oh-my-slam comment "attributes key=value,…"), so color=segment colours
// stay exact.
export function parsePly(buffer, budget = Infinity) {
  const bytes = new Uint8Array(buffer);
  const h = parseHeader(bytes);
  const total = h.count;
  const { count: n, step } = budgetSelection(total, budget);
  const names = new Set(h.props.map((p) => p.name));
  const arrays = { position: new Float32Array(n * 3) };
  if (GROUPS.normal.every((x) => names.has(x))) arrays.normal = new Float32Array(n * 3);
  if (GROUPS.color.every((x) => names.has(x))) arrays.color = new Uint8Array(n * 3);
  if (names.has('label')) arrays.label = new Int32Array(n);
  // where each property goes: [array, component index, components] or null
  const sinks = h.props.map((p) => {
    for (const [key, cols] of Object.entries(GROUPS)) {
      const c = cols.indexOf(p.name);
      if (c >= 0 && arrays[key]) return [arrays[key], c, 3];
    }
    return p.name === 'label' ? [arrays.label, 0, 1] : null;
  });
  if (h.format.startsWith('binary')) {
    const view = new DataView(buffer, h.body);
    const layout = h.props.map((p) => TYPES[p.type]);
    const size = layout.reduce((s, [, b]) => s + b, 0);
    if (view.byteLength < size * total) throw new Error(`the PLY body is too short for ${total} vertices`);
    // the properties that are kept: reader, offset in the vertex, array, component, components
    const rd = [], ro = [], ra = [], rc = [], rn = [];
    let off = 0;
    layout.forEach(([get, b], j) => {
      const s = sinks[j];
      if (s) { rd.push(READERS[get]); ro.push(off); ra.push(s[0]); rc.push(s[1]); rn.push(s[2]); }
      off += b;
    });
    const R = rd.length;
    for (let i = 0; i < n; i++) {
      const at = Math.floor(i * step) * size;
      for (let r = 0; r < R; r++) ra[r][i * rn[r] + rc[r]] = rd[r](view, at + ro[r]);
    }
  } else {
    const text = new TextDecoder('latin1').decode(bytes.subarray(h.body));
    let v = 0;  // vertex lines read
    let i = 0;  // points kept
    for (const line of text.split(/\r?\n/)) {
      if (v >= total) break;
      const t = line.trim();
      if (!t) continue;
      if (i < n && v === Math.floor(i * step)) {
        const vals = t.split(/\s+/);
        if (vals.length < h.props.length) throw new Error(`PLY vertex ${v} has ${vals.length} values, not ${h.props.length}`);
        for (let j = 0; j < h.props.length; j++) {
          const s = sinks[j];
          if (!s) continue;
          const x = Number(vals[j]);
          if (Number.isNaN(x) && !/^[-+]?nan$/i.test(vals[j])) {
            throw new Error(`PLY vertex ${v}: "${vals[j]}" is not a number (property ${h.props[j].name})`);
          }
          s[0][i * s[2] + s[1]] = x;
        }
        i++;
      }
      v++;
    }
    if (v < total) throw new Error(`the PLY body has ${v} vertices, not ${total}`);
  }
  const recorded = h.comments.find((c) => c.startsWith('attributes '));
  const attrs = recorded ? recorded.slice('attributes '.length) : '';
  return { header: { count: n, total, voxel: 0, step, attrs, format: h.format, comments: h.comments }, arrays };
}

// The header of a PLY file (ArrayBuffer) without reading its body: { format, count (vertices),
// comments, props }. Throws as parsePly does for a file that is not a PLY this viewer can draw.
export function plyHeader(buffer) {
  const { format, count, comments, props } = parseHeader(new Uint8Array(buffer));
  return { format, count, comments, props };
}

// parsePly off the page's thread (plyworker.js), so that reading a large cloud never blocks the
// page. The buffer is handed to the worker (detached here). Where no module worker can be started,
// it is read on this thread.
export function loadPly(buffer, budget = Infinity) {
  let worker;
  try {
    worker = new Worker(new URL('./plyworker.js', import.meta.url), { type: 'module' });
  } catch {
    return Promise.resolve().then(() => parsePly(buffer, budget));
  }
  return new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) => {
      worker.terminate();
      if (data.error) reject(new Error(data.error)); else resolve(data.cloud);
    };
    worker.onerror = (e) => {
      e.preventDefault();
      worker.terminate();
      reject(new Error(e.message || 'the PLY reader could not be started'));
    };
    worker.postMessage({ buffer, budget }, [buffer]);
  });
}

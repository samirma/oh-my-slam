// PLY files read in the browser (the 3D scene viewer of http_server.md: files opened from disk are
// never uploaded): `format binary_little_endian 1.0` or `ascii 1.0`, the first element `vertex`,
// with the properties the oh-my-slam writers emit (core/ply.py): x y z, nx ny nz, red green blue,
// label; others are read and ignored. The result is the cloud structure of data.js, plus the
// header's comments (where `mapper.sh locate` writes its cameras, cameras.js plyCameras).
// Throws an Error saying why a file is not a PLY this viewer can draw.

const TYPES = {
  char: ['getInt8', 1], int8: ['getInt8', 1], uchar: ['getUint8', 1], uint8: ['getUint8', 1],
  short: ['getInt16', 2], int16: ['getInt16', 2], ushort: ['getUint16', 2], uint16: ['getUint16', 2],
  int: ['getInt32', 4], int32: ['getInt32', 4], uint: ['getUint32', 4], uint32: ['getUint32', 4],
  float: ['getFloat32', 4], float32: ['getFloat32', 4], double: ['getFloat64', 8], float64: ['getFloat64', 8],
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

// The cloud of a PLY file (ArrayBuffer): { header: { count, total, voxel: 0, attrs, format,
// comments }, arrays: { position, normal?, color?, label? } }. `attrs` is what the writer recorded
// (the oh-my-slam comment "attributes key=value,…"), so color=segment colours stay exact.
export function parsePly(buffer) {
  const bytes = new Uint8Array(buffer);
  const h = parseHeader(bytes);
  const n = h.count;
  const names = new Set(h.props.map((p) => p.name));
  const arrays = { position: new Float32Array(n * 3) };
  if (GROUPS.normal.every((x) => names.has(x))) arrays.normal = new Float32Array(n * 3);
  if (GROUPS.color.every((x) => names.has(x))) arrays.color = new Uint8Array(n * 3);
  if (names.has('label')) arrays.label = new Int32Array(n);
  // where each property goes: [array, component index] or null
  const sinks = h.props.map((p) => {
    for (const [key, cols] of Object.entries(GROUPS)) {
      const k = cols.indexOf(p.name);
      if (k >= 0 && arrays[key]) return [arrays[key], k, 3];
    }
    return p.name === 'label' ? [arrays.label, 0, 1] : null;
  });
  if (h.format.startsWith('binary')) {
    const view = new DataView(buffer, h.body);
    const layout = h.props.map((p) => TYPES[p.type]);
    const stride = layout.reduce((s, [, size]) => s + size, 0);
    if (view.byteLength < stride * n) throw new Error(`the PLY body is too short for ${n} vertices`);
    let at = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < layout.length; j++) {
        const [get, size] = layout[j];
        const s = sinks[j];
        if (s) s[0][i * s[2] + s[1]] = view[get](at, true);
        at += size;
      }
    }
  } else {
    const text = new TextDecoder('latin1').decode(bytes.subarray(h.body));
    const lines = text.split(/\r?\n/);
    let i = 0;
    for (const line of lines) {
      if (i >= n) break;
      const t = line.trim();
      if (!t) continue;
      const v = t.split(/\s+/);
      if (v.length < h.props.length) throw new Error(`PLY vertex ${i} has ${v.length} values, not ${h.props.length}`);
      for (let j = 0; j < h.props.length; j++) {
        const s = sinks[j];
        if (s) s[0][i * s[2] + s[1]] = Number(v[j]);
      }
      i++;
    }
    if (i < n) throw new Error(`the PLY body has ${i} vertices, not ${n}`);
  }
  const recorded = h.comments.find((c) => c.startsWith('attributes '));
  const attrs = recorded ? recorded.slice('attributes '.length) : '';
  return { header: { count: n, total: n, voxel: 0, attrs, format: h.format, comments: h.comments }, arrays };
}

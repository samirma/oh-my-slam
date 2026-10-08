// Test doubles and fixtures of the browser modules' unit tests.

// segmentation/colors.py PALETTE_HEX: the object colours of ids 1-19 (spec §2.4 colour contract),
// and the mid-grey of unsegmented points.
export const PALETTE_HEX = [
  '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#9c4dff', '#42d4f4', '#f032e6',
  '#a8f04a', '#ff9ec7', '#1fb5a3', '#dcbeff', '#9a6324', '#8ef0c0', '#808000', '#ffc49b',
  '#5a9cff', '#b4339c', '#c9a227',
];
export const UNSEGMENTED_HEX = '#808080';

export function rgbOf(hex) { return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)); }

// An object of a scene as schema/openlabel.py object_entry writes it, in the colour of its id
// (cuboid: its 10 values, or null for none).
export function objectEntry(id, label, cuboid, { score = 0.9 } = {}) {
  const hex = PALETTE_HEX[(id - 1) % PALETTE_HEX.length];
  const data = {
    num: [{ name: 'score', val: score }], text: [{ name: 'color_hex', val: hex }], vec: [{ name: 'color', val: rgbOf(hex) }],
  };
  if (cuboid) {
    const [w, d, h] = cuboid.slice(7);
    data.cuboid = [{ name: 'obb', val: cuboid, coordinate_system: 'map', attributes: { num: [
      { name: 'width_m', val: w }, { name: 'depth_m', val: d }, { name: 'height_m', val: h }, { name: 'volume_m3', val: w * d * h }] } }];
  }
  return { name: `${label} ${id}`, type: label, ontology_uid: '0', coordinate_system: 'map', object_data: data };
}

// Replace globals for one test; returns the function that puts the previous ones back.
export function stubGlobals(values) {
  const saved = Object.keys(values).map((k) => [k, Object.getOwnPropertyDescriptor(globalThis, k)]);
  for (const [k, v] of Object.entries(values)) {
    Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });
  }
  return () => {
    for (const [k, d] of saved) {
      if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k];
    }
  };
}

// fetch answered by `handler(url, init)` (a Response, or a thrown error); `calls` records every
// request.
export function fakeFetch(handler) {
  const calls = [];
  const restore = stubGlobals({
    fetch: async (url, init = {}) => {
      calls.push({ url: String(url), init });
      return handler(String(url), init);
    },
  });
  return { calls, restore };
}

export function json(body, status = 200, headers = {}) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', ...headers } });
}

// The bytes of a PLY as core/ply.py writes it: x y z (float), nx ny nz (float), red green blue
// (uchar), label (int), in that order, for the arrays given (`types` gives one of them another PLY
// type: {name: type}); `extra` adds properties after them ([name, type, values]), which a reader
// ignores.
const PLY_TYPES = {
  float: ['setFloat32', 4], double: ['setFloat64', 8], uchar: ['setUint8', 1], char: ['setInt8', 1],
  short: ['setInt16', 2], ushort: ['setUint16', 2], int: ['setInt32', 4], uint: ['setUint32', 4],
};
export function plyBytes({ position, normal, color, label, extra = [] }, { encoding = 'binary', comments = [], types = {} } = {}) {
  const n = position.length / 3;
  const cols = [];  // [name, type, value(i)]
  ['x', 'y', 'z'].forEach((c, k) => cols.push([c, 'float', (i) => position[3 * i + k]]));
  if (normal) ['nx', 'ny', 'nz'].forEach((c, k) => cols.push([c, 'float', (i) => normal[3 * i + k]]));
  if (color) ['red', 'green', 'blue'].forEach((c, k) => cols.push([c, 'uchar', (i) => color[3 * i + k]]));
  if (label) cols.push(['label', 'int', (i) => label[i]]);
  for (const [name, type, values] of extra) cols.push([name, type, (i) => values[i]]);
  for (const c of cols) c[1] = types[c[0]] || c[1];
  const head = ['ply', `format ${encoding === 'binary' ? 'binary_little_endian' : 'ascii'} 1.0`,
    ...comments.map((c) => `comment ${c}`), `element vertex ${n}`,
    ...cols.map(([name, type]) => `property ${type} ${name}`), 'end_header', ''].join('\n');
  const headBytes = new TextEncoder().encode(head);
  if (encoding !== 'binary') {
    const rows = [];
    for (let i = 0; i < n; i++) {
      rows.push(cols.map(([, type, v]) => (type === 'float' || type === 'double'
        ? String(Number(v(i).toPrecision(9))) : String(v(i)))).join(' '));
    }
    const body = new TextEncoder().encode(rows.map((r) => `${r}\n`).join(''));
    return concat(headBytes, body);
  }
  const size = cols.reduce((s, [, type]) => s + PLY_TYPES[type][1], 0);
  const body = new DataView(new ArrayBuffer(size * n));
  for (let i = 0; i < n; i++) {
    let off = i * size;
    for (const [, type, v] of cols) {
      const [set, b] = PLY_TYPES[type];
      body[set](off, v(i), true);
      off += b;
    }
  }
  return concat(headBytes, new Uint8Array(body.buffer));
}

export function concat(...parts) {
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out.buffer;
}

// The binary cloud document of /api/cloud as viewer/routes.py cloud_document writes it: uint32 LE
// length J of the JSON header (space-padded so that 4 + J is a multiple of 4), the header, then
// the buffers, each on a 4-byte boundary at 4 + J + offset.
const DOC_TYPES = { position: ['float32', 3], color: ['uint8', 3], label: ['int32', 1], normal: ['float32', 3] };
export function cloudDocument(head, arrays) {
  const buffers = [];
  let offset = 0;
  for (const [name, a] of Object.entries(arrays)) {
    const [type, size] = DOC_TYPES[name];
    buffers.push({ name, type, size, offset, bytes: a.byteLength });
    offset += Math.ceil(a.byteLength / 4) * 4;
  }
  let text = JSON.stringify({ ...head, buffers });
  while ((4 + text.length) % 4) text += ' ';
  const out = new Uint8Array(4 + text.length + offset);
  new DataView(out.buffer).setUint32(0, text.length, true);
  out.set(new TextEncoder().encode(text), 4);
  for (const b of buffers) out.set(new Uint8Array(arrays[b.name].buffer, arrays[b.name].byteOffset, b.bytes), 4 + text.length + b.offset);
  return out.buffer;
}

// The operations of server.sh's OpenAPI document (web/openapi.py: each mode's describe() entry
// under x-oms, its parameters' under their schemas' x-oms), reduced to what the app reads, with
// the validation path and a fixed endpoint beside them.
const IMAGE_SUFFIXES = ['.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'];
const VIDEO_SUFFIXES = ['.avi', '.m4v', '.mkv', '.mov', '.mp4', '.webm'];
function param(name, flag, kind, extra = {}) {
  return { name, flag, kind, help: `the ${name}`, required: false, default: null, choices: null, multiple: false,
    ordered: false, accepts: null, applies: [], applies_text: '', ...extra };
}
const PLY_ONLY = { applies: [{ option: 'format', in: ['ply'] }], applies_text: 'only with -f ply' };
const OPERATIONS = [
  ['reconstruct', 'reconstruct.sh', 'Single-image reconstruction (stdout or -o file).', 'required',
    'reconstructs the image with the inference server',
    [['stdout', 'json', 'json'], ['stdout', 'png', 'depth'], ['stdout', 'ply', 'ply']],
    [param('image', '-i', 'image', { required: true, accepts: IMAGE_SUFFIXES }),
      param('format', '-f', 'enum', { default: 'json', choices: ['json', 'depth', 'ply'] }),
      param('attrs', '-p', 'attrs', { default: 'color=rgb,stride=1', ...PLY_ONLY })]],
  ['mapper-update', 'mapper.sh update', 'add images or a video to a map (created if missing)', 'required',
    'infers depth and objects of every new keyframe',
    [['stdout', 'json', 'json'], ['stdout', 'ply', 'ply'], ['-m', 'map', null]],
    [param('inputs', '-i', 'images_or_video', { required: true, multiple: true, ordered: true, accepts: [...IMAGE_SUFFIXES, ...VIDEO_SUFFIXES] }),
      param('map', '-m', 'map', { required: true }),
      param('format', '-f', 'enum', { default: 'json', choices: ['json', 'ply'] }),
      param('fps', '-fps', 'number', { default: 2, applies: [{ option: 'inputs', is: 'video', suffixes: VIDEO_SUFFIXES }], applies_text: 'only for a video' })]],
  ['mapper-locate', 'mapper.sh locate', 'camera pose of images in an existing map (read-only)', 'conditional',
    'only for retrieval in maps of more keyframes than are matched exhaustively',
    [['stdout', 'json', 'json'], ['stdout', 'ply', 'ply']],
    [param('inputs', '-i', 'images', { required: true, multiple: true }), param('map', '-m', 'map', { required: true })]],
  ['segment', 'segment.sh', 'Instance segmentation of one image.', 'required', 'segments the image with the inference server',
    [['stdout', 'json', 'json'], ['stdout', 'png', 'png']],
    [param('image', '-i', 'image', { required: true }), param('min_score', '--min-score', 'number', { default: 0.5 })]],
];
export function openapiDocument() {
  const paths = { '/api/health': { get: { summary: 'health' } }, '/api/maps/{name}': { get: { summary: 'a map' } } };
  for (const [id, label, description, inference, text, outputs, params] of OPERATIONS) {
    const body = { content: { 'application/json': { schema: { type: 'object',
      properties: Object.fromEntries(params.map((p) => [p.name, { description: p.help, 'x-oms': p }])) } } } };
    paths[`/api/ops/${id}`] = { post: { operationId: id, requestBody: body, 'x-oms': {
      id: label, description, inference, inference_text: text,
      outputs: outputs.map(([via, format, value]) => ({ via, format, when: value ? [{ option: 'format', in: [value] }] : [] })) } } };
    paths[`/api/ops/${id}/validate`] = { post: { operationId: `${id}-validate`, requestBody: body } };
  }
  paths['/api/ops/legacy'] = { post: { operationId: 'legacy' } };  // no registry entry: not an operation of the app
  return { openapi: '3.1.0', paths };
}

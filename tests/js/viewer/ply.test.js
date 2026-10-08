// lib/ply.js: a PLY as the oh-my-slam writers emit it (core/ply.py), read in the browser within the
// display budget (spec §2.5), for the server.sh web application's point-cloud results (spec §2.6
// "Image"); lib/plyworker.js reads it off the page's thread.
import { afterEach, describe, expect, test } from 'bun:test';
import { budgetSelection, loadPly, parsePly, plyHeader } from '../../../src/oh_my_slam/viewer/static/lib/ply.js';
import { concat, plyBytes, stubGlobals } from '../helpers.js';

const N = 10;
const cloud = {
  position: Float32Array.from({ length: 3 * N }, (_, i) => Math.fround(Math.sin(i) * 3.7)),
  normal: Float32Array.from({ length: 3 * N }, (_, i) => Math.fround(Math.cos(i))),
  color: Uint8Array.from({ length: 3 * N }, (_, i) => (i * 37) % 256),
  label: Int32Array.from({ length: N }, (_, i) => i % 4),
};
const COMMENTS = ['oh-my-slam map frame (z up), metres', 'attributes color=segment,normals=on'];
const text = (s) => new TextEncoder().encode(s).buffer;

function pick(a, k, idx) { return idx.flatMap((i) => [...a.slice(k * i, k * i + k)]); }

describe.each(['binary', 'ascii'])('%s PLY', (encoding) => {
  const bytes = () => plyBytes(cloud, { encoding, comments: COMMENTS });

  test('every vertex is read exactly, with the header the writer recorded', () => {
    const { header, arrays } = parsePly(bytes());
    expect(header).toEqual({
      count: N, total: N, voxel: 0, step: 1, attrs: 'color=segment,normals=on',
      format: `${encoding === 'binary' ? 'binary_little_endian' : 'ascii'} 1.0`, comments: COMMENTS,
    });
    expect(arrays.position).toEqual(cloud.position);
    expect(arrays.normal).toEqual(cloud.normal);
    expect(arrays.color).toEqual(cloud.color);
    expect(arrays.label).toEqual(cloud.label);
  });

  test('above the budget, that many vertices evenly spaced in the file order, each exactly', () => {
    const { header, arrays } = parsePly(bytes(), 4);
    const idx = [0, 2, 5, 7];  // vertex floor(i * 10 / 4)
    expect([header.count, header.total, header.step]).toEqual([4, N, 2.5]);
    expect([...arrays.position]).toEqual(pick(cloud.position, 3, idx));
    expect([...arrays.normal]).toEqual(pick(cloud.normal, 3, idx));
    expect([...arrays.color]).toEqual(pick(cloud.color, 3, idx));
    expect([...arrays.label]).toEqual(pick(cloud.label, 1, idx));
  });

  test('properties it does not draw are read past, and only the groups present get arrays', () => {
    const extra = [['intensity', 'double', Array.from({ length: N }, (_, i) => i / 3)],
      ['ring', 'short', Array.from({ length: N }, (_, i) => -i)], ['flag', 'char', Array(N).fill(-1)],
      ['id', 'uint', Array(N).fill(7)], ['w', 'ushort', Array(N).fill(9)]];
    const { header, arrays } = parsePly(plyBytes({ position: cloud.position, color: cloud.color, extra }, { encoding }));
    expect(Object.keys(arrays).sort()).toEqual(['color', 'position']);
    expect(arrays.position).toEqual(cloud.position);
    expect(arrays.color).toEqual(cloud.color);
    expect(header.attrs).toBe('');  // nothing recorded
  });
});

test('kept properties of every PLY number type are read', () => {
  const position = new Float32Array([1.5, -3, -7, 2.25, 300, 100]);
  const color = new Uint8Array([10, 20, 30, 255, 0, 128]);
  const label = new Int32Array([3, 4000000000 - 2 ** 32]);  // a uint label above 2^31 wraps as int32 does
  const types = { x: 'double', y: 'short', z: 'char', red: 'ushort', label: 'uint' };
  for (const encoding of ['binary', 'ascii']) {
    const buffer = plyBytes({ position, color, label: [3, 4000000000] }, { encoding, types });
    const { arrays } = parsePly(buffer);
    expect([...arrays.position]).toEqual([...position]);
    expect(arrays.color).toEqual(color);
    expect(arrays.label).toEqual(label);
  }
});

test('a later element and CRLF line ends are accepted; a NaN coordinate is read as NaN', () => {
  const ply = 'ply\r\nformat ascii 1.0\r\nelement vertex 2\r\nproperty float x\r\nproperty float y\r\n'
    + 'property float z\r\nelement face 0\r\nproperty list uchar int vertex_indices\r\nend_header\r\n'
    + '1 2 3\r\nnan -NaN 4.5\r\n';
  const { header, arrays } = parsePly(text(ply));
  expect(header.count).toBe(2);
  expect([...arrays.position.slice(0, 3)]).toEqual([1, 2, 3]);
  expect(Number.isNaN(arrays.position[3]) && Number.isNaN(arrays.position[4])).toBe(true);
  expect(arrays.position[5]).toBe(4.5);
});

test('a header longer than one read block is found', () => {
  const comments = Array.from({ length: 1500 }, (_, i) => `note ${i} ${'x'.repeat(60)}`);
  const { header, arrays } = parsePly(plyBytes(cloud, { comments }));
  expect(header.comments).toHaveLength(1500);
  expect(arrays.label).toEqual(cloud.label);
});

describe('bytes that are not a PLY this viewer can draw are refused with the reason', () => {
  const head = (lines) => text(`${['ply', ...lines, 'end_header'].join('\n')}\n`);
  const xyz = ['property float x', 'property float y', 'property float z'];
  test.each([
    ['not a PLY file (it does not start with "ply")', text('PK\u0003\u0004 a zip file')],
    ['not a PLY file (no "end_header" line)', text('ply\nformat ascii 1.0\nelement vertex 1\n')],
    ['the first PLY element must be "vertex"', head(['format ascii 1.0', 'element face 1', 'element vertex 1', ...xyz])],
    ['list properties on vertices are not supported', head(['format ascii 1.0', 'element vertex 1', 'property list uchar int i'])],
    ['unknown PLY property type "half"', head(['format ascii 1.0', 'element vertex 1', 'property half x'])],
    ['unsupported PLY format "binary_big_endian 1.0" (binary_little_endian 1.0 or ascii 1.0)',
      head(['format binary_big_endian 1.0', 'element vertex 1', ...xyz])],
    ['the PLY file has no vertex element', head(['format ascii 1.0'])],
    ['bad vertex count in the PLY header', head(['format ascii 1.0', 'element vertex -3', ...xyz])],
    ['bad vertex count in the PLY header', head(['format ascii 1.0', 'element vertex many', ...xyz])],
    ['the PLY vertices have no x y z', head(['format ascii 1.0', 'element vertex 1', 'property float x', 'property float y'])],
  ])('%s', (message, bytes) => {
    expect(() => parsePly(bytes)).toThrow(message);
    expect(() => plyHeader(bytes)).toThrow(message);
  });

  test('a body cut short', () => {
    const binary = new Uint8Array(plyBytes(cloud));
    expect(() => parsePly(binary.slice(0, binary.length - 5).buffer)).toThrow(`the PLY body is too short for ${N} vertices`);
    const ascii = new TextDecoder().decode(plyBytes(cloud, { encoding: 'ascii' }));
    const lines = ascii.split('\n');
    expect(() => parsePly(text(lines.slice(0, -3).join('\n')))).toThrow(`the PLY body has ${N - 2} vertices, not ${N}`);
    const lastCut = `${lines.slice(0, -2).join('\n')}\n${lines.at(-2).split(' ').slice(0, 4).join(' ')}\n`;
    expect(() => parsePly(text(lastCut))).toThrow(`PLY vertex ${N - 1} has 4 values, not 10`);
  });

  test('a value that is not a number', () => {
    const ply = 'ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 two 3\n';
    expect(() => parsePly(text(ply))).toThrow('PLY vertex 0: "two" is not a number (property y)');
  });
});

test('plyHeader reads the header only', () => {
  const head = new Uint8Array(plyBytes(cloud, { comments: COMMENTS }));
  const end = new TextDecoder().decode(head).indexOf('end_header\n') + 'end_header\n'.length;
  const h = plyHeader(concat(head.slice(0, end), new Uint8Array([1, 2, 3])));  // a body that is no PLY body
  expect(h).toEqual({
    format: 'binary_little_endian 1.0', count: N, comments: COMMENTS,
    props: [['x', 'float'], ['y', 'float'], ['z', 'float'], ['nx', 'float'], ['ny', 'float'], ['nz', 'float'],
      ['red', 'uchar'], ['green', 'uchar'], ['blue', 'uchar'], ['label', 'int']].map(([name, type]) => ({ name, type })),
  });
});

test('budgetSelection: every vertex when they fit, else `budget` of them evenly spaced', () => {
  expect(budgetSelection(10)).toEqual({ count: 10, step: 1 });
  expect(budgetSelection(10, 10)).toEqual({ count: 10, step: 1 });
  expect(budgetSelection(10, 4)).toEqual({ count: 4, step: 2.5 });
  expect(budgetSelection(10, 4.9)).toEqual({ count: 4, step: 2.5 });
  expect(budgetSelection(10, 0)).toEqual({ count: 1, step: 10 });  // at least one point
  expect(budgetSelection(0, 5)).toEqual({ count: 0, step: 1 });
  const { count, step } = budgetSelection(2503, 1000);
  const idx = Array.from({ length: count }, (_, i) => Math.floor(i * step));
  expect(new Set(idx).size).toBe(1000);
  expect(idx.at(-1)).toBeLessThan(2503);
});

describe('loadPly reads off the page thread', () => {
  let restore = () => {};
  afterEach(() => restore());

  test('in a module worker (plyworker.js), the buffer handed over', async () => {
    const buffer = plyBytes(cloud, { comments: COMMENTS });
    const expected = parsePly(buffer.slice(0), 4);
    const got = await loadPly(buffer, 4);
    expect(buffer.byteLength).toBe(0);  // transferred to the worker
    expect(got.header).toEqual(expected.header);
    for (const k of ['position', 'normal', 'color', 'label']) expect(got.arrays[k]).toEqual(expected.arrays[k]);
  });

  test('a worker that refuses the bytes rejects with its reason', async () => {
    await expect(loadPly(text('not a ply'))).rejects.toThrow('not a PLY file (it does not start with "ply")');
  });

  test('where no module worker can be started, on this thread', async () => {
    restore = stubGlobals({ Worker: class { constructor() { throw new Error('no module workers'); } } });
    const buffer = plyBytes(cloud);
    const got = await loadPly(buffer, 3);
    expect(got.header.count).toBe(3);
    expect(buffer.byteLength).toBeGreaterThan(0);  // not handed over
  });

  test('a worker that fails to start rejects, with its message or a generic one', async () => {
    const workers = [];
    restore = stubGlobals({
      Worker: class {
        constructor(url, options) { Object.assign(this, { url: String(url), options, terminated: false }); workers.push(this); }
        postMessage(data, transfer) { this.sent = { data, transfer }; }
        terminate() { this.terminated = true; }
      },
    });
    const buffer = new ArrayBuffer(8);
    const first = loadPly(buffer, 5);
    const w = workers[0];
    expect(w.url).toEndWith('/viewer/static/lib/plyworker.js');
    expect(w.options).toEqual({ type: 'module' });
    expect(w.sent).toEqual({ data: { buffer, budget: 5 }, transfer: [buffer] });
    let prevented = false;
    w.onerror({ message: 'SyntaxError in plyworker.js', preventDefault: () => { prevented = true; } });
    await expect(first).rejects.toThrow('SyntaxError in plyworker.js');
    expect(prevented && w.terminated).toBe(true);
    const second = loadPly(new ArrayBuffer(8));
    workers[1].onerror({ preventDefault() {} });
    await expect(second).rejects.toThrow('the PLY reader could not be started');
    const third = loadPly(new ArrayBuffer(8));
    workers[2].onmessage({ data: { cloud: { header: { count: 0 } } } });
    expect(await third).toEqual({ header: { count: 0 } });
    expect(workers[2].terminated).toBe(true);
  });
});

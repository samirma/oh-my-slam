// lib/cloudview.js: a PLY a page holds (the server.sh web application's point-cloud results, spec
// §2.6 "Image"), read within the display budget, described in words, shown upright when its header
// names a single image's camera frame, and moved with the keys of a focused view.
import { describe, expect, test } from 'bun:test';
import * as view from '../../../src/oh_my_slam/viewer/static/lib/cloudview.js';
import { budgetNote, DISPLAY_POINT_BUDGET } from '../../../src/oh_my_slam/viewer/static/lib/controls.js';
import { parsePly, plyHeader } from '../../../src/oh_my_slam/viewer/static/lib/ply.js';
import { plyBytes } from '../helpers.js';

const { cloudFacts, IMAGE_FRAME, isUpright, keyMove, PAN_STEP, readCloud, ROTATE_STEP_DEG, ZOOM_STEP } = view;
const MAP_FRAME = 'oh-my-slam map frame (z up), metres';  // reconstruction.cloud.MAP_FRAME

test('it shows the viewer\'s display budget and reads PLY headers with the viewer\'s reader', () => {
  expect(view.budgetNote).toBe(budgetNote);
  expect(view.DISPLAY_POINT_BUDGET).toBe(DISPLAY_POINT_BUDGET);
  expect(view.plyHeader).toBe(plyHeader);
});

test('a single image\'s camera frame is shown upright; a map\'s frame as it is', () => {
  expect(isUpright({ comments: [IMAGE_FRAME, 'attributes color=rgb'] })).toBe(true);
  expect(isUpright({ comments: [MAP_FRAME] })).toBe(false);
  expect(isUpright({})).toBe(false);
  // a level camera: x right, z forward becomes +y, y down becomes -z
  expect(view.LEVEL_UPRIGHT).toEqual([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]);
});

describe('cloudFacts: what a text alternative of the cloud states', () => {
  const n = 5;
  const arrays = {
    position: new Float32Array(3 * n), normal: new Float32Array(3 * n),
    color: new Uint8Array(3 * n), label: new Int32Array(n),
  };

  test('the points, what each carries, the recorded attributes, the frame and the format', () => {
    const cloud = parsePly(plyBytes(arrays, { comments: [IMAGE_FRAME, 'attributes color=segment,normals=on,label=on'] }), 2);
    expect(cloudFacts(cloud)).toEqual({
      count: 2, total: 5, carries: ['position', 'colour (color=segment)', 'normal', 'object id'],
      attrs: 'color=segment,normals=on,label=on', frame: IMAGE_FRAME, format: 'binary_little_endian 1.0', upright: true,
      note: budgetNote(cloud.header), cameras: [],
    });
    expect(cloudFacts(cloud).note).toStartWith('Showing 2 of 5 points: evenly spaced');
  });

  test('a PLY of another writer: colour without a recorded mode, no frame, every point drawn', () => {
    const cloud = parsePly(plyBytes({ position: arrays.position, color: arrays.color }, { encoding: 'ascii', comments: ['made elsewhere'] }));
    expect(cloudFacts(cloud)).toEqual({
      count: 5, total: 5, carries: ['position', 'colour'], attrs: '', frame: '', format: 'ascii 1.0', upright: false, note: '',
      cameras: [],
    });
    expect(cloudFacts({ header: { count: 1, total: 1 }, arrays: { position: new Float32Array(3) } }))
      .toMatchObject({ carries: ['position'], attrs: '', frame: '', format: '', upright: false });
  });

  test('a map\'s frame is named', () => {
    const cloud = parsePly(plyBytes({ position: arrays.position }, { comments: [MAP_FRAME] }));
    expect(cloudFacts(cloud).frame).toBe(MAP_FRAME);
  });
});

describe('the cameras a PLY header names (mapper.sh locate)', () => {
  // mapping/locate.py camera_record: the frame's properties, transform and stream properties
  const record = (extra = {}, q = [0, 0, 0, 1]) => ({
    image: 'q.jpg', ...extra,
    transform_src_to_dst: { quaternion: q, translation: [1, 2, 3] },
    stream_properties: { intrinsics_pinhole: { width_px: 640, height_px: 480,
      camera_matrix: [500, 0, 320, 0, 0, 510, 240, 0, 0, 0, 1, 0], distortion_coeffs: [0, 0, 0, 0, 0] } },
  });

  test('a quaternion (scalar last) is the rotation it names, normalised', () => {
    expect(view.quaternionMatrix([0, 0, 0, 2])).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    const R = view.quaternionMatrix([0, 0, Math.SQRT1_2, Math.SQRT1_2]);  // 90 degrees about z
    expect(R.map((r) => r.map((v) => Math.round(v * 1e9) / 1e9 + 0))).toEqual([[0, -1, 0], [1, 0, 0], [0, 0, 1]]);
    expect(view.quaternionMatrix([0, 0, 0, 0])).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  });

  test('a record is a camera of cameras.js, of its kind', () => {
    const c = view.recordCamera('located_0', record({ located: true }));
    expect(c).toEqual({ name: 'located_0', kind: 'located', input: 1, T: [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
      position: [1, 2, 3], K: [500, 510, 320, 240], size: [640, 480], source: 'q.jpg' });
    // a located camera's input: its key's position (located_<k>), from 1
    expect(view.recordCamera('located_12', record({ located: true })).input).toBe(13);
    expect(view.recordCamera('located', record({ located: true })).input).toBeNull();
    expect(view.recordCamera('last_known', record({ last_known: true }))).toMatchObject({ kind: 'last_known', input: null });
    const { image, ...unnamed } = record();
    expect(view.recordCamera('f000001', unnamed)).toMatchObject({ kind: 'camera', source: null });
    for (const bad of [null, {}, { ...record(), transform_src_to_dst: { translation: [0, 0, 0] } },
      { ...record(), transform_src_to_dst: { quaternion: [0, 0, 0, 1] } }, { ...record(), stream_properties: {} }]) {
      expect(view.recordCamera('x', bad)).toBeNull();
    }
  });

  test('the header lines that record a camera, in order; other comments are left out', () => {
    const header = { comments: [MAP_FRAME, 'attributes color=rgb', `located_2 ${JSON.stringify(record({ located: true }))}`,
      'located_3 {not json}', 'note {"a": 1}', `last_known ${JSON.stringify(record({ last_known: true }))}`] };
    expect(view.headerCameras(header).map((c) => [c.name, c.kind])).toEqual([['located_2', 'located'], ['last_known', 'last_known']]);
    expect(view.headerCameras({})).toEqual([]);
    const cloud = parsePly(plyBytes({ position: new Float32Array(3) }, { comments: header.comments }));
    expect(cloudFacts(cloud).cameras).toEqual([{ name: 'located_2', kind: 'located', source: 'q.jpg' },
      { name: 'last_known', kind: 'last_known', source: 'q.jpg' }]);
  });
});

test('readCloud reads off the page thread within the display budget', async () => {
  const position = Float32Array.from({ length: 30 }, (_, i) => i);
  const cloud = await readCloud(plyBytes({ position }), 4);
  expect([cloud.header.count, cloud.header.total]).toEqual([4, 10]);
  expect([...cloud.arrays.position.slice(3, 6)]).toEqual([6, 7, 8]);  // vertex floor(1 * 2.5)
  const all = await readCloud(plyBytes({ position }));
  expect(all.header.count).toBe(10);
});

describe('keyMove: the keys of a focused view', () => {
  const key = (k, mods = {}) => keyMove({ key: k, ...mods });
  test('arrows rotate, Shift + arrows pan, + and - zoom, 0 or Home resets', () => {
    expect(key('ArrowLeft')).toEqual(['rotate', ROTATE_STEP_DEG, 0]);
    expect(key('ArrowRight')).toEqual(['rotate', -ROTATE_STEP_DEG, 0]);
    expect(key('ArrowUp')).toEqual(['rotate', -0, ROTATE_STEP_DEG]);
    expect(key('ArrowDown')).toEqual(['rotate', -0, -ROTATE_STEP_DEG]);
    expect(key('ArrowLeft', { shiftKey: true })).toEqual(['pan', -PAN_STEP, 0]);
    expect(key('ArrowUp', { shiftKey: true })).toEqual(['pan', 0, PAN_STEP]);
    for (const k of ['+', '=']) expect(key(k)).toEqual(['zoom', ZOOM_STEP]);
    for (const k of ['-', '_']) expect(key(k)).toEqual(['zoom', 1 / ZOOM_STEP]);
    for (const k of ['0', 'Home']) expect(key(k)).toEqual(['reset']);
  });
  test('other keys, and keys with Alt, Ctrl or Cmd, are left to the page', () => {
    for (const k of ['a', 'Enter', ' ', 'Tab', 'PageUp']) expect(key(k)).toBeNull();
    for (const mod of ['altKey', 'ctrlKey', 'metaKey']) expect(key('ArrowLeft', { [mod]: true })).toBeNull();
  });
  test('steps: 15° a rotation, a tenth of the view a pan, 1.25 times a zoom', () => {
    expect([ROTATE_STEP_DEG, PAN_STEP, ZOOM_STEP]).toEqual([15, 0.1, 1.25]);
  });
});

test('a CloudView makes the move a key asks for, and says whether it moved', () => {
  const calls = [];
  const fake = { rotate: (...a) => calls.push(['rotate', ...a]), pan: (...a) => calls.push(['pan', ...a]),
    zoom: (...a) => calls.push(['zoom', ...a]), reset: (...a) => calls.push(['reset', ...a]) };
  const keyOf = view.CloudView.prototype.key;
  expect(keyOf.call(fake, { key: 'ArrowRight', shiftKey: true })).toBe(true);
  expect(keyOf.call(fake, { key: 'Home' })).toBe(true);
  expect(keyOf.call(fake, { key: 'x' })).toBe(false);
  expect(calls).toEqual([['pan', PAN_STEP, 0], ['reset']]);
});

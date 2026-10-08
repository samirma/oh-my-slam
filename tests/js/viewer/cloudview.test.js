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
      note: budgetNote(cloud.header),
    });
    expect(cloudFacts(cloud).note).toStartWith('Showing 2 of 5 points: evenly spaced');
  });

  test('a PLY of another writer: colour without a recorded mode, no frame, every point drawn', () => {
    const cloud = parsePly(plyBytes({ position: arrays.position, color: arrays.color }, { encoding: 'ascii', comments: ['made elsewhere'] }));
    expect(cloudFacts(cloud)).toEqual({
      count: 5, total: 5, carries: ['position', 'colour'], attrs: '', frame: '', format: 'ascii 1.0', upright: false, note: '',
    });
    expect(cloudFacts({ header: { count: 1, total: 1 }, arrays: { position: new Float32Array(3) } }))
      .toMatchObject({ carries: ['position'], attrs: '', frame: '', format: '', upright: false });
  });

  test('a map\'s frame is named', () => {
    const cloud = parsePly(plyBytes({ position: arrays.position }, { comments: [MAP_FRAME] }));
    expect(cloudFacts(cloud).frame).toBe(MAP_FRAME);
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

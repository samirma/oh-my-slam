// lib/plyworker.js: parsePly off the page's thread, its arrays transferred back; the reason when the
// bytes are not a PLY this viewer can draw.
import { afterAll, beforeAll, expect, test } from 'bun:test';
import { plyBytes, stubGlobals } from '../helpers.js';

const posted = [];
let restore;
beforeAll(async () => {
  restore = stubGlobals({ postMessage: (msg, transfer) => posted.push({ msg, transfer }), onmessage: null });
  await import('../../../src/oh_my_slam/viewer/static/lib/plyworker.js');
});
afterAll(() => restore());

test('a PLY comes back as its cloud, every array transferred', () => {
  const position = new Float32Array([0, 1, 2, 3, 4, 5]);
  const label = new Int32Array([0, 7]);
  self.onmessage({ data: { buffer: plyBytes({ position, label }), budget: 1 } });
  const { msg, transfer } = posted.pop();
  expect(msg.cloud.header).toMatchObject({ count: 1, total: 2, step: 2 });
  expect([...msg.cloud.arrays.position]).toEqual([0, 1, 2]);
  expect([...msg.cloud.arrays.label]).toEqual([0]);
  expect(transfer).toEqual([msg.cloud.arrays.position.buffer, msg.cloud.arrays.label.buffer]);
});

test('bytes that are no PLY come back as the reason', () => {
  self.onmessage({ data: { buffer: new TextEncoder().encode('{"not": "a ply"}').buffer, budget: 10 } });
  expect(posted.pop()).toEqual({ msg: { error: 'not a PLY file (it does not start with "ply")' }, transfer: undefined });
});

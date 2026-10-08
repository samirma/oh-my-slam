// js/result.js: a result as the response carried it (spec §2.6 "Image"): a scene description is
// shown in full with its objects listed (label, id, colour, score), a point cloud drawn in 3D with
// its PLY header below, an image as received; with the command's per-stage timings.
import { describe, expect, test } from 'bun:test';
import { isScene, plyHeader, plyHeaderText, resultKind, sceneObjects, SHOW_TEXT_MAX, stageShares } from '../../../src/oh_my_slam/web/static/js/result.js';
import { objectEntry, plyBytes } from '../helpers.js';

describe('a scene description and its objects', () => {
  const doc = { openlabel: { metadata: { schema_url: 'https://openlabel.asam.net/V1-0-0/schema/openlabel_json_schema.json' }, objects: {
    12: objectEntry(12, 'lamp', null, { score: 0.61 }),
    3: objectEntry(3, 'chair', [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
    7: { name: 'thing 7', object_data: {} },
  } } };

  test('an OpenLABEL document is a scene description; other JSON is not', () => {
    expect(isScene(doc)).toBe(true);
    for (const other of [null, 'text', 3, {}, { openlabel: 'x' }, { openlabel: null }, [1]]) expect(isScene(other)).toBe(false);
  });

  test('its objects by id: label, id, colour and score as the document states them', () => {
    expect(sceneObjects(doc)).toEqual([
      { id: 3, label: 'chair', hex: '#ffe119', score: 0.9 },
      { id: 7, label: 'thing 7', hex: null, score: null },  // no type: its name
      { id: 12, label: 'lamp', hex: '#dcbeff', score: 0.61 },
    ]);
    expect(sceneObjects({ openlabel: { objects: { 1: {} } } })).toEqual([{ id: 1, label: '', hex: null, score: null }]);
    expect(sceneObjects({ openlabel: {} })).toEqual([]);
    expect(sceneObjects(null)).toEqual([]);
  });
});

describe('stageShares: each stage\'s share of the total', () => {
  test('of the command\'s total, `total` apart', () => {
    const { total, parts } = stageShares([{ name: 'depth', ms: 600 }, { name: 'segment', ms: 300 }, { name: 'total', ms: 1200 }]);
    expect(total).toEqual({ name: 'total', ms: 1200 });
    expect(parts).toEqual([{ name: 'depth', ms: 600, share: 0.5 }, { name: 'segment', ms: 300, share: 0.25 }]);
  });
  test('of their sum when there is no total; a stage without a time takes none; shares stay within 0 and 1', () => {
    expect(stageShares([{ name: 'a', ms: 1 }, { name: 'b', ms: 3 }, { name: 'c', ms: null }]).parts.map((s) => s.share)).toEqual([0.25, 0.75, 0]);
    expect(stageShares([{ name: 'a', ms: 0 }]).parts[0].share).toBe(0);
    expect(stageShares([{ name: 'a', ms: 5 }, { name: 'b', ms: -1 }, { name: 'total', ms: 2 }]).parts.map((s) => s.share)).toEqual([1, 0]);
    expect(stageShares([])).toEqual({ total: undefined, parts: [] });
  });
});

describe('how a result is shown', () => {
  test('JSON or text in full, images as received, anything else (a PLY) as a download and, if it is one, a cloud', () => {
    expect(resultKind('application/json', 'json', 1000)).toBe('text');
    expect(resultKind('text/plain', '', 10)).toBe('text');
    expect(resultKind('application/octet-stream', 'json', 10)).toBe('text');
    expect(resultKind('image/png', 'png', 10)).toBe('image');
    expect(resultKind('application/x-ply', 'ply', 10)).toBe('other');
    expect(resultKind('application/octet-stream', '', 10)).toBe('other');
  });
  test('a text body above 8 MiB is offered as a download only', () => {
    expect(SHOW_TEXT_MAX).toBe(8 * 1024 * 1024);
    expect(resultKind('application/json', 'json', SHOW_TEXT_MAX)).toBe('text');
    expect(resultKind('application/json', 'json', SHOW_TEXT_MAX + 1)).toBe('other');
  });
});

describe('a PLY result\'s header', () => {
  const bytes = plyBytes({ position: new Float32Array(3000), color: new Uint8Array(3000) }, { comments: ['attributes color=rgb'] });

  test('its text up to end_header and what it declares', async () => {
    const ply = await plyHeader(new Blob([bytes]));
    expect(ply.header).toBe(['ply', 'format binary_little_endian 1.0', 'comment attributes color=rgb', 'element vertex 1000',
      'property float x', 'property float y', 'property float z', 'property uchar red', 'property uchar green',
      'property uchar blue', 'end_header', ''].join('\n'));
    expect(ply.format).toBe('binary_little_endian');
    expect(ply.elements).toEqual([{ name: 'vertex', count: 1000 }]);
    expect(plyHeaderText(ply, bytes.byteLength)).toEqual({
      summary: 'Its PLY header (binary_little_endian: element vertex 1,000)',
      note: 'The 14.6 KB of binary little endian point data after the header are in the download.',
    });
  });

  test('CRLF lines, a later element, and a file that ends with its header', async () => {
    const ply = await plyHeader(new Blob(['ply\r\nformat ascii 1.0\r\nelement vertex 2\r\nelement face 0\r\nend_header']));
    expect(ply.header).toBe('ply\r\nformat ascii 1.0\r\nelement vertex 2\r\nelement face 0\r\nend_header');
    expect(ply.elements).toEqual([{ name: 'vertex', count: 2 }, { name: 'face', count: 0 }]);
    expect(plyHeaderText({ header: 'ply\nend_header\n', format: '', elements: [] }, 20).summary).toBe('Its PLY header (: no element)');
  });

  test('a body that is not a PLY has none', async () => {
    expect(await plyHeader(new Blob(['{"openlabel": {}}']))).toBeNull();
    expect(await plyHeader(new Blob(['ply\nformat ascii 1.0\n']))).toBeNull();  // no end_header
    expect(await plyHeader(new Blob([]))).toBeNull();
  });
});

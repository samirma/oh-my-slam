// js/result.js: a result as the response carried it (spec §2.6 "Image"): a scene description is
// shown in full with its objects listed (label, id, colour, score), a point cloud drawn in 3D with
// its PLY header below, an image as received; with the command's per-stage timings.
import { describe, expect, test } from 'bun:test';
import { coord, imageAlt, imageFacts, isScene, lengthText, metadataCameras, noObjectsText, plyHeader, plyHeaderText, pngInfo, resultKind, scaleBar,
  sceneCameras, sceneObjects, SHOW_TEXT_MAX, stageShares } from '../../../src/oh_my_slam/web/static/js/result.js';
import { Operation } from '../../../src/oh_my_slam/web/static/js/store.js';
import { concat, objectEntry, openapiDocument, plyBytes } from '../helpers.js';

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

describe('a scene description\'s cameras', () => {
  // as mapping/export.py (a map's keyframes) and mapping/locate.py (located, last known) write them
  const frame = (src, t, props = {}) => ({ frame_properties: { timestamp: 0, streams: { [src]: { uri: `frames/${src}.jpg` } },
    transforms: { [`${src}_to_map`]: { src, dst: 'map', transform_src_to_dst: { quaternion: [0, 0, 0, 1], translation: t } } }, ...props } });
  test('located and last known cameras first, then the map\'s frames, each with its position', () => {
    const doc = { openlabel: { frames: {
      0: frame('camera_1', [0, 0, 0], { keyframe: 'f000000' }),
      3: frame('camera_1', [1, 0.0002, -2], { keyframe: 'f000003' }),
      7: frame('last_known', [1, 0, -2], { last_known: true, keyframe: 'f000003', image: 'frames/f000003.jpg' }),
      9: frame('located_1', [0.5, 0.25, 0], { located: true, image: '/in/q.jpg' }),
      8: frame('located_0', [2, 2, 2], { located: true, image: '/in/p.jpg' }),
      2: { frame_properties: { timestamp: 2 } },  // no pose: not a camera
      4: frame('camera_2', [0, 1, 0]),  // a frame without its keyframe's name
    } } };
    expect(sceneCameras(doc)).toEqual([
      { frame: 8, name: 'located_0', kind: 'located', position: [2, 2, 2], image: '/in/p.jpg' },
      { frame: 9, name: 'located_1', kind: 'located', position: [0.5, 0.25, 0], image: '/in/q.jpg' },
      { frame: 7, name: 'last_known', kind: 'last_known', position: [1, 0, -2], image: 'frames/f000003.jpg' },
      { frame: 0, name: 'f000000', kind: 'frame', position: [0, 0, 0], image: 'frames/camera_1.jpg' },
      { frame: 3, name: 'f000003', kind: 'frame', position: [1, 0.0002, -2], image: 'frames/camera_1.jpg' },
      { frame: 4, name: 'frame 4', kind: 'frame', position: [0, 1, 0], image: 'frames/camera_2.jpg' },
    ]);
    const bare = { openlabel: { frames: { 1: { frame_properties: { transforms: { a: { src: 'a', transform_src_to_dst: {} } } } }, 2: null } } };
    expect(sceneCameras(bare)).toEqual([{ frame: 1, name: 'frame 1', kind: 'frame', position: [], image: '' }]);
    expect(sceneCameras({ openlabel: {} })).toEqual([]);
    expect(sceneCameras(null)).toEqual([]);
  });
  test('a coordinate in metres, to the millimetre', () => {
    expect([coord(1.23456), coord(-0.0002), coord(-2), coord(undefined)]).toEqual(['1.235', '0.000', '-2.000', '—']);
  });
});

describe('a scene without objects says why', () => {
  const locate = new Operation('mapper-locate', openapiDocument().paths['/api/ops/mapper-locate'].post);
  test('a result of a scope that lists no objects: what it covers, and what the other scope does', () => {
    const single = { openlabel: { metadata: { scope: 'single' }, objects: {} } };
    expect(noObjectsText(single, locate)).toBe("No objects in this result: its scope is single (the cameras' poses only, with no "
      + 'objects). Choose scope full (the whole map plus the cameras) to include them.');
    const alone = new Operation('mapper-locate', openapiDocument().paths['/api/ops/mapper-locate'].post);
    alone.params = [{ name: 'mode', choices: ['single'], choice_text: { single: 'only these cameras' } }];  // no other scope
    expect(noObjectsText(single, alone)).toBe('No objects in this result: its mode is single (only these cameras).');
  });
  test('otherwise, simply none', () => {
    expect(noObjectsText({ openlabel: { metadata: {} } }, locate)).toBe('No objects.');
    expect(noObjectsText({ openlabel: { metadata: { scope: 'odd' } } }, locate)).toBe('No objects.');
    expect(noObjectsText({ openlabel: { metadata: { scope: 'single' } } }, null)).toBe('No objects.');
    expect(noObjectsText(null, locate)).toBe('No objects.');
  });
});

describe('a bird\'s-eye image\'s cameras, from its metadata', () => {
  test('located and last known first, then the keyframes by name, each with its position', () => {
    const t = (x) => ({ transform_src_to_dst: { quaternion: [0, 0, 0, 1], translation: [x, 0, 0] } });
    expect(metadataCameras({ f000010: { keyframe: 'f000010', frame: 10, image: 'frames/f000010.jpg', ...t(3) },
      f000002: { keyframe: 'f000002', frame: 2, ...t(2) }, located_1: { located: true, image: '/in/b.jpg', ...t(1) },
      last_known: { last_known: true, image: 'frames/f000010.jpg', ...t(3) }, odd: null })).toEqual([
      { frame: null, name: 'located_1', kind: 'located', position: [1, 0, 0], image: '/in/b.jpg' },
      { frame: null, name: 'last_known', kind: 'last_known', position: [3, 0, 0], image: 'frames/f000010.jpg' },
      { frame: 2, name: 'f000002', kind: 'frame', position: [2, 0, 0], image: '' },
      { frame: 10, name: 'f000010', kind: 'frame', position: [3, 0, 0], image: 'frames/f000010.jpg' },
      { frame: null, name: 'odd', kind: 'frame', position: [], image: '' },
    ]);
    expect(metadataCameras(null)).toEqual([]);
  });
});

// A PNG's bytes, its chunks as given ([type, bytes]; the CRC is not checked by the reader)
function png(chunks) {
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])];
  for (const [type, data] of chunks) {
    const head = new Uint8Array(8);
    new DataView(head.buffer).setUint32(0, data.length);
    head.set(new TextEncoder().encode(type), 4);
    parts.push(head, data, new Uint8Array(4));
  }
  return concat(...parts);
}
const bytes = (...xs) => new Uint8Array(xs.flatMap((x) => (typeof x === 'string' ? [...new TextEncoder().encode(x)] : x)));
function ihdr(w, h) { const d = new Uint8Array(13); new DataView(d.buffer).setUint32(0, w); new DataView(d.buffer).setUint32(4, h); return d; }

describe('a PNG result\'s own metadata', () => {
  test('its size and its tEXt and uncompressed iTXt chunks', () => {
    const info = pngInfo(png([['IHDR', ihdr(550, 480)], ['tEXt', bytes('layers', [0], 'cloud,obb')],
      ['iTXt', bytes('map', [0, 0, 0], 'en', [0], [0], 'maps/café')], ['iTXt', bytes('zipped', [0, 1, 0], [0], [0], 'xx')],
      ['iTXt', bytes('broken', [0, 0, 0], 'no end')], ['tEXt', bytes([0], 'no keyword')], ['IDAT', bytes([1, 2, 3])],
      ['IEND', bytes()], ['tEXt', bytes('after', [0], 'the end')]]));
    expect(info).toEqual({ width: 550, height: 480, text: { layers: 'cloud,obb', map: 'maps/café' } });
  });
  test('bytes that are no PNG have none; a PNG without metadata has its size only', () => {
    expect(pngInfo(new TextEncoder().encode('{"openlabel": {}}').buffer)).toBeNull();
    expect(pngInfo(new ArrayBuffer(3))).toBeNull();
    expect(pngInfo(png([['IHDR', ihdr(4, 2)]]))).toEqual({ width: 4, height: 2, text: {} });
    expect(pngInfo(png([['IHDR', new Uint8Array(4)]]))).toEqual({ width: null, height: null, text: {} });  // a short IHDR
  });
  test('what the page shows of it: description, scale, layers, legend and cameras', () => {
    const legend = [{ mark: 'located', shape: 'triangle', fill: '#ffffff', edge: '#000000', text: 'a located camera' }, { mark: 'x' }];
    const facts = imageFacts({ Description: 'the map from above', metres_per_pixel: '0.02', layers: 'cloud,obb',
      legend: JSON.stringify(legend), cameras: JSON.stringify({ last_known: { last_known: true } }) });
    expect(facts).toEqual({ description: 'the map from above', mpp: 0.02, layers: ['cloud', 'obb'], legend: [legend[0]],
      cameras: [{ frame: null, name: 'last_known', kind: 'last_known', position: [], image: '' }] });
    expect(imageFacts({})).toEqual({ description: null, mpp: null, layers: null, legend: null, cameras: null });
    expect(imageFacts({ metres_per_pixel: 'x', legend: '{}', cameras: '[]' })).toEqual({ description: null, mpp: null, layers: null,
      legend: null, cameras: null });
    expect(imageFacts(undefined).mpp).toBeNull();
    expect(imageFacts({ metres_per_pixel: '-1' }).mpp).toBeNull();
  });
  test('a scale bar of a 1, 2 or 5 step within a quarter of the image', () => {
    expect(scaleBar(0.01, 550)).toEqual({ metres: 1, label: '1 m', percent: (1 / 0.01 / 550) * 100 });
    expect(scaleBar(0.01, 300).label).toBe('50 cm');
    expect(scaleBar(0.1, 2048).label).toBe('50 m');
    expect(scaleBar(0.02, 1000).metres).toBe(5);
    expect([scaleBar(null, 100), scaleBar(0.01, null), scaleBar(0, 100)]).toEqual([null, null, null]);
    expect([lengthText(0.01), lengthText(0.5), lengthText(2), lengthText(0.005)]).toEqual(['1 cm', '50 cm', '2 m', '0.5 cm']);
  });
  test('its text alternative: what it is, its size, its own description', () => {
    const facts = imageFacts({ Description: 'Seen from above.' });
    expect(imageAlt("the bird's-eye image of the map", { width: 5, height: 4 }, facts))
      .toBe("The bird's-eye image of the map, as received, 5 × 4 px. Seen from above.");
    expect(imageAlt('', null, imageFacts({}))).toBe('The result image, as received');
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

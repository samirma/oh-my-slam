// lib/cameras.js: the cameras of the scene (api/meta, viewer/bundle.py scene_cameras) as frustums
// at their poses, listed with their centres, and the viewpoint "Go to" moves to (spec §2.5).
import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';
import { buildFrustums, cameraRows, cameraView, fmtCoord, FRAME_COLOR, poseMatrix } from '../../../src/oh_my_slam/viewer/static/lib/cameras.js';

// a camera as api/meta lists it: T camera-to-scene (row-major, OpenCV axes)
function camera(T, extra = {}) {
  return { name: 'f000000', T, position: [T[0][3], T[1][3], T[2][3]], K: [500, 500, 320, 240], size: [640, 480], source: 'f0.jpg', ...extra };
}
const IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
// 90° about z, then 1, 2, 3 metres away: the camera's x axis is the scene's y
const TURNED = [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]];
const round = (v, digits = 9) => v.toArray().map((x) => +x.toFixed(digits) + 0);

test('poseMatrix: the row-major camera-to-scene transform', () => {
  const M = poseMatrix(camera(TURNED));
  expect(round(new THREE.Vector3(0, 0, 0).applyMatrix4(M))).toEqual([1, 2, 3]);  // the centre
  expect(round(new THREE.Vector3(1, 0, 0).applyMatrix4(M))).toEqual([1, 3, 3]);  // camera x: scene y
});

describe('buildFrustums', () => {
  test('none without cameras', () => {
    expect(buildFrustums([], 3).children).toEqual([]);
  });

  test('a frustum per camera at its pose: four rays to the image corners and the image outline', () => {
    const g = buildFrustums([camera(TURNED)], 10);
    const [lines] = g.children;
    expect(lines.name).toBe('frustums');
    expect(lines.frustumCulled).toBe(false);
    expect(`#${lines.material.color.getHexString(THREE.SRGBColorSpace)}`).toBe(FRAME_COLOR);
    const p = lines.geometry.attributes.position;
    expect(p.count).toBe(16);  // 8 segments
    const d = 0.5;  // one camera: 5 % of the scene size
    const M = poseMatrix(camera(TURNED));
    const corner = (u, v) => round(new THREE.Vector3((u - 320) / 500 * d, (v - 240) / 500 * d, d).applyMatrix4(M), 6);
    const at = (i) => round(new THREE.Vector3().fromBufferAttribute(p, i), 6);  // float32
    expect([0, 4, 8, 12].map(at)).toEqual(Array(4).fill([1, 2, 3]));  // every ray starts at the centre
    expect([1, 5, 9, 13].map(at)).toEqual([corner(0, 0), corner(640, 0), corner(640, 480), corner(0, 480)]);
    expect([3, 7, 11, 15].map(at)).toEqual([corner(640, 0), corner(640, 480), corner(0, 480), corner(0, 0)]);
  });

  test('frustums are smaller among several cameras, and never tiny', () => {
    const depth = (g) => {
      const p = g.children[0].geometry.attributes.position;
      return new THREE.Vector3().fromBufferAttribute(p, 1).z;  // identity pose: the corner's z
    };
    expect(depth(buildFrustums([camera(IDENTITY), camera(IDENTITY)], 10))).toBeCloseTo(0.25, 6);
    expect(depth(buildFrustums([camera(IDENTITY)], 0.1))).toBeCloseTo(0.05, 6);
  });
});

describe('cameraView: the viewpoint at a camera', () => {
  test('at its centre, along its optical axis, with the field of view of its whole image', () => {
    const view = cameraView(camera(IDENTITY), new THREE.Matrix4(), new THREE.Box3(), 4 / 3);
    expect(round(view.eye)).toEqual([0, 0, 0]);
    expect(round(view.target)).toEqual([0, 0, 1]);  // no scene box: 1 m ahead
    expect(view.fov).toBeCloseTo(2 * Math.atan(240 / 500) * 180 / Math.PI, 9);
    // a narrower view fits the image's width
    expect(cameraView(camera(IDENTITY), new THREE.Matrix4(), new THREE.Box3(), 0.5).fov)
      .toBeCloseTo(2 * Math.atan(320 / 500 / 0.5) * 180 / Math.PI, 9);
    // a long lens is not narrower than 5°, a fisheye not wider than 150°
    expect(cameraView(camera(IDENTITY, { K: [50000, 50000, 320, 240] }), new THREE.Matrix4(), new THREE.Box3(), 1).fov).toBe(5);
    expect(cameraView(camera(IDENTITY, { K: [10, 10, 320, 240] }), new THREE.Matrix4(), new THREE.Box3(), 1).fov).toBe(150);
  });

  test('in the display frame, its pivot towards the middle of the scene (0.5 m at least)', () => {
    const display = new THREE.Matrix4().makeTranslation(0, 0, 10);
    const box = new THREE.Box3(new THREE.Vector3(-1, -1, 11), new THREE.Vector3(1, 1, 15));  // centre 3 m ahead
    const view = cameraView(camera(IDENTITY), display, box, 1);
    expect(round(view.eye)).toEqual([0, 0, 10]);
    expect(round(view.target)).toEqual(round(new THREE.Vector3(0, 0, 10 + Math.sqrt(24) / 2)));  // at most half the box's diagonal
    const wide = new THREE.Box3(new THREE.Vector3(-5, -5, 11), new THREE.Vector3(5, 5, 15));
    expect(round(cameraView(camera(IDENTITY), display, wide, 1).target)).toEqual([0, 0, 13]);  // the middle, 3 m ahead
    const behind = new THREE.Box3(new THREE.Vector3(-1, -1, 5), new THREE.Vector3(1, 1, 6));
    expect(round(cameraView(camera(IDENTITY), display, behind, 1).target)).toEqual([0, 0, 10.5]);
  });
});

test('coordinates in metres to the millimetre, with no "-0.000"', () => {
  expect(fmtCoord(1.23456)).toBe('1.235');
  expect(fmtCoord(-2)).toBe('-2.000');
  expect(fmtCoord(-0.0004)).toBe('0.000');
  expect(fmtCoord(0.0004)).toBe('0.000');
  expect(fmtCoord(-0.0006)).toBe('-0.001');
});

test('the camera table: every camera with its centre, and its image when that is not its name', () => {
  const rows = cameraRows([
    camera(TURNED, { name: 'f000003', source: 'IMG_0042.jpg' }),
    camera(IDENTITY, { name: 'photo.jpg', source: 'photo.jpg', position: [-0.0001, 1e-9, 12.3456] }),
    camera(IDENTITY, { name: 'f000007', source: '' }),
  ]);
  expect(rows).toEqual([
    { name: 'f000003', source: 'IMG_0042.jpg', coords: ['1.000', '2.000', '3.000'] },
    { name: 'photo.jpg', source: null, coords: ['0.000', '0.000', '12.346'] },
    { name: 'f000007', source: null, coords: ['0.000', '0.000', '0.000'] },
  ]);
  expect(cameraRows([])).toEqual([]);
});

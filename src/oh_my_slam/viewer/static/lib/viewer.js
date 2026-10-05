// The 3D view (spec §2.5), independent of any page: put it in a host element, give it a cloud
// (cloud.js), the objects of a scene (obbs.js) and cameras (cameras.js). It draws the point-cloud,
// segmentation, camera, label and OBB layers, frames the scene, moves to a camera, and highlights a
// selected object. view.sh's page (app.js) and the server.sh web application use it alike.
//
// Frames are drawn on demand: an idle view draws nothing, so that a large cloud does not keep the
// GPU busy next to the inference server. Whatever changes the canvas calls invalidate(); the loop
// also draws whenever the viewpoint moved (orbiting and its damping, go-to).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { buildCloud, robustBox } from './cloud.js';
import { buildObb, inkFor, OBB_WIDTH_PX } from './obbs.js';
import { buildFrustums, cameraView, LOCATED_COLOR } from './cameras.js';
import { LabelLayer } from './labels.js';

const BACKGROUND = '#15171c';
const DEFAULT_FOV = 55;
const REDRAW_FRAMES = 2;  // frames drawn after each change: a margin for anything that lands late
// The initial view: a bird's-eye three-quarter view, HOME_PITCH below the horizon, of the whole
// scene (the points' 2nd-98th percentile box and the camera centres near it), from the south-west
// (HOME_AZIMUTH) so that the up axis stays vertical on screen.
const HOME_PITCH = 60, HOME_AZIMUTH = new THREE.Vector2(-1, -1.2).normalize();
// camera centres further than this many diagonals of the cloud's box from it do not widen the
// initial view (a keyframe posed far off would shrink the whole cloud to a dot); they are drawn
const HOME_CAMERA_REACH = 2;
// A viewpoint change beyond VIEW_EPS (1 µm, 1 µrad: far below a pixel) moves the labels and draws.
// Orbit damping's last creep stays below it, so a settled view stops drawing.
const VIEW_EPS = 1e-6;
const SELECTED_WIDTH_PX = 2.5 * OBB_WIDTH_PX;
export const GROUPS = ['points', 'segments', 'cameras', 'obbs', 'labels'];

function differs(a, b) {
  const x = a.elements, y = b.elements;
  for (let i = 0; i < 16; i++) if (Math.abs(x[i] - y[i]) > VIEW_EPS) return true;
  return false;
}

export class Viewer {
  constructor(host) {
    this.host = host;
    this.layers = { points: true, segments: false, cameras: true, labels: true, obbs: true };
    this.objects = [];          // objects with a box: sceneObjects() entries + line, anchor, label
    this.objectRgb = new Map(); // id -> [r, g, b] of every object (the segmentation layer)
    this.cameras = [];
    this.cloud = null;          // header of the cloud on screen
    this.bbox = new THREE.Box3();
    this.cloudBox = new THREE.Box3();
    this.selected = null;
    this.frames = 0;            // frames drawn so far
    this.cloudDrawn = false;    // a frame showing a cloud has been drawn
    this._selectListeners = new Set();
    this._crowdedListeners = new Set();
    this._redraw = 1;           // the first frame: the empty view's background
    this._labelsDirty = true;
    this._homePending = false;

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputColorSpace = THREE.SRGBColorSpace;  // material colours are converted back exactly
    renderer.toneMapping = THREE.NoToneMapping;         // no tone mapping: colours stay exact
    host.appendChild(renderer.domElement);
    this.renderer = renderer;
    this.labels = new LabelLayer(host);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BACKGROUND);
    const camera = new THREE.PerspectiveCamera(DEFAULT_FOV, 1, 0.01, 5000);
    camera.up.set(0, 0, 1);  // map and display frames are z-up
    camera.position.set(-3, -3.6, 2.7);  // until the cloud is framed (e.g. an empty map)
    this.camera = camera;
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.screenSpacePanning = true;
    this.controls = controls;
    controls.addEventListener('change', () => this.invalidate());
    for (const type of ['pointerdown', 'pointerup', 'wheel', 'keydown', 'input', 'change', 'click']) {
      window.addEventListener(type, () => this.invalidate(), { capture: true, passive: true });
    }
    document.addEventListener('visibilitychange', () => this.invalidate());
    renderer.domElement.addEventListener('webglcontextrestored', () => this.invalidate());

    this.root = new THREE.Group();  // display transform (an image: camera frame -> z-up)
    this.root.matrixAutoUpdate = false;
    this.scene.add(this.root);
    this.groups = {};
    for (const k of GROUPS) {
      this.groups[k] = new THREE.Group();
      this.groups[k].name = k;
      this.root.add(this.groups[k]);
    }
    this._lastView = new THREE.Matrix4();
    this._lastProj = new THREE.Matrix4();
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    const loop = () => { requestAnimationFrame(loop); this._frame(); };
    requestAnimationFrame(loop);
  }

  invalidate(frames = REDRAW_FRAMES) { this._redraw = Math.max(this._redraw, frames); }

  // ------------------------------------------------------------ frame
  setDisplayTransform(rowsMajor) {
    this.root.matrix.copy(new THREE.Matrix4().fromArray(rowsMajor.flat()).transpose());
    this.root.updateMatrixWorld(true);
    this.invalidate();
  }

  focalPx() {
    const h = Math.max(this.host.clientHeight, 1);
    return (h * this.renderer.getPixelRatio()) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
  }

  _pointMaterials() {
    const out = [];
    for (const g of [this.groups.points, this.groups.segments]) g.traverse((m) => { if (m.material?.uniforms?.focal) out.push(m.material); });
    return out;
  }

  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
    for (const o of this.objects) o.line.material.resolution.set(w, h);
    const focal = this.focalPx();
    for (const m of this._pointMaterials()) m.uniforms.focal.value = focal;
    this._labelsDirty = true;
    this.invalidate();  // resizing the canvas clears it
    if (this._homePending && w > 0 && h > 0) this.resetView();
  }

  setFov(fov) {
    this.camera.fov = fov;
    this.camera.updateProjectionMatrix();
    const focal = this.focalPx();
    for (const m of this._pointMaterials()) m.uniforms.focal.value = focal;
    this._labelsDirty = true;
    this.invalidate();
  }

  // ------------------------------------------------------------ content
  _clear(g) {
    for (const c of [...g.children]) {
      g.remove(c);
      c.traverse((x) => { x.geometry?.dispose(); x.material?.dispose(); });
    }
  }

  // The cloud (data.js structure) of the point-cloud and segmentation layers. The first cloud
  // sets the box the initial view frames.
  setCloud(cloud) {
    this._clear(this.groups.points);
    this._clear(this.groups.segments);
    const { points, segments } = buildCloud(cloud, this.objectRgb, this.focalPx());
    this.groups.points.add(points);
    if (segments) this.groups.segments.add(segments);
    this.cloud = cloud.header;
    const pos = cloud.arrays.position;
    if (this.cloudBox.isEmpty() && pos.length) {
      this.cloudBox = robustBox(pos, this.root.matrixWorld);
      this._updateBox();
    }
    this.invalidate();
  }

  // The objects of a scene (sceneObjects): colours for the segmentation layer, and a box and a
  // label for each one with a cuboid.
  setObjects(objects) {
    this._clear(this.groups.obbs);
    this.labels.remove('labels');
    this.objectRgb = new Map(objects.filter((o) => o.rgb).map((o) => [o.id, o.rgb]));
    this.objects = [];
    const [w, h] = [this.host.clientWidth, this.host.clientHeight];
    for (const obj of objects) {
      if (!obj.cuboid) continue;
      const { line, anchor, size } = buildObb(obj, this.root.matrixWorld);
      line.material.resolution.set(w, h);
      this.groups.obbs.add(line);
      const o = Object.assign({}, obj, { line, anchor, size });
      this.labels.add(Object.assign(o, {
        tag: String(obj.id), name: obj.label, background: obj.hex, ink: inkFor(obj.rgb || [128, 128, 128]),
        title: `${obj.label} ${obj.id}`, note: `${obj.id} ${obj.label}`, group: 'labels',
      }));
      this.objects.push(o);
    }
    this._applySelection();
    this._labelsDirty = true;
    this.invalidate();
  }

  // The cameras' frustums (map frames solid, located cameras dashed, in their own colour and
  // labelled "located" with their image name).
  setCameras(cams) {
    this._clear(this.groups.cameras);
    this.labels.remove('cameras');
    this.cameras = cams;
    const size = this.cloudBox.isEmpty() ? 1 : this.cloudBox.getSize(new THREE.Vector3()).length();
    for (const lines of [...buildFrustums(cams, size).children]) this.groups.cameras.add(lines);  // frames, then located
    const ink = inkFor([255, 176, 46]);
    cams.forEach((f, i) => {
      if (!f.located) return;
      const c = new THREE.Vector3(f.T[0][3], f.T[1][3], f.T[2][3]).applyMatrix4(this.root.matrixWorld);
      this.labels.add({ id: 1e9 + i, tag: 'located', name: f.source || f.name, background: LOCATED_COLOR,
        ink, anchor: c, size: size * 0.05, title: `located camera ${f.name}`, note: `located ${f.source || f.name}`,
        group: 'cameras' });
    });
    this._updateBox();
    this._labelsDirty = true;
    this.invalidate();
  }

  _updateBox() {
    this.bbox = this.cloudBox.clone();
    const reach = this.cloudBox.isEmpty() ? null
      : this.cloudBox.clone().expandByScalar(HOME_CAMERA_REACH * this.cloudBox.getSize(new THREE.Vector3()).length());
    for (const f of this.cameras) {
      const c = new THREE.Vector3(f.T[0][3], f.T[1][3], f.T[2][3]).applyMatrix4(this.root.matrixWorld);
      if (!reach || reach.containsPoint(c)) this.bbox.expandByPoint(c);
    }
  }

  setLayer(key, on) {
    this.layers[key] = on;
    for (const k of GROUPS) this.groups[k].visible = this.layers[k];
    this._labelsDirty = true;
    this.invalidate();
  }

  // ------------------------------------------------------------ selection (API only)
  // select(id) highlights that object's box and label (null: none) and tells every onSelect
  // listener, so that a host page can highlight it in its other views too.
  select(id) {
    const next = id == null ? null : Number(id);
    if (next === this.selected) return;
    this.selected = next;
    this._applySelection();
    for (const cb of this._selectListeners) cb(next);
  }

  onSelect(cb) { this._selectListeners.add(cb); return () => this._selectListeners.delete(cb); }

  _applySelection() {
    for (const o of this.objects) {
      const on = o.id === this.selected;
      o.line.material.linewidth = on ? SELECTED_WIDTH_PX : OBB_WIDTH_PX;
      o.line.renderOrder = on ? 2 : 0;
      o.div.classList.toggle('selected', on);
    }
    this.invalidate();
  }

  // onCrowded(cb): cb(items) after every label layout, with the labels that found no room.
  onCrowded(cb) { this._crowdedListeners.add(cb); return () => this._crowdedListeners.delete(cb); }

  // ------------------------------------------------------------ viewpoint
  _placeView(eye, target, near, far) {
    this.camera.position.copy(eye);
    this.controls.target.copy(target);
    this.camera.near = near; this.camera.far = far;
    this.camera.updateProjectionMatrix();
    this.controls.enableDamping = false;  // no leftover orbit inertia moves the camera afterwards
    this.controls.update();
    this.controls.enableDamping = true;
  }

  // the distance from `target` along `dir` (unit, target -> eye) at which all corners of `box`
  // lie inside the viewport, with `pad` of margin
  _fitDistance(box, target, dir, pad = 1.06) {
    const tanV = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) / pad;
    const tanH = tanV * this.camera.aspect;
    const fwd = dir.clone().negate();
    const right = new THREE.Vector3().crossVectors(fwd, this.camera.up).normalize();
    const up = new THREE.Vector3().crossVectors(right, fwd).normalize();
    let dist = 0.1;
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
      const rel = new THREE.Vector3(x, y, z).sub(target);
      const along = rel.dot(dir);  // towards the eye
      dist = Math.max(dist, Math.abs(rel.dot(right)) / tanH + along, Math.abs(rel.dot(up)) / tanV + along);
    }
    return dist;
  }

  // The initial view. A view without a size (a hidden tab or pane) has no aspect to fit to: it is
  // framed on the first resize that gives it one.
  resetView() {
    this.setFov(DEFAULT_FOV);
    this._homePending = !(this.host.clientWidth > 0 && this.host.clientHeight > 0);
    if (this._homePending || this.bbox.isEmpty()) return;
    const box = this.bbox;
    const pitch = THREE.MathUtils.degToRad(HOME_PITCH);
    const dir = new THREE.Vector3(HOME_AZIMUTH.x * Math.cos(pitch), HOME_AZIMUTH.y * Math.cos(pitch), Math.sin(pitch));
    const target = box.getCenter(new THREE.Vector3());
    const dist = this._fitDistance(box, target, dir);
    const size = box.getSize(new THREE.Vector3()).length();
    this._placeView(target.clone().addScaledVector(dir, dist), target, Math.max(dist / 1000, 0.005),
      dist * 50 + size * 10);
  }

  // Move the viewpoint to camera `i`: its centre, looking along its optical axis, its whole image
  // in view.
  goToCamera(i) {
    const f = this.cameras[i];
    if (!f) return;
    const to = cameraView(f, this.root.matrixWorld, this.bbox, this.camera.aspect);
    this.setFov(to.fov);
    this._placeView(to.eye, to.target, 0.01, Math.max(this.camera.far, 1000));
  }

  // ------------------------------------------------------------ loop
  _frame() {
    this.controls.update();
    this.camera.updateMatrixWorld();
    if (differs(this._lastView, this.camera.matrixWorld) || differs(this._lastProj, this.camera.projectionMatrix)) {
      this._lastView.copy(this.camera.matrixWorld);
      this._lastProj.copy(this.camera.projectionMatrix);
      this._labelsDirty = true;
      this.invalidate();
    }
    if (this._labelsDirty) {
      this._labelsDirty = false;
      const crowded = this.labels.layout(this.camera, this.host.clientWidth, this.host.clientHeight,
        (g) => this.layers[g] && (g !== 'cameras' || this.layers.labels));
      for (const cb of this._crowdedListeners) cb(crowded);
    }
    if (this._redraw > 0) {
      this._redraw--;
      this.renderer.render(this.scene, this.camera);
      this.frames++;
      if (this.cloud) this.cloudDrawn = true;
    }
  }
}

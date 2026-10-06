// What every page shares: the operations (read from /api/openapi.json, never listed here), the
// service's health (polled: the inference server, the requests running and waiting), and the
// request this page runs, if any (it belongs to the page: leaving the page interrupts it).
import { getJson } from './api.js';

const HEALTH_POLL_MS = 3000;
const HEALTH_POLL_BUSY_MS = 1000;  // while this page's request runs or waits

// One operation per command mode, as the OpenAPI document describes it: the registry entry of the
// mode (`x-oms`) and of each parameter, in the order of the command's options.
export class Operation {
  constructor(id, post) {
    this.id = id;
    this.post = post;
    this.x = post['x-oms'];
    const schema = post.requestBody.content['application/json'].schema;
    this.params = Object.values(schema.properties).map((s) => s['x-oms']);
  }

  get label() { return this.x.id; }
  get description() { return this.x.description; }
  param(name) { return this.params.find((p) => p.name === name); }
  ofKind(...kinds) { return this.params.filter((p) => kinds.includes(p.kind)); }
  // the parameter naming a map the mode writes (an output written via that option)
  get writesMap() {
    const flags = new Set(this.x.outputs.map((o) => o.via));
    return this.params.find((p) => p.kind === 'map' && flags.has(p.flag)) || null;
  }
  // its single image input, if it takes one and no map (the Image page's operations)
  get singleImage() {
    const images = this.ofKind('image');
    return images.length === 1 && !this.mapParam ? images[0] : null;
  }
  get mapParam() { return this.ofKind('map')[0] || null; }
  // the outputs the response carries (what the command writes to stdout)
  get results() { return this.x.outputs.filter((o) => o.via === 'stdout'); }
}

export const PATH_IN = ['image', 'images', 'images_or_video', 'map'];

class Store {
  constructor() {
    this.ops = new Map();
    this.health = null;
    this.active = null;  // this page's request in progress (a RequestView)
    this._listeners = new Set();
  }

  on(cb) { this._listeners.add(cb); return () => this._listeners.delete(cb); }
  emit(what, data) { for (const cb of [...this._listeners]) cb(what, data); }

  async start() {
    const doc = await getJson('/api/openapi.json');
    for (const [path, item] of Object.entries(doc.paths)) {
      const m = /^\/api\/ops\/([^/]+)$/.exec(path);
      if (m && item.post && item.post['x-oms']) this.ops.set(m[1], new Operation(m[1], item.post));
    }
    this.pollHealth();
  }

  async pollHealth() {
    clearTimeout(this._healthTimer);
    const asked = performance.now();
    try {
      this.health = await getJson('/api/health');
      this.health.askedAt = asked;
    } catch (err) {
      this.health = { status: 'unreachable', message: err.message };
    }
    this.emit('health', this.health);
    this._healthTimer = setTimeout(() => this.pollHealth(), this.active ? HEALTH_POLL_BUSY_MS : HEALTH_POLL_MS);
  }

  // whether the inference server can take requests (a loading server is fine: commands wait)
  get inferenceUp() {
    const s = this.health?.inference?.status;
    return s === 'ready' || s === 'loading';
  }

  get startCommand() { return this.health?.inference?.start_command || null; }

  // the requests in progress on the service, in arrival order
  get inProgress() { return this.health?.service?.in_progress || []; }
}

export const store = new Store();

// Why `op` cannot run now (the inference server is down and the mode always needs it), else null;
// a conditional need is decided by the service's own check (/validate) for the actual request.
export function blockedReason(op) {
  if (!store.health || store.health.status === 'unreachable' || store.inferenceUp || op.x.inference !== 'required') return null;
  const cmd = store.startCommand;
  return `${op.label} ${op.x.inference_text}, and the inference server is ${store.health.inference?.status || 'down'}.`
    + (cmd ? ` Start it with ${cmd} in a terminal, then try again.` : '');
}

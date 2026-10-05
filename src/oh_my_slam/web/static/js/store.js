// What every page shares: the operations (read from /api/openapi.json, never listed here), the
// service's health, and the job list kept current by the server-sent events of /api/jobs/events.
import { getJson } from './api.js';

const HEALTH_POLL_MS = 4000;

// One operation per command mode, as the OpenAPI document describes it: the registry entry of the
// mode (`x-oms`) and of each parameter, in the order of the command's options.
export class Operation {
  constructor(id, post) {
    this.id = id;
    this.post = post;
    this.x = post['x-oms'];
    const schema = post.requestBody.content['application/json'].schema;
    this.params = Object.values(schema.properties).map((s) => s['x-oms']);
    this.viewerQuery = (post.parameters || []).some((p) => p.in === 'query' && p.name === 'viewer');
  }

  get label() { return this.x.id; }
  get description() { return this.x.description; }
  param(name) { return this.params.find((p) => p.name === name); }
  ofKind(...kinds) { return this.params.filter((p) => kinds.includes(p.kind)); }
  // the mode's output is the browser (view.sh): the job saves a viewer
  get browser() { return this.x.outputs.some((o) => o.via === 'browser'); }
  // the parameter naming a map the mode writes (an output written via that option)
  get writesMap() {
    const flags = new Set(this.x.outputs.map((o) => o.via));
    return this.params.find((p) => p.kind === 'map' && flags.has(p.flag)) || null;
  }
  // its single image input, if it takes one (the Image page's modes)
  get singleImage() { return this.ofKind('image').length === 1 ? this.ofKind('image')[0] : null; }
  get mapParam() { return this.ofKind('map')[0] || null; }
}

export const PATH_IN = ['image', 'images', 'images_or_video', 'map'];

class Store {
  constructor() {
    this.ops = new Map();
    this.health = null;
    this.jobs = new Map();
    this._listeners = new Set();
    this.ready = null;
  }

  on(cb) { this._listeners.add(cb); return () => this._listeners.delete(cb); }
  emit(what, data) { for (const cb of [...this._listeners]) cb(what, data); }

  start() {
    this.ready = (async () => {
      const [doc, jobs] = await Promise.all([getJson('/api/openapi.json'), getJson('/api/jobs')]);
      for (const [path, item] of Object.entries(doc.paths)) {
        const m = /^\/api\/ops\/([^/]+)$/.exec(path);
        if (m && item.post) this.ops.set(m[1], new Operation(m[1], item.post));
      }
      for (const j of jobs) this.jobs.set(j.id, j);
      this.emit('jobs');
      this._events();
      this.pollHealth();
    })();
    return this.ready;
  }

  async pollHealth() {
    clearTimeout(this._healthTimer);
    try {
      this.health = await getJson('/api/health');
    } catch (err) {
      this.health = { status: 'unreachable', message: err.message };
    }
    this.emit('health', this.health);
    this._healthTimer = setTimeout(() => this.pollHealth(), HEALTH_POLL_MS);
  }

  _events() {
    const es = new EventSource('/api/jobs/events');
    // A stream that broke (service restarted, network) starts again from scratch: the events in
    // between are lost, so the whole list is read again once it reopens.
    let broken = false;
    es.addEventListener('error', () => { broken = true; });
    es.addEventListener('open', async () => {
      if (!broken) return;
      broken = false;
      try {
        const jobs = await getJson('/api/jobs');
        for (const j of jobs) { this.jobs.set(j.id, j); this.emit('job', j); }
        this.emit('jobs');
      } catch { broken = true; }
    });
    es.addEventListener('job', (e) => {
      const j = JSON.parse(e.data);
      this.jobs.set(j.id, j);
      this.emit('job', j);
      this.emit('jobs');
    });
    this.events = es;
  }

  counts() {
    const c = { queued: 0, running: 0 };
    for (const j of this.jobs.values()) if (j.state in c) c[j.state]++;
    return c;
  }

  // whether the inference server can take requests (a loading server is fine: commands wait)
  get inferenceUp() {
    const s = this.health?.inference?.status;
    return s === 'ready' || s === 'loading';
  }

  get startCommand() { return this.health?.inference?.start_command || null; }
}

export const store = new Store();

// Why `op` cannot run now (the inference server is down and the mode always needs it), else null;
// a conditional need is decided by the service's own check (/validate) for the actual request.
export function blockedReason(op) {
  if (!store.health || store.inferenceUp || op.x.inference !== 'required') return null;
  const cmd = store.startCommand;
  return `${op.label} ${op.x.inference_text}, and the inference server is down.`
    + (cmd ? ` Start it with ${cmd}, then try again.` : '');
}

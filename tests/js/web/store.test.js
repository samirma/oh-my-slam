// js/store.js: the operations as the OpenAPI document describes them (spec §2.6: forms and pages are
// rendered from the API description, never listed in the app), the service's health, and why an
// operation cannot run now ("Actionable errors": disabled with an explanation while the inference
// server is down, the rest available).
import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { blockedReason, Operation, PATH_IN, store } from '../../../src/oh_my_slam/web/static/js/store.js';
import { fakeFetch, json, openapiDocument, stubGlobals } from '../helpers.js';

const DOC = openapiDocument();
const op = (id) => new Operation(id, DOC.paths[`/api/ops/${id}`].post);
const HEALTH = {
  status: 'ok',
  service: { workspace: 'ws', requests: { running: 1, waiting: 1 }, in_progress: [{ command: ['segment.sh', '-i', 'a.jpg'], state: 'running' }] },
  inference: { status: 'down', start_command: './start_inference_server.sh' },
};

describe('Operation: one command mode, from its registry entry', () => {
  test('its label, description and parameters in the order of the command\'s options', () => {
    const o = op('mapper-update');
    expect([o.id, o.label, o.description]).toEqual(['mapper-update', 'mapper.sh update', 'add images or a video to a map (created if missing)']);
    expect(o.params.map((p) => p.name)).toEqual(['inputs', 'map', 'format', 'fps']);
    expect(o.param('fps').flag).toBe('-fps');
    expect(o.param('nope')).toBeUndefined();
    expect(o.ofKind('map', 'number').map((p) => p.name)).toEqual(['map', 'fps']);
  });

  test('the mode that writes maps is the one whose output is the map its option names', () => {
    expect(op('mapper-update').writesMap.name).toBe('map');
    expect(op('mapper-locate').writesMap).toBeNull();  // takes a map, writes none
    expect(op('reconstruct').writesMap).toBeNull();
  });

  test('the Image page\'s operations take a single image and no map', () => {
    expect(op('reconstruct').singleImage.name).toBe('image');
    expect(op('segment').singleImage.name).toBe('image');
    expect(op('mapper-locate').singleImage).toBeNull();
    expect(op('mapper-update').singleImage).toBeNull();
    expect(op('mapper-locate').mapParam.name).toBe('map');
    expect(op('segment').mapParam).toBeNull();
  });

  test('its results: what the command writes to stdout, not the map it writes', () => {
    expect(op('mapper-update').results.map((r) => r.format)).toEqual(['json', 'ply']);
    expect(op('reconstruct').results.map((r) => r.format)).toEqual(['json', 'png', 'ply']);
  });

  test('path inputs: images, a video or a map of the workspace', () => {
    expect(PATH_IN).toEqual(['image', 'images', 'images_or_video', 'map']);
  });
});

describe('the store', () => {
  let net;
  let timers;
  let restore;
  beforeEach(() => {
    timers = [];
    restore = stubGlobals({ setTimeout: (fn, ms) => timers.push({ fn, ms }), clearTimeout: () => {} });
  });
  afterEach(() => {
    net?.restore();
    restore();
    Object.assign(store, { health: null, active: null });
    store.ops.clear();
  });

  test('start: the operations of the OpenAPI document, then the health', async () => {
    const seen = [];
    const off = store.on((what, data) => seen.push([what, data]));
    net = fakeFetch((url) => json(url === '/api/openapi.json' ? DOC : HEALTH));
    await store.start();
    while (!seen.length) await new Promise((resolve) => setImmediate(resolve));  // the first poll, not awaited
    expect([...store.ops.keys()]).toEqual(['reconstruct', 'mapper-update', 'mapper-locate', 'segment']);
    expect(store.ops.get('segment')).toBeInstanceOf(Operation);
    expect(net.calls.map((c) => c.url)).toEqual(['/api/openapi.json', '/api/health']);
    expect(seen).toEqual([['health', HEALTH]]);
    off();
    await store.pollHealth();
    expect(seen).toHaveLength(1);  // unsubscribed
  });

  test('the health is polled, faster while this page\'s request runs', async () => {
    net = fakeFetch(() => json(HEALTH));
    await store.pollHealth();
    expect(timers.map((t) => t.ms)).toEqual([3000]);
    store.active = {};
    await timers[0].fn();  // the next poll
    expect(timers.map((t) => t.ms)).toEqual([3000, 1000]);
    expect(net.calls).toHaveLength(2);
  });

  test('an unreachable service is a health of its own', async () => {
    const seen = [];
    const off = store.on((what, data) => seen.push(data));
    net = fakeFetch(() => { throw new TypeError('fetch failed'); });
    await store.pollHealth();
    off();
    expect(seen).toEqual([{ status: 'unreachable', message: 'the service did not answer; is server.sh running?' }]);
    expect(store.inferenceUp).toBe(false);
  });

  test('what the health says: the inference server up (a loading one takes requests), how to start it, the requests', () => {
    expect([store.inferenceUp, store.startCommand, store.inProgress]).toEqual([false, null, []]);
    for (const [status, up] of [['ready', true], ['loading', true], ['down', false], ['error', false], ['stopping', false]]) {
      store.health = { ...HEALTH, inference: { ...HEALTH.inference, status } };
      expect(store.inferenceUp).toBe(up);
    }
    expect(store.startCommand).toBe('./start_inference_server.sh');
    expect(store.inProgress).toEqual(HEALTH.service.in_progress);
  });
});

describe('blockedReason: why an operation cannot run now', () => {
  afterEach(() => { store.health = null; });
  test('an operation that always needs the inference server, while it is down: what to do', () => {
    store.health = HEALTH;
    expect(blockedReason(op('segment'))).toBe('segment.sh segments the image with the inference server, and the inference '
      + 'server is down. Start it with ./start_inference_server.sh in a terminal, then try again.');
    store.health = { ...HEALTH, inference: { status: 'error', start_command: null } };
    expect(blockedReason(op('segment'))).toBe('segment.sh segments the image with the inference server, and the inference server is error.');
    store.health = { status: 'ok', service: HEALTH.service };
    expect(blockedReason(op('segment'))).toEndWith('and the inference server is down.');
  });
  test('nothing blocks it otherwise: the rest stays available', () => {
    store.health = HEALTH;
    expect(blockedReason(op('mapper-locate'))).toBeNull();  // needs it only for some maps: the service's check decides
    store.health = { ...HEALTH, inference: { status: 'loading', start_command: null } };
    expect(blockedReason(op('segment'))).toBeNull();
    store.health = { status: 'unreachable', message: 'x' };
    expect(blockedReason(op('segment'))).toBeNull();  // the top bar says the service is unreachable
    store.health = null;
    expect(blockedReason(op('segment'))).toBeNull();
  });
});

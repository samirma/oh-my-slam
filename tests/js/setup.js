// Preloaded before every test file (bunfig.toml).
//
// * The browser modules import three.js by the names of the pages' import map ('three',
//   'three/addons/…'); here those names are the vendored copy the pages load.
// * The unit tests run offline (spec §4): fetch fails for any request a test did not replace it
//   for (helpers.js fakeFetch).
import { mock } from 'bun:test';
import { readdirSync } from 'node:fs';
import { join } from 'node:path';

const VENDOR = join(import.meta.dir, '../../src/oh_my_slam/viewer/static/vendor/three');

const three = await import(join(VENDOR, 'build/three.module.js'));
mock.module('three', () => three);
for (const rel of readdirSync(join(VENDOR, 'addons'), { recursive: true })) {
  if (!rel.endsWith('.js')) continue;
  const addon = await import(join(VENDOR, 'addons', rel));
  mock.module(`three/addons/${rel}`, () => addon);
}

export function offlineFetch(url) {
  return Promise.reject(new Error(`the unit tests run offline: no request to ${url}`));
}
globalThis.fetch = offlineFetch;

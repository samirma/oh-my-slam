// The 3D scene viewer's file reading, off the page's thread: a PLY parsed by the viewer's own
// reader (lib/ply.js), a scene JSON parsed and validated (scene/openlabel.js). A large file never
// blocks the page. Messages: {id, kind: 'ply', buffer} → {id, cloud} (its arrays transferred);
// {id, kind: 'json', buffer} → {id, doc, errors}; a failure → {id, error}.
import { parsePly } from '/static/viewer/lib/ply.js';
import { sceneErrors } from './openlabel.js';

self.onmessage = async ({ data }) => {
  const { id, kind } = data;
  try {
    if (kind === 'ply') {
      const cloud = parsePly(data.buffer);
      self.postMessage({ id, cloud }, Object.values(cloud.arrays).map((a) => a.buffer));
    } else {
      let doc;
      try { doc = JSON.parse(new TextDecoder().decode(data.buffer)); } catch (err) { throw new Error(`it is not JSON: ${err.message}`); }
      self.postMessage({ id, doc, errors: await sceneErrors(doc) });
    }
  } catch (err) {
    self.postMessage({ id, error: err.message });
  }
};

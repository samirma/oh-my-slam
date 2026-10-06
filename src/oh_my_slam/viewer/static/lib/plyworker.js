// parsePly (ply.js) off the page's thread: {buffer, budget} → {cloud} (its arrays transferred), or
// {error} with the reason the bytes are not a PLY this viewer can draw.
import { parsePly } from './ply.js';

self.onmessage = ({ data }) => {
  try {
    const cloud = parsePly(data.buffer, data.budget);
    self.postMessage({ cloud }, Object.values(cloud.arrays).map((a) => a.buffer));
  } catch (err) {
    self.postMessage({ error: err.message });
  }
};

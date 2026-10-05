// Independent layer toggles (spec §2.5): point cloud, segmentation, camera poses, labels, boxes.
import { el } from './dom.js';

// [key, name, what it shows]
export const LAYERS = [
  ['points', 'Point cloud', 'The derived cloud, coloured by the color attribute below'],
  ['segments', 'Segmentation',
    'Only the points of each object, in the object\'s colour, drawn over the point cloud'],
  ['cameras', 'Camera poses', 'A frustum at each camera\'s pose'],
  ['labels', 'Labels', 'Each box\'s id and label, in its colour'],
  ['obbs', 'Oriented boxes', 'The objects\' oriented bounding boxes, in their colours'],
];

// One checkbox row per layer into `container`; `layers` holds each one's state ({key: bool}),
// `onChange(key, on)` is called on a change, `disabled` lists layers with nothing to show.
export function buildLayerControls(container, layers, onChange, disabled = new Set()) {
  for (const [k, name, help] of LAYERS) {
    const cb = el('input', { type: 'checkbox', id: `layer-${k}` });
    cb.checked = layers[k];
    cb.addEventListener('change', () => { layers[k] = cb.checked; onChange(k, cb.checked); });
    const row = el('label', { class: 'row check', 'data-layer': k, for: `layer-${k}`, title: help }, cb,
      el('span', {}, name));
    if (disabled.has(k)) { cb.disabled = true; row.classList.add('disabled'); }
    container.append(row);
  }
}

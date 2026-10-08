// A point-cloud result drawn in the page (http_server.md "Image": a point cloud "is loaded in the
// page and drawn in 3D with the colours and attributes it carries, and the user can rotate, pan and
// zoom it"; "the drawing reuses the §2.5 viewer's own rendering of clouds"). The viewer's own
// modules, which this service serves under /static/viewer/ from the viewer's package, read the
// response bytes the page already holds (no other request, no upload) and draw them; they are loaded
// only when a result needs them. This module embeds that view in the result: its text alternative,
// its buttons, the display-budget notice, and a clear message when the bytes cannot be drawn.
import { el, nextId, notice } from './dom.js';

const fmtN = (n) => Number(n).toLocaleString('en-US');

// The display budget: the viewer's, or a smaller one a test sets (window.__cloudBudget).
export function budgetOf(m) {
  const test = Number(window.__cloudBudget);
  return test > 0 ? test : m.DISPLAY_POINT_BUDGET;
}

// The points of a cloud (cloudview.js cloudFacts): those drawn, of those in the file when fewer.
export function points(f) {
  return f.count < f.total ? `${fmtN(f.count)} drawn of ${fmtN(f.total)}` : fmtN(f.count);
}

export function joinWords(words) {
  return words.length < 2 ? words.join('') : `${words.slice(0, -1).join(', ')} and ${words[words.length - 1]}`;
}

// The facts of the cloud's caption, [term, description] pairs, from its cloudFacts.
export function factRows(f) {
  return [
    ['Points', points(f)],
    ['Each point carries', joinWords(f.carries)],
    f.attrs ? ['Recorded attributes', f.attrs.replaceAll(',', ', ')] : null,
    f.frame ? ['Frame', f.upright ? `${f.frame}; shown upright, as view.sh shows an image (level camera)` : f.frame] : null,
    f.format ? ['PLY format', f.format] : null,
  ].filter(Boolean);
}

// The status once the cloud is drawn.
export function drawnText(f) {
  return f.count < f.total
    ? `Drawn: ${fmtN(f.count)} of its ${fmtN(f.total)} points (see below).`
    : `Drawn: all ${fmtN(f.count)} points.`;
}

// The buttons that move the view, in labelled groups: [group, [[symbol, name, move(view, m)], …]],
// `m` the viewer's module (its step sizes are the arrow keys').
export const TOOLS = [
  ['Rotate', [['◀', 'Rotate left', (v, m) => v.rotate(m.ROTATE_STEP_DEG)], ['▶', 'Rotate right', (v, m) => v.rotate(-m.ROTATE_STEP_DEG)],
    ['▲', 'Rotate up', (v, m) => v.rotate(0, m.ROTATE_STEP_DEG)], ['▼', 'Rotate down', (v, m) => v.rotate(0, -m.ROTATE_STEP_DEG)]]],
  ['Pan', [['←', 'Pan left', (v, m) => v.pan(-m.PAN_STEP)], ['→', 'Pan right', (v, m) => v.pan(m.PAN_STEP)],
    ['↑', 'Pan up', (v, m) => v.pan(0, m.PAN_STEP)], ['↓', 'Pan down', (v, m) => v.pan(0, -m.PAN_STEP)]]],
  ['Zoom', [['+', 'Zoom in', (v, m) => v.zoom(m.ZOOM_STEP)], ['−', 'Zoom out', (v, m) => v.zoom(1 / m.ZOOM_STEP)]]],
];

// The section of a PLY result: `blob` its bytes as received. Returns { el, ready (resolves once it
// is drawn, or its error shown), dispose }.
export function cloudSection(blob) {
  const id = nextId('cloud');
  const status = el('p', { class: 'cloud-status muted', role: 'status' }, 'Reading the point cloud…');
  const host = el('div', {
    class: 'cloud-canvas', role: 'img', tabindex: '0', 'aria-label': 'The point cloud in 3D, being read',
    'aria-describedby': `${id}-keys`, 'data-testid': 'cloud-canvas',
  });
  const facts = el('dl', { class: 'cloud-facts', 'data-testid': 'cloud-facts' });
  const note = el('div', { class: 'cloud-note' });
  const toolbar = el('div', { class: 'cloud-tools', 'data-testid': 'cloud-tools' });
  const keys = el('p', { class: 'help', id: `${id}-keys` },
    'Drag to rotate, right-drag or Shift-drag to pan, scroll to zoom. With the view focused, the arrow keys rotate, '
    + 'Shift with the arrow keys pans, + and − zoom, and 0 brings back the first view.');
  const figure = el('figure', { class: 'cloud-figure' }, host, toolbar, keys, el('figcaption', {}, facts));
  const box = el('section', { class: 'cloud', 'data-testid': 'cloud', 'data-state': 'loading', 'aria-labelledby': `${id}-h` },
    el('h3', { id: `${id}-h` }, 'The point cloud in 3D'), status, note, figure);
  let view = null;
  let disposed = false;

  function failed(err) {
    view?.dispose();
    view = null;
    box.dataset.state = 'error';
    box.dataset.error = err.message;
    figure.hidden = true;
    status.replaceWith(notice('error', el('strong', {}, 'This point cloud cannot be drawn: '), `${err.message}. `,
      'The download above still holds the result exactly as received.'));
  }

  const ready = (async () => {
    try {
      const m = await import('/static/viewer/lib/cloudview.js');
      const cloud = await m.readCloud(await blob.arrayBuffer(), budgetOf(m));
      if (disposed) return;
      view = new m.CloudView(host);
      view.show(cloud);
      box.cloudView = view;  // for tests and debugging
      const f = m.cloudFacts(cloud);
      facts.replaceChildren(...factRows(f).map(([k, v]) => el('div', {}, el('dt', {}, k), el('dd', { 'data-fact': k }, v))));
      host.setAttribute('aria-label', `The point cloud in 3D: ${points(f)} points, each with ${joinWords(f.carries)}`);
      if (f.note) note.replaceChildren(notice('info', f.note));
      for (const [name, buttons] of TOOLS) {
        const gid = `${id}-${name.toLowerCase()}`;
        toolbar.append(el('div', { class: 'tool-group', role: 'group', 'aria-labelledby': gid },
          el('span', { id: gid, class: 'tool-name' }, name),
          ...buttons.map(([symbol, label, move]) => {
            const b = el('button', { type: 'button', class: 'icon', 'aria-label': label, title: label, 'data-move': label },
              el('span', { 'aria-hidden': 'true' }, symbol));
            b.addEventListener('click', () => move(view, m));
            return b;
          })));
      }
      const reset = el('button', { type: 'button', 'data-move': 'Reset view' }, 'Reset view');
      reset.addEventListener('click', () => view.reset());
      toolbar.append(reset);
      status.textContent = 'Drawing…';
      await new Promise((resolve) => {
        const wait = () => (disposed || !view || view.drawn ? resolve() : requestAnimationFrame(wait));
        wait();
      });
      if (disposed || !view) return;
      box.dataset.state = 'drawn';
      status.textContent = drawnText(f);
    } catch (err) {
      if (!disposed) failed(err);
    }
  })();

  return { el: box, ready, dispose() { disposed = true; view?.dispose(); view = null; } };
}

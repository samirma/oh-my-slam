// The §2.5 viewer embedded in a page: view.sh's own page, served by the viewer's own routes at its
// stable URL (/viewer/map/<name>/, /viewer/job/<id>/), so its layers, controls and camera features
// are those of view.sh. The same URL opens it full screen. The page joins the host page's
// selection through the viewer's select / onSelect API, and a click on a box selects it.
import { el } from './dom.js';
import { clickToSelect } from './pick.js';

export function embeddedViewer(url, title, selection) {
  const frame = el('iframe', { class: 'viewer-frame', src: url, title, 'data-testid': 'viewer' });
  const full = el('a', { href: url, class: 'button secondary', target: '_blank', rel: 'noopener' }, 'Open the viewer full screen');
  const box = el('section', { class: 'viewer-box', 'aria-label': title },
    el('div', { class: 'viewer-head' }, el('h3', {}, 'Viewer'), full), frame);
  frame.addEventListener('load', () => {
    const win = frame.contentWindow;
    const wait = () => {
      let st, viewer;
      try { st = win.__viewer; viewer = win.__viewerApp; } catch { return; }  // navigated away
      if (win.document.body?.dataset.error) { box.dataset.error = win.document.body.dataset.error; return; }
      if (!st || !st.ready || !viewer) { setTimeout(wait, 100); return; }
      if (selection) {
        viewer.onSelect((id) => selection.set(id));
        selection.join((id) => viewer.select(id));
        clickToSelect(viewer, (id) => selection.set(id));
      }
      box.dataset.ready = 'true';
    };
    wait();
  });
  return box;
}

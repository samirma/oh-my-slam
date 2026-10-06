// The server.sh web application (http_server.md "Web application"): a shell (top bar, pages) over
// the public API only. Every page has a stable URL (hash routing), so a map's page can be
// bookmarked, reloaded or shared on this machine:
//   #/image[?op=<op>]  #/maps[?filter=<text>]  #/maps/new  #/maps/<name>[?op=<op>]
//   #/maps/<name>/update
// A request in progress belongs to its page: leaving the page asks first (see request.js).
import { el, clear, notice } from './dom.js';
import { store } from './store.js';
import { confirmAction } from './dialog.js';
import { interruptConsequence } from './request.js';
import { imagePage } from './pages/image.js';
import { mapsPage, mapPage } from './pages/maps.js';
import { mapFlowPage } from './pages/mapflow.js';

const main = document.getElementById('main');
let cleanup = null;
let current = null;  // the route shown
let navigated = false;  // the first page of a visit keeps the browser's own focus

function parse(hash) {
  const raw = hash.replace(/^#\/?/, '');
  const [path, query = ''] = raw.split('?');
  const parts = path.split('/').filter(Boolean).map((p) => { try { return decodeURIComponent(p); } catch { return p; } });
  return { parts, path: parts.join('/'), query: new URLSearchParams(query) };
}

// `event`: the hashchange that navigated (its timeStamp starts the in-page timing of the route)
function route(event) {
  const start = event && event.timeStamp ? event.timeStamp : performance.now();
  const { parts, path, query } = parse(location.hash);
  const [page = 'image', a, b] = parts;
  if (cleanup) { try { cleanup(); } catch (err) { console.error(err); } cleanup = null; }
  clear(main);
  for (const link of document.querySelectorAll('#nav a')) {
    if (link.dataset.page === page) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  }
  let render;
  if (page === 'image' && !a) render = () => imagePage(main, { op: query.get('op') });
  else if (page === 'maps' && !a) render = () => mapsPage(main, { filter: query.get('filter') || '' });
  else if (page === 'maps' && a === 'new' && !b) render = () => mapFlowPage(main, { name: null });
  else if (page === 'maps' && b === 'update' && parts.length === 3) render = () => mapFlowPage(main, { name: a });
  else if (page === 'maps' && !b) render = () => mapPage(main, { name: a, op: query.get('op') });
  else {
    render = () => {
      main.append(el('h1', {}, 'Page not found'),
        notice('error', `There is no page ${location.hash}. `, el('a', { href: '#/image' }, 'Go to the Image page'), ' or ', el('a', { href: '#/maps' }, 'the maps'), '.'));
    };
  }
  try {
    cleanup = render() || null;
  } catch (err) {
    console.error(err);
    main.append(notice('error', `This page failed: ${err.message}. Reload the page; if it fails again, check server.sh's log.`));
  }
  document.body.dataset.page = page;
  // a new page: its title, focus on its heading, and an announcement for screen readers
  const h1 = main.querySelector('h1');
  const title = h1 ? h1.textContent.trim() : 'oh-my-slam';
  document.title = `${title} — oh-my-slam`;
  if (h1 && navigated && path !== current) {
    h1.tabIndex = -1;
    h1.focus({ preventScroll: false });
    document.getElementById('announce').textContent = `${title} page`;
  }
  current = path;
  navigated = true;
  // how long this navigation took in the page, from the hash change to the rendered page
  window.__app.lastRoute = { where: path, start, end: performance.now() };
}

// ------------------------------------------------------------------------- leaving a running page
// Every hashchange is a navigation (a page keeps its own state in the URL with replaceState, which
// fires none). While a request runs in this page, the navigation waits for the user's agreement.
let leaving = null;
async function onHashChange(event) {
  const target = location.hash;
  if (leaving === target || !store.active?.inFlight) { leaving = null; route(event); return; }
  history.replaceState(null, '', new URL(event.oldURL).hash || '#/');
  const req = store.active;
  const yes = await confirmAction({
    title: 'Leave this page and interrupt its request?',
    body: [`A request runs in this page. ${interruptConsequence(req.op)}`],
    yes: 'Interrupt it and leave', no: 'Stay on this page', danger: true,
  });
  if (!yes) return;
  await req.interrupt(false);
  leaving = target;
  location.hash = target;
}
window.addEventListener('hashchange', (e) => { onHashChange(e); });

// ------------------------------------------------------------------------------------ the top bar
function drawHealth() {
  const h = store.health;
  const ws = document.getElementById('workspace');
  const inf = document.getElementById('inference');
  const reqs = document.getElementById('requests');
  if (!h) return;
  if (h.status === 'unreachable') {
    ws.textContent = '?';
    inf.replaceChildren(el('span', { class: 'pill down' }, 'unknown'));
    reqs.replaceChildren(el('span', { class: 'pill down' }, 'service unreachable'), ` ${h.message}`);
    document.body.dataset.service = 'down';
    return;
  }
  document.body.dataset.service = 'up';
  ws.textContent = h.service.workspace;
  ws.title = h.service.data;
  const s = h.inference.status;
  const up = s === 'ready' || s === 'loading';
  inf.replaceChildren(el('span', { class: `pill ${s === 'ready' ? 'up' : up ? 'loading' : 'down'}`, 'data-testid': 'inference-status' },
    s === 'ready' ? '● ready' : s === 'loading' ? '◐ loading' : `○ ${s}`));
  if (!up && h.inference.start_command) {
    inf.append(el('span', { class: 'start' }, ' start it with ', el('code', { 'data-testid': 'start-command' }, h.inference.start_command)));
  }
  document.body.dataset.inference = up ? 'up' : 'down';
  const c = h.service.requests || { running: 0, waiting: 0 };
  const parts = [];
  if (c.running) parts.push(el('span', { class: 'pill busy' }, `${c.running} running`));
  if (c.waiting) parts.push(el('span', { class: 'pill loading' }, `${c.waiting} waiting for its turn`));
  reqs.replaceChildren(...(parts.length ? parts : [el('span', { class: 'muted' }, 'none')]));
  reqs.dataset.running = c.running;
  reqs.dataset.waiting = c.waiting;
}
store.on((what) => { if (what === 'health') drawHealth(); });

// a page that goes away discards the uploads no request consumed
window.addEventListener('pagehide', () => { if (cleanup) { try { cleanup(); } catch { /* gone */ } cleanup = null; } });

// the skip link moves the focus to the page (it is no route)
document.querySelector('[data-skip]').addEventListener('click', (e) => { e.preventDefault(); main.focus(); });

window.__app = { store };
store.start().then(() => {
  route();
  document.body.dataset.ready = 'true';
}).catch((err) => {
  main.append(el('h1', {}, 'oh-my-slam'), notice('error', `The service did not answer (${err.message}). Check that server.sh is running, then reload this page.`));
  document.body.dataset.ready = 'true';
});

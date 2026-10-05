// The server.sh web application (http_server.md "Web application"): a shell (top bar, pages) over
// the public API only. Every page has a stable URL (hash routing), so a map, a job or a view can be
// bookmarked, reloaded or shared on this machine and returns in the same state:
//   #/image[?op=<op>][/<job>]  #/maps  #/maps/new  #/maps/<name>  #/maps/<name>/update
//   #/jobs  #/jobs/<id>  #/scene[?ply=<url>&json=<url>]
// The viewer is full screen on its own URL (/viewer/map/<name>/, /viewer/job/<id>/).
import { el, clear, notice } from './dom.js';
import { store } from './store.js';
import { imagePage } from './pages/image.js';
import { mapsPage, mapPage } from './pages/maps.js';
import { mapFlowPage } from './pages/mapflow.js';
import { jobsPage, jobPage } from './pages/jobs.js';
import { scenePage } from './pages/scene.js';

const main = document.getElementById('main');
let cleanup = null;

function parse() {
  const raw = location.hash.replace(/^#\/?/, '');
  const [path, query = ''] = raw.split('?');
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  return { parts, query: new URLSearchParams(query) };
}

function route() {
  const { parts, query } = parse();
  const [page = 'image', a, b] = parts;
  if (cleanup) { try { cleanup(); } catch (err) { console.error(err); } cleanup = null; }
  clear(main);
  for (const link of document.querySelectorAll('#nav a')) {
    if (link.dataset.page === page) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  }
  let render;
  if (page === 'image') render = () => imagePage(main, { job: a || null, op: query.get('op') });
  else if (page === 'maps' && !a) render = () => mapsPage(main, { filter: query.get('filter') || '' });
  else if (page === 'maps' && a === 'new') render = () => mapFlowPage(main, { name: null });
  else if (page === 'maps' && b === 'update') render = () => mapFlowPage(main, { name: a });
  else if (page === 'maps') render = () => mapPage(main, { name: a });
  else if (page === 'jobs' && !a) render = () => jobsPage(main);
  else if (page === 'jobs') render = () => jobPage(main, { id: a });
  else if (page === 'scene') render = () => scenePage(main, { query });
  else render = () => { main.append(el('h1', {}, 'Not found'), notice('error', `There is no page ${location.hash}.`)); };
  try {
    cleanup = render() || null;
  } catch (err) {
    console.error(err);
    main.append(notice('error', `This page failed: ${err.message}`));
  }
  document.body.dataset.page = page;
  // a new page: its title, focus on its heading, and an announcement for screen readers
  const h1 = main.querySelector('h1');
  const title = h1 ? h1.textContent.trim() : 'oh-my-slam';
  document.title = `${title} — oh-my-slam`;
  const where = parts.join('/');
  if (h1 && navigated && where !== lastPage) {
    h1.tabIndex = -1;
    h1.focus({ preventScroll: false });
    document.getElementById('announce').textContent = `${title} page`;
  }
  lastPage = where;
  navigated = true;
}
let navigated = false;  // the first page of a visit keeps the browser's own focus
let lastPage = null;  // a change within a page (its query) keeps the focus where it is

// ------------------------------------------------------------------------------------- the top bar
function drawHealth() {
  const h = store.health;
  const ws = document.getElementById('workspace');
  const inf = document.getElementById('inference');
  if (!h) return;
  if (h.status === 'unreachable') {
    ws.textContent = '?';
    inf.replaceChildren(el('span', { class: 'pill down' }, 'unknown'), ` the service did not answer: ${h.message}`);
    return;
  }
  ws.textContent = h.service.workspace;
  ws.title = h.service.data;
  const s = h.inference.status;
  const up = s === 'ready' || s === 'loading';
  inf.replaceChildren(el('span', { class: `pill ${up ? (s === 'ready' ? 'up' : 'loading') : 'down'}` }, s === 'ready' ? 'ready' : s === 'loading' ? 'loading' : 'down'));
  if (!up && h.inference.start_command) {
    inf.append(' start it with ', el('code', { 'data-testid': 'start-command' }, h.inference.start_command));
  }
  document.body.dataset.inference = up ? 'up' : 'down';
}

function drawCounts() {
  const c = store.counts();
  document.getElementById('job-counts').textContent = `${c.queued} queued · ${c.running} running`;
}

store.on((what) => {
  if (what === 'health') drawHealth();
  if (what === 'jobs') drawCounts();
});

window.addEventListener('hashchange', route);
window.__app = { store };
store.start().then(() => {
  route();
  document.body.dataset.ready = 'true';
}).catch((err) => {
  main.append(notice('error', `The service did not answer (${err.message}). Is server.sh running?`));
});

// A request running in its page (http_server.md "Requests", "Responsive feedback",
// "Interruption"): there are no jobs, so the operation runs within this page's HTTP request and its
// answer is the result. While it is in progress the page shows whether it is running or waiting for
// its turn (the service's health lists the requests in progress, each with the command line its
// validation gave) and for how long. It belongs to the page: interrupting it — the Interrupt
// button, leaving or reloading the page — closes the connection, which interrupts the command as
// Ctrl-C would, so each of these asks first and states the consequence.
import { el, clear, fmtClock, fmtSeconds, notice } from './dom.js';
import { runOperation, ApiError } from './api.js';
import { store } from './store.js';
import { confirmAction } from './dialog.js';
import { resultView } from './result.js';

const TICK_MS = 250;
const STATE_TEXT = {
  sending: 'Sending', waiting: 'Waiting for its turn', running: 'Running', done: 'Done',
  failed: 'Failed', interrupted: 'Interrupted',
};

export function sameCommand(a, b) { return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => x === b[i]); }

// This page's request among the service's requests in progress (its health's `in_progress`): the
// entry with its command line, and of identical requests the one that arrived nearest to
// `sentAt`; null when none is listed.
export function ownEntry(inProgress, command, sentAt) {
  const mine = inProgress.filter((e) => sameCommand(e.command, command));
  if (!mine.length) return null;
  return mine.reduce((a, b) => (Math.abs(b.arrived_at - sentAt) < Math.abs(a.arrived_at - sentAt) ? b : a));
}

// What an ended request says of its wait for its turn: `waited` seconds, exactly or at least
// (`exact`); '' below a second.
export function waitedNote(waited, exact) {
  return waited >= 1 ? `including ${exact ? '' : 'at least '}${fmtSeconds(waited)} waiting for its turn` : '';
}

// What interrupting a request of `op` leaves behind, in the commands' terms.
export function interruptConsequence(op) {
  return op.writesMap
    ? `${op.label} stops as Ctrl-C would stop it: the map is left exactly as it was before this update, and there is no result.`
    : `${op.label} stops as Ctrl-C would stop it, and there is no result.`;
}

function onBeforeUnload(e) {
  e.preventDefault();
  e.returnValue = '';  // the browser asks before leaving or reloading
  return '';
}

export class RequestView {
  // `container`: where the request and then its result are shown; `op`; `form`: its OpForm (the
  // messages of a refused request go next to its fields); `downloadName(format)`: the result's file
  // name.
  constructor(container, { op, form, downloadName }) {
    this.container = container;
    this.op = op;
    this.form = form;
    this.downloadName = downloadName;
    this.state = null;
    this.controller = null;
    this.result = null;
    this.waited = 0;
    this.waitedExact = false;
    this.seenWaiting = 0;  // the last time (s into the request) it was seen waiting
    this._off = store.on((what) => { if (what === 'health') this.track(); });
  }

  get inFlight() { return this.controller !== null; }

  // Run with these values; `command`: the command line its validation answered. Resolves to true
  // when it succeeded.
  async run(values, command) {
    this.dispose(false);
    this.command = command || [];
    this.t0 = performance.now();
    this.sentAt = Date.now() / 1000;
    this.controller = new AbortController();
    this.interruptedBy = null;
    this.state = null;
    this.waited = 0;
    this.waitedExact = false;
    this.seenWaiting = 0;
    this.stateEl = el('span', { class: 'state', role: 'status' });
    this.clock = el('span', { class: 'elapsed', 'data-testid': 'elapsed' }, '0:00');
    this.waitNote = el('span', { class: 'waited muted' });
    this.cancelBtn = el('button', { type: 'button', class: 'danger', 'data-action': 'interrupt' }, 'Interrupt…');
    this.cancelBtn.addEventListener('click', () => this.interrupt());
    this.outcome = el('div', { class: 'outcome' });
    this.box = el('section', { class: 'request', 'data-testid': 'request', 'aria-label': `Request: ${this.op.label}` },
      el('div', { class: 'request-head' },
        el('p', { class: 'request-status' }, this.stateEl, el('span', { 'aria-hidden': 'true' }, ' · '), this.clock, ' ', this.waitNote),
        el('div', { class: 'actions' }, this.cancelBtn)),
      el('p', { class: 'command-row' }, el('span', { class: 'sub' }, 'Command: '), el('code', { class: 'command' }, this.command.join(' '))),
      this.belongs = el('p', { class: 'muted belongs' }, 'This request belongs to this page: leaving or reloading the page interrupts it.'),
      this.outcome);
    clear(this.container).append(this.box);
    this.box.scrollIntoView({ block: 'nearest' });
    this.setState('sending');
    store.active = this;
    window.addEventListener('beforeunload', onBeforeUnload);
    this.timer = setInterval(() => this.tick(), TICK_MS);
    store.pollHealth();  // its state at once, then every second while it runs
    let ok = false;
    const answer = runOperation(this.op.id, values, this.controller.signal);
    this.form?.consumed();  // the request consumes its uploads, whatever its outcome
    try {
      const res = await answer;
      this.end('done');
      await this.showResult(res);
      ok = true;
    } catch (err) {
      if (err.name === 'AbortError' || this.interruptedBy) {
        this.end('interrupted');
        this.outcome.replaceChildren(notice('warn', el('strong', {}, 'Interrupted. '), interruptConsequence(this.op),
          ' Run it again when you are ready.'));
      } else {
        this.end('failed');
        this.showError(err);
      }
    }
    return ok;
  }

  // This page's request is waiting or running: which one, from the service's list of the requests
  // in progress (the entry with its command line; for identical requests, the one that arrived
  // nearest to when this one was sent).
  track() {
    if (!this.inFlight) return;
    const e = ownEntry(store.inProgress, this.command, this.sentAt);
    if (!e) return;
    if (e.state === 'running' && e.started_at) { this.waited = Math.max(0, e.started_at - e.arrived_at); this.waitedExact = true; }
    if (e.state !== 'running') this.seenWaiting = (performance.now() - this.t0) / 1000;
    this.setState(e.state === 'running' ? 'running' : 'waiting');
  }

  setState(s) {
    if (s === this.state) return;
    this.state = s;
    this.box.dataset.state = s;
    this.stateEl.textContent = STATE_TEXT[s] || s;
    this.stateEl.className = `state state-${s}`;
  }

  tick() {
    const elapsed = (performance.now() - this.t0) / 1000;
    if (this.state === 'running') {
      this.clock.textContent = fmtClock(elapsed - this.waited);
      this.waitNote.textContent = this.waited >= 1 ? `after waiting ${fmtClock(this.waited)}` : '';
    } else {
      this.clock.textContent = fmtClock(elapsed);
    }
  }

  end(state) {
    clearInterval(this.timer);
    this.tick();
    const total = (performance.now() - this.t0) / 1000;
    this.setState(state);
    this.clock.textContent = `${state === 'done' ? 'in' : 'after'} ${fmtSeconds(total)}`;
    // it started between the last time it was seen waiting and its end: at least that long waiting
    this.waitNote.textContent = waitedNote(this.waitedExact ? this.waited : this.seenWaiting, this.waitedExact);
    this.controller = null;
    this.cancelBtn.remove();
    this.belongs.remove();
    if (store.active === this) store.active = null;
    window.removeEventListener('beforeunload', onBeforeUnload);
    store.pollHealth();
  }

  async showResult(res) {
    const output = this.form?.result();
    const format = output?.format || (res.mediaType === 'application/json' ? 'json' : '');
    this.result = await resultView({ ...res, format, downloadName: this.downloadName(format) });
    this.outcome.replaceChildren(this.result.el);
  }

  showError(err) {
    const parts = [el('strong', {}, `${this.op.label} failed: `), err.message || 'unknown error'];
    if (err.code) parts.push(el('span', { class: 'muted' }, ` (${err.code}${err.status ? `, HTTP ${err.status}` : ''})`));
    if (err instanceof ApiError && err.status === 503 && store.startCommand && !(err.message || '').includes(store.startCommand)) {
      parts.push(el('p', {}, 'Start the inference server with ', el('code', {}, store.startCommand), ', then run it again.'));
    }
    if (err instanceof ApiError && Object.keys(err.byParameter).length && this.form) {
      this.form.show(err.byParameter, err.problems);
      parts.push(el('p', {}, 'Fix the fields marked above, then run it again.'));
    }
    this.outcome.replaceChildren(notice('error', ...parts));
  }

  // Interrupt it, after saying what that does (unless `ask` is false: the user already confirmed,
  // e.g. leaving the page). Resolves true when it was interrupted.
  async interrupt(ask = true) {
    if (!this.inFlight) return true;
    if (ask) {
      const yes = await confirmAction({
        title: 'Interrupt this request?',
        body: [interruptConsequence(this.op)],
        yes: 'Interrupt it', no: 'Keep it running', danger: true,
      });
      if (!yes || !this.inFlight) return !this.inFlight;
    }
    this.interruptedBy = 'user';
    this.controller.abort();
    return true;
  }

  dispose(all = true) {
    if (this.inFlight) { this.interruptedBy = 'left'; this.controller.abort(); }
    this.result?.dispose();
    this.result = null;
    if (all) this._off();
  }
}

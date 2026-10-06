// Creating or updating a map (http_server.md "Maps"): a guided flow over the mapping operation's
// options — the operation whose output is the map folder it names, read from the API description.
// Step 1 holds its input parameters, their order visible and editable where it matters (`ordered`);
// step 2 the map (#/maps/new: a name; #/maps/<name>/update: that map); step 3 every other option;
// step 4 states what starting it does before it runs, and asks for confirmation (a long
// operation). The request then runs in this page, and its result is shown here.
import { el, notice, humanize } from '../dom.js';
import { enc } from '../api.js';
import { store, PATH_IN } from '../store.js';
import { OpForm } from '../form.js';
import { runPanel } from '../runpanel.js';
import { confirmAction } from '../dialog.js';

export function mapFlowPage(main, { name }) {
  const op = [...store.ops.values()].find((o) => o.writesMap);
  main.append(el('h1', {}, name ? `Update map ${name}` : 'New map'));
  if (!op) { main.append(notice('info', 'No operation of this service writes a map.')); return null; }
  main.append(el('p', { class: 'lead' }, name
    ? `Extend map ${name} with new images or a video. The latest observation wins: what the new inputs show replaces what the map had.`
    : 'Create a map from images or a video, in the order they were taken.'));
  const mapP = op.writesMap;
  const inputs = op.params.filter((p) => PATH_IN.includes(p.kind) && p !== mapP).map((p) => p.name);
  const step = (n, title, ...body) => el('section', { class: 'step', 'aria-labelledby': `step-${n}`, 'data-step': n },
    el('h2', { id: `step-${n}` }, el('span', { class: 'step-n', 'aria-hidden': 'true' }, `${n}`), ` ${title}`), ...body);
  const s1 = el('div', {}), s2 = el('div', {}), s3 = el('div', {}), s4 = el('div', {});
  const done = el('div', { class: 'flow-done' });
  main.append(
    step(1, inputs.map(humanize).join(', ') || 'Inputs', s1),
    step(2, 'Map', s2),
    step(3, 'Options', s3),
    step(4, 'Review and start', s4), done);

  // one form, its fields placed in the steps
  const host = el('div', {});
  const form = new OpForm(host, op, name ? { fixed: { [mapP.name]: name } } : {});
  for (const f of form.fields) (inputs.includes(f.p.name) ? s1 : f.p === mapP ? s2 : s3).append(f.el);
  if (name) s2.append(el('p', {}, 'The new inputs extend map ', el('a', { href: `#/maps/${enc(name)}` }, name), '.'));
  if (!s3.children.length) s3.append(el('p', { class: 'muted' }, 'No other option.'));
  s4.append(form.commandRow, form.general);

  const label = name ? `Update ${name}…` : 'Create the map…';
  const panel = runPanel({
    op, form, label,
    downloadName: (fmt) => `${op.id}-${form.values()[mapP.name] || 'map'}${fmt ? `.${fmt}` : ''}`,
    confirm: (v) => {
      const target = form.values()[mapP.name];
      return confirmAction({
        title: name ? `Update map ${target}?` : `Create map ${target}?`,
        body: [`This runs ${v.command.join(' ')}.`,
          `${op.label} ${op.x.inference_text}; it can take many minutes, and requests that use the inference server run one at a time, so it may first wait for its turn.`,
          'It runs within this page: keep the page open until it ends. Interrupting it, or leaving or reloading the page, stops it and leaves the map exactly as it was. The map changes only when it succeeds.'],
        yes: name ? 'Start the update' : 'Start mapping', no: 'Not now',
      });
    },
    onEnd: (ok) => {
      if (!ok) return null;
      const target = form.values()[mapP.name];
      done.replaceChildren(notice('ok', `Map ${target} is ${name ? 'updated' : 'created'}. `,
        el('a', { href: `#/maps/${enc(target)}`, 'data-action': 'open-map' }, `Open map ${target}`), ' to see its summary and history.'));
      form.quiet();
      for (const n of inputs) form.field(n)?.clearAll();  // these inputs are in the map now
      return 'keep-cleared';
    },
  });
  s4.append(panel.el);
  form.changed();
  return () => { panel.dispose(); form.destroy(); };
}

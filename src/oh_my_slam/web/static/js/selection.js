// Selecting an object anywhere on a page highlights it everywhere else (http_server.md "Structure"):
// a table row, a region of the segmented image, a box in the viewer. Each view joins the page's
// Selection with a function that shows an id (null: none) and calls set(id) when the user picks one.
export class Selection {
  constructor() { this.id = null; this.views = new Set(); }

  join(show) { this.views.add(show); show(this.id); return () => this.views.delete(show); }

  set(id) {
    const next = id == null || id === '' ? null : Number(id);
    if (next === this.id) return;
    this.id = next;
    for (const show of [...this.views]) show(next);
  }

  toggle(id) { this.set(Number(id) === this.id ? null : id); }
}

// Rows of a table (tr[data-id]) as a selection view: click, Enter or Space selects; the selected
// row is marked (aria-selected) and scrolled into view.
export function linkRows(tbody, selection) {
  tbody.addEventListener('click', (e) => {
    const tr = e.target.closest('tr[data-id]');
    if (tr && !e.target.closest('a, button')) selection.toggle(tr.dataset.id);
  });
  tbody.addEventListener('keydown', (e) => {
    const tr = e.target.closest('tr[data-id]');
    if (tr && (e.key === 'Enter' || e.key === ' ') && e.target === tr) { e.preventDefault(); selection.toggle(tr.dataset.id); }
  });
  return selection.join((id) => {
    for (const tr of tbody.querySelectorAll('tr[data-id]')) {
      const on = id != null && Number(tr.dataset.id) === id;
      tr.classList.toggle('selected', on);
      tr.setAttribute('aria-selected', String(on));
      if (on && !tr.matches(':focus-within')) tr.scrollIntoView({ block: 'nearest' });
    }
  });
}

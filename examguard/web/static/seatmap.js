(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };

  async function api(path, opts = {}) {
    const res = await fetch(path, { credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, ...opts });
    if (res.status === 401) { location.href = '/login'; throw new Error('Signed out'); }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }

  let seats = [];          // every seat on every camera: {label, x, y, camera, student_ref}
  let roster = [];
  let cam = 0;
  let selected = -1;
  let dirty = false;
  const stage = $('#stage');
  const pic = $('#pic');

  function say(text, bad) { const m = $('#msg'); m.textContent = text || ''; m.style.color = bad ? 'var(--high)' : ''; }
  function setDirty(v) { dirty = v; if (v) say('Unsaved changes'); }
  window.addEventListener('beforeunload', (e) => { if (dirty) { e.preventDefault(); e.returnValue = ''; } });

  function nextLabel() {
    const used = new Set(seats.filter((s) => s.camera === cam).map((s) => s.label.toLowerCase()));
    for (let n = seats.filter((s) => s.camera === cam).length + 1; n < 1000; n++) {
      if (!used.has(String(n))) return String(n);
    }
    return '?';
  }

  function render() {
    stage.querySelectorAll('.seat-mark').forEach((n) => n.remove());
    seats.forEach((s, i) => {
      if (s.camera !== cam) return;
      const m = el('div', 'seat-mark' + (i === selected ? ' selected' : ''), s.label);
      m.style.left = (s.x * 100) + '%';
      m.style.top = (s.y * 100) + '%';
      m.dataset.i = i;
      m.title = s.label + (nameOf(s.student_ref) ? ' · ' + nameOf(s.student_ref) : '');
      m.addEventListener('pointerdown', (ev) => startDrag(ev, i));
      stage.appendChild(m);
    });
    const list = $('#seat-list');
    const mine = seats.map((s, i) => [s, i]).filter(([s]) => s.camera === cam);
    list.replaceChildren(...mine.map(([s, i]) => {
      const row = el('div', 'seat-row' + (i === selected ? ' selected' : ''));
      const inp = el('input'); inp.value = s.label; inp.maxLength = 20; inp.setAttribute('aria-label', 'Seat label');
      inp.addEventListener('focus', () => { selected = i; markSelected(); });
      inp.addEventListener('input', () => { s.label = inp.value.trim(); setDirty(true); const m = stage.querySelector('[data-i="' + i + '"]'); if (m) m.textContent = s.label; });
      row.appendChild(inp);
      if (roster.length) {
        const sel = el('select'); sel.setAttribute('aria-label', 'Student for seat ' + s.label);
        sel.appendChild(new Option('No student assigned', ''));
        roster.forEach((r) => sel.appendChild(new Option(r.name + (r.student_id ? ' (' + r.student_id + ')' : ''), r.ref)));
        sel.value = s.student_ref || '';
        sel.addEventListener('change', () => { s.student_ref = sel.value; setDirty(true); });
        row.appendChild(sel);
      }
      const del = el('button', 'btn small ghost', 'Remove'); del.type = 'button';
      del.addEventListener('click', () => { seats.splice(i, 1); selected = -1; setDirty(true); render(); });
      row.appendChild(del);
      return row;
    }));
    $('#seat-hint').textContent = mine.length
      ? mine.length + (mine.length === 1 ? ' seat' : ' seats') + ' on this camera.'
      : 'No seats yet. Click on the picture, or press "Fill from faces".';
  }

  function markSelected() {
    stage.querySelectorAll('.seat-mark').forEach((n) => n.classList.toggle('selected', Number(n.dataset.i) === selected));
  }
  const nameOf = (ref) => { const r = roster.find((x) => x.ref === ref); return r ? r.name : ''; };

  // click on the picture = new seat; drag a mark = move it
  let drag = null;
  function frac(ev) {
    const r = stage.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height))];
  }
  function startDrag(ev, i) {
    ev.preventDefault(); ev.stopPropagation();
    selected = i; drag = { i, moved: false };
    stage.setPointerCapture && stage.setPointerCapture(ev.pointerId);
    markSelected();
  }
  stage.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    const [x, y] = frac(ev);
    const s = seats[drag.i]; s.x = Math.round(x * 10000) / 10000; s.y = Math.round(y * 10000) / 10000;
    drag.moved = true;
    const m = stage.querySelector('[data-i="' + drag.i + '"]');
    if (m) { m.style.left = (s.x * 100) + '%'; m.style.top = (s.y * 100) + '%'; }
  });
  stage.addEventListener('pointerup', () => { if (drag && drag.moved) setDirty(true); const was = drag; drag = null; if (was) render(); });
  stage.addEventListener('click', (ev) => {
    if (ev.target !== pic) return;
    const [x, y] = frac(ev);
    seats.push({ label: nextLabel(), x: Math.round(x * 10000) / 10000, y: Math.round(y * 10000) / 10000, camera: cam, student_ref: '' });
    selected = seats.length - 1; setDirty(true); render();
  });

  function loadPicture() {
    pic.onerror = () => say('No picture from this camera yet. Check the dashboard shows live video, then refresh.', true);
    pic.onload = () => say('');
    pic.src = '/api/seatmap/snapshot?cam=' + cam + '&t=' + Date.now();
  }

  $('#refresh').addEventListener('click', loadPicture);
  $('#fill').addEventListener('click', async () => {
    try {
      const d = await api('/api/seatmap/suggest?cam=' + cam);
      if (!d.seats.length) { say('No faces are visible right now. Seat people in view and try again.', true); return; }
      const have = seats.some((s) => s.camera === cam);
      if (have && !window.confirm('Replace the ' + seats.filter((s) => s.camera === cam).length + ' seats on this camera with ' + d.seats.length + ' seats placed on the visible faces?')) return;
      const keep = seats.filter((s) => s.camera !== cam);
      seats = keep.concat(d.seats.map((s) => ({ ...s, camera: cam, student_ref: '' })));
      selected = -1; setDirty(true); render(); loadPicture();
      say(d.seats.length + ' seats placed. Rows are guessed: check the labels, fix any that are wrong, then save.');
    } catch (e) { say(e.message, true); }
  });
  $('#clear').addEventListener('click', () => {
    if (!seats.some((s) => s.camera === cam)) return;
    if (!window.confirm('Remove every seat on this camera?')) return;
    seats = seats.filter((s) => s.camera !== cam); selected = -1; setDirty(true); render();
  });
  $('#save').addEventListener('click', async () => {
    try {
      const d = await api('/api/seatmap', { method: 'PUT', body: JSON.stringify({ seats }) });
      seats = d.seats; dirty = false; render(); say('Saved. It applies immediately.');
    } catch (e) { say(e.message, true); }
  });
  document.querySelectorAll('.cam-tab').forEach((b) => b.addEventListener('click', () => {
    cam = Number(b.dataset.cam); selected = -1;
    document.querySelectorAll('.cam-tab').forEach((x) => x.classList.toggle('active', x === b));
    render(); loadPicture();
  }));

  api('/api/seatmap').then((d) => { seats = d.seats; roster = d.roster || []; render(); loadPicture(); }).catch((e) => say(e.message, true));
})();

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

  const WORDS = { ok: 'OK', warn: 'Check', fail: 'Problem', info: 'Info' };

  async function loadChecks() {
    try {
      const d = await api('/api/health');
      const o = $('#overall');
      o.textContent = d.overall === 'ok' ? 'All good' : d.overall === 'warn' ? 'Needs a look' : 'Problem found';
      o.className = 'chip ' + (d.overall === 'ok' ? 'good' : d.overall === 'fail' ? 'bad' : 'warnchip');
      $('#checks').replaceChildren(...d.items.map((i) => {
        const row = el('div', 'check lvl-' + i.level);
        row.appendChild(el('span', 'dot'));
        const body = el('div', 'check-body');
        body.appendChild(el('div', 'check-title', i.title + ' · ' + (WORDS[i.level] || i.level)));
        body.appendChild(el('div', 'check-detail', i.detail));
        if (i.advice) body.appendChild(el('div', 'check-advice', i.advice));
        row.appendChild(body);
        return row;
      }));
    } catch (e) { /* transient */ }
  }

  let prof = null;
  function showMsg(text) {
    const m = $('#prof-msg');
    m.textContent = text || '';
    m.classList.toggle('hidden', !text);
  }

  async function loadProfiles() {
    prof = await api('/api/profile');
    const cur = prof.profiles.find((p) => p.name === prof.current);
    $('#prof-current').textContent = 'running: ' + (cur ? cur.title : prof.current);
    $('#profiles').replaceChildren(...prof.profiles.map((p) => {
      const b = el('button', 'profile-btn' + (p.name === prof.current ? ' current' : '') + (p.name === prof.saved && p.name !== prof.current ? ' chosen' : ''));
      b.type = 'button';
      b.appendChild(el('span', 'profile-title', p.title));
      b.appendChild(el('span', 'profile-for', p.for_whom));
      b.appendChild(el('span', 'profile-tag ' + (p.tested ? 'tested' : 'untested'), p.tested ? 'tested' : 'NOT tested on real hardware'));
      b.addEventListener('click', () => choose(p.name));
      return b;
    }));
    const pending = prof.saved && prof.saved !== prof.current;
    $('#restart').classList.toggle('hidden', !(pending && prof.supervised));
  }

  async function choose(name) {
    try {
      const r = await api('/api/profile', { method: 'POST', body: JSON.stringify({ profile: name }) });
      await loadProfiles();
      showMsg(r.note || (r.restart_needed
        ? 'Saved. It takes effect after ExamGuard restarts.' + (prof.supervised ? ' Press the button below.' : ' Stop it (Ctrl+C) and start it again.')
        : 'This profile is already running.'));
    } catch (e) { showMsg(e.message); }
  }

  $('#restart').addEventListener('click', async () => {
    if (!window.confirm('Restart ExamGuard now? Monitoring pauses for a few seconds. Do not do this during an exam.')) return;
    try {
      await api('/api/restart', { method: 'POST', body: '{}' });
      showMsg('Restarting… this page will reconnect in a few seconds.');
      setTimeout(() => location.reload(), 8000);
    } catch (e) { showMsg(e.message); }
  });

  loadChecks();
  loadProfiles().catch(() => {});
  setInterval(loadChecks, 4000);
})();

(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
  const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const full = (ts) => new Date(ts * 1000).toLocaleString();
  const pct = (x) => (x === null || x === undefined ? '–' : Math.round(x * 100) + '%');

  async function api(path, opts = {}) {
    const res = await fetch(path, { credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, ...opts });
    if (res.status === 401) { location.href = '/login'; throw new Error('Signed out'); }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }
  function say(text, warn) {
    const m = $('#msg');
    m.textContent = text;
    m.classList.toggle('warn', !!warn);
    m.classList.remove('hidden');
  }
  function stats(box, pairs) {
    box.replaceChildren(...pairs.map(([n, label]) => {
      const d = el('div', 'stat'); d.appendChild(el('b', '', String(n))); d.appendChild(el('span', '', label)); return d;
    }));
  }
  function table(t, head, rows) {
    const thead = el('thead'); const tr = el('tr');
    head.forEach((h) => tr.appendChild(el('th', '', h)));
    thead.appendChild(tr);
    const tb = el('tbody');
    rows.forEach((r) => { const row = el('tr'); r.forEach((c) => { const td = el('td'); if (c instanceof Node) td.appendChild(c); else td.textContent = c; row.appendChild(td); }); tb.appendChild(row); });
    if (!rows.length) { const row = el('tr'); const td = el('td', 'muted', 'Nothing yet.'); td.colSpan = head.length; row.appendChild(td); tb.appendChild(row); }
    t.replaceChildren(thead, tb);
  }

  let reasons = [];
  let kinds = [];
  let picked = null;

  // ----------------------------------------------------------- this exam
  async function loadExam() {
    const d = await api('/api/exam');
    const cur = d.current;
    $('#exam-chip').textContent = cur ? 'Running since ' + clock(cur.exam.started) : 'No exam running';
    $('#exam-chip').className = 'chip ' + (cur ? 'good' : '');
    $('#exam-start').classList.toggle('hidden', !!cur);
    $('#exam-finish').classList.toggle('hidden', !cur);
    if (cur) {
      const c = cur.counts;
      stats($('#exam-stats'), [[c.alerts, 'alerts'], [c.confirmed, 'confirmed'], [c.dismissed, 'dismissed'], [c.waiting, 'waiting'], [c.missed, 'reported missed']]);
    } else {
      stats($('#exam-stats'), []);
    }
    const pick = $('#exam-pick');
    pick.replaceChildren(...d.exams.map((e) => new Option('Exam ' + e.id + ' · ' + full(e.started) + (e.open ? ' (running)' : ''), e.id)));
    if (d.exams.length) {
      if (picked === null || !d.exams.some((e) => e.id === picked)) picked = d.exams[0].id;
      pick.value = String(picked);
      await loadSummary(picked);
    } else {
      $('#sum-table').replaceChildren(); $('#sum-stats').replaceChildren(); $('#sum-missed').textContent = 'No exam yet.';
    }
  }

  async function loadSummary(id) {
    const s = await api('/api/exams/' + id);
    const c = s.counts;
    stats($('#sum-stats'), [[c.alerts, 'alerts'], [c.confirmed, 'confirmed'], [c.dismissed, 'dismissed'],
      [c.waiting, 'still waiting' + (c.ignored ? ' (' + c.ignored + ' ignored)' : '')], [c.expired, 'expired'], [c.missed, 'reported missed']]);
    const rows = s.alerts.map((a) => {
      let reason = '';
      if (a.status === 'dismissed') {
        const sel = el('select');
        sel.appendChild(new Option('(no reason given)', ''));
        reasons.forEach((r) => sel.appendChild(new Option(r.label, r.key)));
        sel.value = a.dismiss_reason || '';
        sel.addEventListener('change', async () => {
          if (!sel.value) { sel.value = a.dismiss_reason || ''; return; }
          try { await api('/api/incidents/' + a.id + '/reason', { method: 'POST', body: JSON.stringify({ reason: sel.value }) }); a.dismiss_reason = sel.value; say('Reason saved.'); loadReport(); }
          catch (e) { say(e.message, true); sel.value = a.dismiss_reason || ''; }
        });
        reason = sel;
      } else if (a.status === 'pending') {
        reason = a.ignored ? 'ignored for now' : 'waiting for review';
      }
      return [clock(a.ts), a.student + (a.seat ? ' · ' + a.seat : ''), a.kind_label, a.status + (a.reviewer ? ' (' + a.reviewer + ')' : ''), reason];
    });
    table($('#sum-table'), ['Time', 'Student', 'Rule', 'Outcome', 'Reason'], rows);
    $('#sum-missed').textContent = s.missed.length
      ? s.missed.map((m) => clock(m.ts) + ' ' + m.kind_label + (m.note ? ' (' + m.note + ')' : '') + ' – ' + m.reported_by).join(' · ')
      : 'Nothing reported.';
    if (s.exam.open && c.waiting) say(c.waiting + ' alert(s) are still waiting for a decision.', false);
  }

  $('#exam-pick').addEventListener('change', (e) => { picked = Number(e.target.value); loadSummary(picked).catch((x) => say(x.message, true)); });
  $('#exam-start').addEventListener('click', async () => { try { await api('/api/exam/start', { method: 'POST', body: '{}' }); picked = null; await loadExam(); } catch (e) { say(e.message, true); } });
  $('#exam-finish').addEventListener('click', async () => {
    if (!window.confirm('Finish this exam? Alerts still waiting stay in the queue; you can still correct dismissal reasons afterwards.')) return;
    try { const s = await api('/api/exam/finish', { method: 'POST', body: '{}' }); picked = s.exam.id; await loadExam(); await loadReport(); say('Exam finished. Check the reasons below, then look at the report.'); }
    catch (e) { say(e.message, true); }
  });
  $('#miss-send').addEventListener('click', async () => {
    try {
      await api('/api/missed', { method: 'POST', body: JSON.stringify({ kind: $('#miss-kind').value, minutes_ago: Number($('#miss-ago').value || 0), note: $('#miss-note').value }) });
      $('#miss-note').value = '';
      say('Recorded. Thank you: this shows where ExamGuard misses things.');
      await loadExam(); await loadReport();
    } catch (e) { say(e.message, true); }
  });

  // -------------------------------------------------------------- report
  async function loadReport() {
    const r = await api('/api/learning/report');
    table($('#rep-table'), ['Rule', 'Alerts', 'Confirmed', 'Dismissed', 'of which normal behaviour', 'bad detection', 'Undecided / ignored', 'Reported missed', 'Dismissed share'],
      r.rules.map((g) => [g.kind_label, g.alerts, g.confirmed, g.dismissed, g.normal_behaviour, g.bad_detection, g.not_decided, g.missed, pct(g.dismissed_share) + (g.decided && g.decided < 20 ? ' (only ' + g.decided + ' decided)' : '')]));
    table($('#teach-table'), ['Teacher', 'Decided', 'Confirmed', 'Dismissed', 'Dismissed share'],
      r.teachers.map((t) => [t.reviewer, t.decided, t.confirmed, t.dismissed, pct(t.dismissed_share)]));
  }

  // --------------------------------------------------------- suggestions
  async function loadSuggestions() {
    const d = await api('/api/learning/suggestions');
    $('#min-n').textContent = d.thresholds.min_decided;
    const box = $('#sug-list');
    if (!d.suggestions.length) box.replaceChildren(el('p', 'muted small-text', 'No suggestions. Press "Look for suggestions" after you have reviewed enough alerts.'));
    else box.replaceChildren(...d.suggestions.map((s) => {
      const card = el('div', 'sugg' + (s.status === 'pending' ? '' : ' decided'));
      const word = s.direction === 'less_sensitive' ? 'less sensitive (fewer alerts)' : 'more sensitive (more alerts)';
      card.appendChild(el('h4', '', s.kind_label + ': change ' + s.setting + ' from ' + s.current + ' to ' + s.suggested + ' (' + word + ')'));
      const e = s.evidence || {};
      card.appendChild(el('div', 'muted small-text',
        'Based on ' + e.decided + ' decided alerts: ' + e.confirmed + ' confirmed, ' + e.dismissed + ' dismissed (' + e.normal_behaviour + ' for normal behaviour, ' + e.bad_detection + ' bad detection), ' + e.missed + ' reported missed. This setting means: ' + (e.meaning || '') + '.'));
      if (s.status === 'pending') {
        const row = el('div', 'actions');
        const no = el('button', 'btn small', 'Reject');
        const yes = el('button', 'btn small primary', 'Approve');
        no.disabled = yes.disabled = !d.can_approve;
        no.addEventListener('click', () => decide(s.id, 'reject'));
        yes.addEventListener('click', () => { if (window.confirm('Approve this change? It takes effect the next time ExamGuard starts.')) decide(s.id, 'approve'); });
        row.append(no, yes);
        card.appendChild(row);
        if (!d.can_approve) card.appendChild(el('div', 'muted small-text', 'Only an administrator can approve or reject.'));
      } else {
        card.appendChild(el('div', 'small-text', s.status === 'approved' ? 'Approved by ' + s.decided_by + '. Applies from the next start.' : 'Rejected by ' + s.decided_by + '.'));
      }
      return card;
    }));
    const keys = Object.keys(d.overrides);
    $('#overrides').textContent = keys.length ? 'Approved changes that apply at start: ' + keys.map((k) => k + ' = ' + d.overrides[k]).join(', ') : 'No approved changes yet: ExamGuard uses its normal settings.';
  }
  async function decide(id, verb) {
    try { const out = await api('/api/learning/suggestions/' + id + '/' + verb, { method: 'POST', body: '{}' }); say(verb === 'approve' ? 'Approved. Restart ExamGuard for it to apply.' : 'Rejected.'); await loadSuggestions(); }
    catch (e) { say(e.message, true); }
  }
  $('#sug-refresh').addEventListener('click', async () => {
    try { const d = await api('/api/learning/suggestions/refresh', { method: 'POST', body: '{}' }); say(d.new ? d.new + ' new suggestion(s).' : 'No new suggestions: not enough decided alerts yet, teachers disagree, or the numbers do not point to a change.'); await loadSuggestions(); }
    catch (e) { say(e.message, true); }
  });

  // ---------------------------------------------------------------- boot
  (async () => {
    try {
      const r = await api('/api/reasons');
      reasons = r.reasons; kinds = r.kinds;
      $('#miss-kind').replaceChildren(...kinds.map((k) => new Option(k.label, k.key)));
      await loadExam(); await loadReport(); await loadSuggestions();
    } catch (e) { say(e.message, true); }
  })();
})();

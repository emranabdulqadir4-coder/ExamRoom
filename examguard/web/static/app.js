(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const baseTitle = document.title;
  const state = { cam: 0, tab: 'pending', current: null, unseen: 0, audio: null, desktop: false, fresh: new Set() };

  let roster = [];
  let reasons = [];
  let marks = { write: false, column: 'Mark', value: 'Fail' };

  function toast(text, warn) {
    const t = $('#toast');
    t.textContent = text;
    t.classList.toggle('warn', !!warn);
    t.classList.remove('hidden');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => t.classList.add('hidden'), warn ? 12000 : 6000);
  }
  const STATUS_TEXT = { pending: 'Needs review', confirmed: 'Confirmed', dismissed: 'Dismissed', expired: 'Expired' };

  // ---------------------------------------------------------------- helpers
  async function api(path, opts = {}) {
    const res = await fetch(path, {
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    if (res.status === 401) { location.href = '/login'; throw new Error('Signed out'); }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const fullTime = (ts) => new Date(ts * 1000).toLocaleString();
  function ago(ts) {
    const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
    if (s < 60) return s + ' s ago';
    if (s < 3600) return Math.round(s / 60) + ' min ago';
    if (s < 86400) return Math.round(s / 3600) + ' h ago';
    return Math.round(s / 86400) + ' d ago';
  }

  // ------------------------------------------------------------------ list
  function card(inc) {
    const b = el('button', 'card sev-' + inc.severity + (state.fresh.has(inc.id) ? ' fresh' : ''));
    b.type = 'button';
    b.dataset.id = inc.id;
    if (inc.has_thumb) {
      const img = el('img', 'thumb');
      img.src = '/media/' + inc.id + '/thumb';
      img.alt = '';
      img.loading = 'lazy';
      b.appendChild(img);
    } else {
      b.appendChild(el('div', 'thumb none', 'no image'));
    }
    const main = el('div', 'card-main');
    main.appendChild(el('div', 'card-title', inc.student + (inc.seat ? ' · seat ' + inc.seat : '') + ' · ' + inc.kind_label));
    main.appendChild(el('div', 'card-sub', clock(inc.ts) + ' · ' + ago(inc.ts)));
    const badges = el('div', 'badges');
    badges.appendChild(el('span', 'badge ' + inc.severity, inc.severity));
    badges.appendChild(el('span', 'badge st-' + inc.status, inc.ignored ? 'Ignored for now' : (STATUS_TEXT[inc.status] || inc.status)));
    if (inc.status === 'dismissed' && inc.dismiss_reason_label) badges.appendChild(el('span', 'badge reason-badge', inc.dismiss_reason_label));
    main.appendChild(badges);
    b.appendChild(main);
    b.addEventListener('click', () => openModal(inc.id));
    return b;
  }

  function auditRow(a) {
    const row = el('div', 'audit-row');
    row.appendChild(el('span', 't', clock(a.ts)));
    row.appendChild(el('span', '', a.actor + ' · ' + a.action.replace(/_/g, ' ') + (a.incident_id ? ' #' + a.incident_id : '')));
    if (a.detail) row.appendChild(el('span', 'd', a.detail));
    return row;
  }

  async function loadList() {
    const list = $('#list');
    let nodes = [];
    if (state.tab === 'audit') {
      const { audit } = await api('/api/audit?limit=200');
      nodes = audit.map(auditRow);
      if (!nodes.length) nodes = [el('div', 'empty', 'Nothing recorded yet.')];
    } else {
      const { incidents, pending } = await api('/api/incidents?status=' + state.tab);
      setPending(pending);
      nodes = incidents.map(card);
      if (!nodes.length) {
        nodes = [el('div', 'empty', state.tab === 'pending'
          ? 'No alerts waiting. New ones appear here instantly.' : 'No reviewed incidents yet.')];
      }
    }
    list.replaceChildren(...nodes);
  }

  function setPending(n) {
    const b = $('#count-pending');
    b.textContent = n;
    b.classList.toggle('zero', n === 0);
  }

  // ----------------------------------------------------------------- modal
  const modal = $('#modal');
  const video = $('#m-video');

  async function openModal(id) {
    let inc;
    try { inc = await api('/api/incidents/' + id); } catch (e) { return; }
    state.current = inc;
    fillModal(inc);
    $('#m-note').value = inc.note || '';
    $('#m-error').classList.add('hidden');
    $('#m-reasons').classList.add('hidden');
    if (inc.has_clip) {
      video.classList.remove('hidden');
      $('#m-no-clip').classList.add('hidden');
      video.src = '/media/' + inc.id + '/clip';
      video.play().catch(() => {});
    } else {
      video.classList.add('hidden');
      $('#m-no-clip').classList.remove('hidden');
    }
    if (!modal.open) modal.showModal();
  }

  function fillModal(inc) {
    $('#m-title').textContent = inc.student + (inc.seat ? ' · seat ' + inc.seat : '') + ' · ' + inc.kind_label;
    $('#m-rule').textContent = inc.kind_label + ' (' + inc.severity + ' priority)';
    $('#m-details').textContent = inc.details;
    $('#m-when').textContent = fullTime(inc.ts) + ' · ' + inc.room + (inc.camera ? ' · ' + inc.camera : '') + (inc.seat ? ' · seat ' + inc.seat : '');
    let st = STATUS_TEXT[inc.status] || inc.status;
    if (inc.reviewer) st += ' by ' + inc.reviewer + (inc.reviewed_ts ? ' at ' + clock(inc.reviewed_ts) : '');
    if (inc.status === 'dismissed' && inc.dismiss_reason_label) st += ' (' + inc.dismiss_reason_label + ')';
    if (inc.ignored) st = 'Ignored for now by ' + inc.ignored_by + ' (still needs a decision)';
    $('#m-status').textContent = st;
    const locked = inc.status === 'expired';
    $('#m-confirm').disabled = locked;
    $('#m-dismiss').disabled = locked;
    const ig = $('#m-ignore');
    ig.disabled = inc.status !== 'pending';
    ig.textContent = inc.ignored ? 'Stop ignoring' : 'Ignore for now';
    fillIdentity(inc, locked);
  }

  function fillIdentity(inc, locked) {
    const box = $('#m-identity');
    if (!roster.length) { box.classList.add('hidden'); return; }
    box.classList.remove('hidden');
    const sel = $('#m-student');
    if (!sel.options.length) {
      sel.appendChild(new Option('Not identified', ''));
      roster.forEach((s) => sel.appendChild(new Option(s.name + (s.student_id ? ' (' + s.student_id + ')' : ''), s.ref)));
    }
    sel.value = inc.student_ref || '';
    sel.disabled = locked;
    const photo = $('#m-roster-photo');
    photo.classList.toggle('none', !inc.student_ref);
    if (inc.student_ref) photo.src = '/api/roster/' + encodeURIComponent(inc.student_ref) + '/photo';
    let note = inc.identity_note || 'No name was suggested for this student: choose one if you know who it is.';
    if (inc.match_score && !inc.id_corrected) note += ' (match score ' + inc.match_score.toFixed(2) + ')';
    $('#m-id-note').textContent = note;
  }

  $('#m-student').addEventListener('change', async (e) => {
    if (!state.current) return;
    try {
      const inc = await api('/api/incidents/' + state.current.id + '/student', {
        method: 'POST', body: JSON.stringify({ ref: e.target.value }) });
      state.current = inc;
      fillModal(inc);
      loadList().catch(() => {});
      if (inc.mark_warning) toast(inc.mark_warning, true);
      else if (inc.mark_note) toast('Spreadsheet updated: ' + inc.mark_note);
    } catch (err) {
      const box = $('#m-error');
      box.textContent = err.message;
      box.classList.remove('hidden');
    }
  });

  function closeModal() {
    video.pause();
    video.removeAttribute('src');
    video.load();
    state.current = null;
  }
  modal.addEventListener('close', closeModal);
  $('#m-close').addEventListener('click', () => modal.close());

  function buildReasonChips() {
    const box = $('#m-reason-chips');
    box.replaceChildren(...reasons.map((r) => {
      const b = el('button', 'btn small reason', r.label);
      b.type = 'button';
      b.dataset.reason = r.key;
      b.addEventListener('click', () => review('dismiss', r.key));
      return b;
    }));
  }

  async function review(decision, reason) {
    const inc = state.current;
    if (!inc) return;
    if (decision === 'confirm') {
      let msg = 'Confirm this incident for ' + inc.student + (inc.seat ? ' (seat ' + inc.seat + ')' : '') + '?\n\nIt will be logged for administrative action under your name.';
      if (inc.student_name && !inc.id_corrected) {
        msg += '\n\nThe name was only suggested (' + (inc.id_source === 'seat' ? 'by the seat map' : 'by face matching') + '): make sure it is the right student.';
      }
      if (marks.write) {
        msg += inc.student_ref
          ? '\n\n"' + marks.value + '" will be written in the ' + marks.column + ' column of the spreadsheet for ' + inc.student_name + '.'
          : '\n\nNo student is identified, so NOTHING will be written to the spreadsheet. Cancel and pick the student first if you know who it is.';
      }
      if (!window.confirm(msg)) return;
    }
    const buttons = [$('#m-confirm'), $('#m-dismiss')];
    buttons.forEach((b) => { b.disabled = true; });
    try {
      const done = await api('/api/incidents/' + inc.id + '/review', {
        method: 'POST',
        body: JSON.stringify({ decision, note: $('#m-note').value, reason: decision === 'dismiss' ? reason : undefined }),
      });
      modal.close();
      if (done.mark_warning) toast(done.mark_warning, true);
      else if (done.mark_note) toast('Spreadsheet updated: ' + done.mark_note);
      state.fresh.delete(inc.id);
      await loadList();
    } catch (e) {
      const err = $('#m-error');
      err.textContent = e.message;
      err.classList.remove('hidden');
      buttons.forEach((b) => { b.disabled = false; });
    }
  }
  $('#m-confirm').addEventListener('click', () => review('confirm'));
  $('#m-dismiss').addEventListener('click', () => {
    // Dismissing takes one more tap: the reason. (Without the reason list, fall back to a plain dismissal.)
    if (!reasons.length) { review('dismiss'); return; }
    $('#m-reasons').classList.toggle('hidden');
  });
  $('#m-ignore').addEventListener('click', async () => {
    const inc = state.current;
    if (!inc) return;
    try {
      await api('/api/incidents/' + inc.id + '/ignore', { method: 'POST', body: JSON.stringify({ on: !inc.ignored }) });
      modal.close();
      toast(inc.ignored ? 'Back in the queue.' : 'Ignored for now: it is still waiting for a decision and is not counted as a false alarm.');
      await loadList();
    } catch (e) {
      const err = $('#m-error');
      err.textContent = e.message;
      err.classList.remove('hidden');
    }
  });

  // ------------------------------------------------------------------ tabs
  $$('.tab').forEach((t) => t.addEventListener('click', () => {
    $$('.tab').forEach((x) => x.classList.toggle('active', x === t));
    state.tab = t.dataset.tab;
    loadList().catch(() => {});
  }));

  // ---------------------------------------------------------------- alerts
  function beep() {
    try {
      if (!state.audio) state.audio = new (window.AudioContext || window.webkitAudioContext)();
      const ctx = state.audio;
      if (ctx.state === 'suspended') ctx.resume();
      [0, 0.22].forEach((delay) => {
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = 'sine';
        o.frequency.value = delay ? 660 : 880;
        g.gain.setValueAtTime(0.0001, ctx.currentTime + delay);
        g.gain.exponentialRampToValueAtTime(0.25, ctx.currentTime + delay + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + delay + 0.2);
        o.connect(g).connect(ctx.destination);
        o.start(ctx.currentTime + delay);
        o.stop(ctx.currentTime + delay + 0.22);
      });
    } catch (e) { /* audio is a nice-to-have */ }
  }

  function announce(inc) {
    beep();
    if (document.hidden) {
      state.unseen += 1;
      document.title = '(' + state.unseen + ') ' + baseTitle;
    }
    if (state.desktop && 'Notification' in window && Notification.permission === 'granted') {
      try {
        new Notification('Exam alert: ' + inc.student, { body: inc.kind_label + ' – ' + inc.details });
      } catch (e) { /* ignore */ }
    }
  }
  window.addEventListener('focus', () => { state.unseen = 0; document.title = baseTitle; });

  $('#notify-btn').addEventListener('click', async (ev) => {
    beep();                                          // unlocks audio (browsers need a click first)
    if ('Notification' in window) {
      const p = await Notification.requestPermission();
      state.desktop = p === 'granted';
    }
    ev.target.textContent = state.desktop ? 'Sound & desktop alerts on' : 'Sound alerts on';
  });

  function connect() {
    const conn = $('#conn');
    const set = (on, text) => {
      conn.classList.toggle('on', on);
      conn.classList.toggle('off', !on);
      conn.lastElementChild.textContent = text;
    };
    const es = new EventSource('/events');
    es.onopen = () => set(true, 'Live alerts connected');
    es.onerror = () => set(false, 'Reconnecting…');
    es.addEventListener('incident', (e) => {
      const inc = JSON.parse(e.data);
      state.fresh.add(inc.id);
      announce(inc);
      loadList().catch(() => {});
    });
    es.addEventListener('incident_updated', (e) => {
      const inc = JSON.parse(e.data);
      if (state.current && state.current.id === inc.id) {
        state.current = inc;
        fillModal(inc);
      }
      loadList().catch(() => {});
    });
  }

  // ----------------------------------------------------------- live status
  async function refreshStatus() {
    try {
      const all = await api('/api/status');
      const s = (all.cameras && all.cameras[state.cam]) ? { ...all, ...all.cameras[state.cam] } : all;
      const stateChip = $('#chip-state');
      const ok = s.state === 'running';
      stateChip.textContent = ok ? 'Live' : s.state;
      stateChip.classList.toggle('good', ok);
      stateChip.classList.toggle('bad', !ok);
      $('#chip-fps').textContent = s.fps + ' fps';
      $('#chip-students').textContent = s.students + (s.students === 1 ? ' face' : ' faces');
      const ph = $('#chip-phone');
      ph.textContent = s.phone_detector ? 'phone detector on' : 'phone detector off';
      ph.classList.toggle('good', s.phone_detector);
      const hd = $('#chip-hands');
      hd.textContent = s.hand_detector ? 'hand detector on' : 'hand detector off';
      hd.classList.toggle('good', s.hand_detector);
      setPending(all.pending);
      const hb = $('#health');
      const advice = s.face_health && s.face_health.advice;
      hb.textContent = advice || '';
      hb.classList.toggle('hidden', !advice);
    } catch (e) { /* transient */ }
  }

  // ------------------------------------------------------------------ demo
  $$('[data-demo]').forEach((b) => b.addEventListener('click', async () => {
    b.disabled = true;
    try { await api('/api/demo/trigger', { method: 'POST', body: JSON.stringify({ kind: b.dataset.demo, cam: state.cam }) }); }
    catch (e) { window.alert(e.message); }
    setTimeout(() => { b.disabled = false; }, 1500);
  }));

  // keep the live image alive if the stream drops
  const feed = $('#feed');
  feed.addEventListener('error', () => setTimeout(() => { feed.src = '/video_feed?cam=' + state.cam + '&t=' + Date.now(); }, 3000));
  $$('#cam-switch .cam-tab').forEach((b) => b.addEventListener('click', () => {
    state.cam = Number(b.dataset.cam);
    $$('#cam-switch .cam-tab').forEach((x) => x.classList.toggle('active', x === b));
    feed.src = '/video_feed?cam=' + state.cam + '&t=' + Date.now();
    refreshStatus();
  }));

  // ------------------------------------------------------------------ boot
  api('/api/roster').then((d) => {
    roster = d.roster || [];
    marks = { write: !!d.write_marks, column: d.mark_column, value: d.mark_value };
  }).catch(() => {});
  api('/api/reasons').then((d) => { reasons = d.reasons || []; buildReasonChips(); }).catch(() => {});
  connect();
  loadList().catch(() => {});
  refreshStatus();
  setInterval(refreshStatus, 2000);
  setInterval(() => { if (state.tab !== 'audit' && !modal.open) loadList().catch(() => {}); }, 30000);
})();

/**
 * BT Studio - shell UI (nav rail, run settings panel, mode titles)
 * demo.js owns all speech logic; this file only reacts to the active tab.
 */
(function () {
  'use strict';

  const MODES = {
    'tab-live': {
      key: 'live', icon: 'edit_note', title: 'Transcript',
      desc: 'Real-time English speech-to-text',
      detail: 'Streaming English speech recognition. Summaries on demand.',
    },
    'tab-translate': {
      key: 'translate', icon: 'hearing', title: 'Interpreter',
      desc: 'Live English → Korean interpretation',
      detail: 'Streaming recognition with continuous English → Korean translation and summaries.',
    },
    'tab-interpreter': {
      key: 'interpreter', icon: 'forum', title: 'Interpreter+',
      desc: 'Sentence-by-sentence English → Korean',
      detail: 'Each finished sentence appears next to its Korean translation.',
    },
    'tab-multilingual': {
      key: 'multilingual', icon: 'language', title: 'Multilingual',
      desc: 'Speech-to-text in 20 languages',
      detail: 'Automatic language detection. The detected language shows in the top bar.',
    },
    'tab-parakeet': {
      key: 'parakeet', icon: 'record_voice_over', title: 'Parakeet',
      desc: 'English speech-to-text',
      detail: 'Powered by NVIDIA Parakeet TDT 0.6B.',
    },
    'tab-conversation': {
      key: 'conversation', icon: 'groups', title: 'Conversation',
      desc: 'System and microphone as separate streams',
      detail: 'Dual-stream recognition: system audio on the left, your microphone on the right.',
    },
  };

  const WIDE = window.matchMedia('(min-width: 1200px)');
  const MOBILE = window.matchMedia('(max-width: 767px)');
  const body = document.body;
  const $ = (id) => document.getElementById(id);

  // ─── Per-viewer preferences (optional; page works without storage) ───
  function loadPref(key) {
    try { return localStorage.getItem('btstudio.' + key); } catch (e) { return null; }
  }
  function savePref(key, value) {
    try { localStorage.setItem('btstudio.' + key, value); } catch (e) { /* ignore */ }
  }

  // ─── Mode title, actions, and settings card follow the active tab ───
  function syncMode() {
    const active = document.querySelector('.tab-btn.tab-active');
    const mode = active && MODES[active.id];
    if (!mode) return;

    $('mode-title').textContent = mode.title;
    $('mode-desc').textContent = mode.desc;
    $('mode-name').textContent = mode.title;
    $('mode-icon').textContent = mode.icon;
    $('mode-detail').textContent = mode.detail;
    document.title = 'BT Studio - ' + mode.title;

    document.querySelectorAll('.mode-actions').forEach((el) => {
      el.classList.toggle('hidden', el.dataset.mode !== mode.key);
    });
  }

  const observer = new MutationObserver(syncMode);
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    observer.observe(btn, { attributes: true, attributeFilter: ['class'] });
    btn.addEventListener('click', () => {
      if (MOBILE.matches) body.classList.remove('nav-open');
    });
  });
  syncMode();

  // ─── Nav rail: collapse on desktop, drawer on mobile ───
  if (loadPref('nav') === 'collapsed') body.classList.add('nav-collapsed');

  $('nav-collapse-btn').addEventListener('click', () => {
    if (MOBILE.matches) {
      body.classList.remove('nav-open');
      return;
    }
    const collapsed = body.classList.toggle('nav-collapsed');
    savePref('nav', collapsed ? 'collapsed' : 'expanded');
  });
  $('nav-open-btn').addEventListener('click', () => body.classList.add('nav-open'));

  // ─── Run settings: docked on wide screens, slide-over below 1200px ───
  if (loadPref('settings') === 'closed') body.classList.add('settings-closed');

  function toggleSettings(forceClose) {
    if (WIDE.matches) {
      const closed = forceClose ? true : !body.classList.contains('settings-closed');
      body.classList.toggle('settings-closed', closed);
      savePref('settings', closed ? 'closed' : 'open');
    } else if (forceClose) {
      body.classList.remove('settings-overlay-open');
    } else {
      body.classList.toggle('settings-overlay-open');
    }
  }
  $('settings-toggle-btn').addEventListener('click', () => toggleSettings(false));
  $('settings-close-btn').addEventListener('click', () => toggleSettings(true));

  // ─── Close overlays ───
  function closeOverlays() {
    body.classList.remove('nav-open', 'settings-overlay-open');
  }
  $('scrim').addEventListener('click', closeOverlays);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeOverlays();
  });
  WIDE.addEventListener('change', closeOverlays);
  MOBILE.addEventListener('change', closeOverlays);
})();

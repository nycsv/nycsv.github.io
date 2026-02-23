/**
 * BT Studio - Main Client
 *
 * WebSocket-based real-time speech recognition client
 * Captures microphone audio, resamples to 16kHz PCM16LE, streams to ASR server
 */

// Fixed server URL
const SERVER_URL = 'wss://ai.eesungkim.com/ws';

// Pending summarization callback (resolved when server replies)
let summarizeCallback = null;

// State machine
const State = {
  IDLE: 'IDLE',
  CONNECTING: 'CONNECTING',
  CONNECTED: 'CONNECTED',
  RECORDING: 'RECORDING',
  STOPPING: 'STOPPING',
};

// Global state
let currentState = State.IDLE;
let ws = null;
let sessionId = null;
let audioContext = null;
let mediaStream = null;
let displayStream = null;
let workletNode = null;

// Tab groups: 'groupA' (Transcript/Interpreter/Interpreter+) or 'groupB' (Multilingual)
let currentTabGroup = 'groupA';
let activeTab = 'live';

// ════════════════════════════════════════════════════════════════════════════════
// GROUP A STATE (Transcript, Interpreter, Interpreter+) - fastconformer backend
// ════════════════════════════════════════════════════════════════════════════════
const groupA = {
  committedText: '',
  partialText: '',
  _rawCommitted: '',
  _formattedPrefix: '',
  _formattedRawLength: 0,
  translationCommitted: '',
  translationPartial: '',
  interpreterSentenceCount: 0,
  interpreterBuffer: { source: '', text: '' },
  interpreterLastFlushLen: 0,
  lastSummarizedWordCount: 0,
  summarizationInFlight: false,
  interpreterAutoScroll: true,
  transcriptAutoScroll: true,
  translateAutoScroll: true,
};

/**
 * Compute committedText from formatted prefix + raw tail for Group A.
 */
function groupA_computeCommittedText() {
  const rawTail = groupA._rawCommitted.substring(groupA._formattedRawLength);
  if (!groupA._formattedPrefix) return groupA._rawCommitted;
  if (!rawTail) return groupA._formattedPrefix;
  let prefix = groupA._formattedPrefix.replace(/[.!?]+\s*$/, '');
  if (!prefix.endsWith(' ') && !rawTail.startsWith(' ')) {
    prefix += ' ';
  }
  return prefix + rawTail;
}

// ════════════════════════════════════════════════════════════════════════════════
// GROUP B STATE (Multilingual) - qwen3 backend
// ════════════════════════════════════════════════════════════════════════════════
const groupB = {
  committedText: '',
  partialText: '',
  _rawCommitted: '',
  _formattedPrefix: '',
  _formattedRawLength: 0,
  detectedLanguage: '',
  multilingualAutoScroll: true,
  lastSummarizedWordCount: 0,
  summarizationInFlight: false,
};

/**
 * Compute committedText from formatted prefix + raw tail for Group B.
 */
function groupB_computeCommittedText() {
  const rawTail = groupB._rawCommitted.substring(groupB._formattedRawLength);
  if (!groupB._formattedPrefix) return groupB._rawCommitted;
  if (!rawTail) return groupB._formattedPrefix;
  let prefix = groupB._formattedPrefix.replace(/[.!?]+\s*$/, '');
  if (!prefix.endsWith(' ') && !rawTail.startsWith(' ')) {
    prefix += ' ';
  }
  return prefix + rawTail;
}

// Summarization state (shared)
let summarizeCallback = null;

// Audio gating: only send audio after server ack
let readyToSendAudio = false;
let ackResolver = null;

// Stats (shared)
let recordingStartTime = null;
let latestLatency = null;
let bufferFillPct = null;

// DOM elements
let dom = {};

// i18n strings (injected by template)
let i18n = {};

/**
 * Initialize application
 */
function init() {
  // Get i18n strings from global (set by template)
  i18n = window.ASR_I18N || {};

  // Cache DOM elements
  dom = {
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    micBtn: document.getElementById('mic-btn'),
    micIcon: document.getElementById('mic-icon'),
    micLabel: document.getElementById('mic-label'),
    micReadyDot: document.getElementById('mic-ready-dot'),
    micReadyText: document.getElementById('mic-ready-text'),
    waveform: document.getElementById('waveform'),
    transcriptPanel: document.getElementById('transcript-panel'),
    transcriptBox: document.getElementById('transcript-box'),
    committedSpan: document.getElementById('transcript-committed'),
    partialSpan: document.getElementById('transcript-partial'),
    cursor: document.getElementById('transcript-cursor'),
    placeholder: document.getElementById('transcript-placeholder'),
    errorMsg: document.getElementById('error-message'),
    statLatency: document.getElementById('stat-latency'),
    statDuration: document.getElementById('stat-duration'),
    statBuffer: document.getElementById('stat-buffer'),
    statusFooterDot: document.getElementById('status-footer-dot'),
    audioSource: document.getElementById('audio-source'),
    copyBtn: document.getElementById('copy-btn'),
    summarizeBtn: document.getElementById('summarize-btn'),
    summarizeBtnTranslate: document.getElementById('summarize-btn-translate'),
    transcriptContent: document.getElementById('transcript-content'),
    copyBtnEn: document.getElementById('copy-btn-en'),
    copyBtnKo: document.getElementById('copy-btn-ko'),
    tabLive: document.getElementById('tab-live'),
    tabTranslate: document.getElementById('tab-translate'),
    translateBox: document.getElementById('translate-box'),
    translateContentEn: document.getElementById('translate-content-en'),
    translateContentKo: document.getElementById('translate-content-ko'),
    translateCommittedEn: document.getElementById('translate-committed-en'),
    translatePartialEn: document.getElementById('translate-partial-en'),
    translateCommittedKo: document.getElementById('translate-committed-ko'),
    translatePartialKo: document.getElementById('translate-partial-ko'),
    translateScrollEn: document.getElementById('translate-scroll-en'),
    translateScrollKo: document.getElementById('translate-scroll-ko'),
    tabInterpreter: document.getElementById('tab-interpreter'),
    interpreterBox: document.getElementById('interpreter-box'),
    interpreterRows: document.getElementById('interpreter-rows'),
    interpreterScroll: document.getElementById('interpreter-scroll'),
    translatePlaceholder: document.getElementById('translate-placeholder'),
    interpreterPlaceholder: document.getElementById('interpreter-placeholder'),
    interpreterJumpBtn: document.getElementById('interpreter-jump-btn'),
    interpreterCopyEn: document.getElementById('interpreter-copy-en'),
    interpreterCopyKo: document.getElementById('interpreter-copy-ko'),
    transcriptJumpBtn: document.getElementById('transcript-jump-btn'),
    translateJumpBtn: document.getElementById('translate-jump-btn'),
    summaryFooterTranscript: document.getElementById('summary-footer-transcript'),
    summaryContentTranscript: document.getElementById('summary-content-transcript'),
    summaryFooterTranslate: document.getElementById('summary-footer-translate'),
    summaryContentTranslate: document.getElementById('summary-content-translate'),
    tabInterview: document.getElementById('tab-interview'),
    interviewBox: document.getElementById('interview-box'),
    tabInterview2: document.getElementById('tab-interview2'),
    interview2Box: document.getElementById('interview2-box'),
    audioTrackIndicators: document.getElementById('audio-track-indicators'),
    indicatorMic: document.getElementById('indicator-mic'),
    indicatorMicLabel: document.getElementById('indicator-mic-label'),
    indicatorSys: document.getElementById('indicator-sys'),
    indicatorSysLabel: document.getElementById('indicator-sys-label'),
    // Multilingual ASR tab
    tabMultilingual: document.getElementById('tab-multilingual'),
    multilingualBox: document.getElementById('multilingual-box'),
    multilingualScroll: document.getElementById('multilingual-scroll'),
    multilingualCommitted: document.getElementById('multilingual-committed'),
    multilingualPartial: document.getElementById('multilingual-partial'),
    multilingualCursor: document.getElementById('multilingual-cursor'),
    multilingualPlaceholder: document.getElementById('multilingual-placeholder'),
    multilingualLangBadge: document.getElementById('multilingual-lang-badge'),
    multilingualSourceLang: document.getElementById('multilingual-source-lang'),
    multilingualCopyBtn: document.getElementById('multilingual-copy-btn'),
    multilingualJumpBtn: document.getElementById('multilingual-jump-btn'),
  };

  // Attach event listeners
  dom.micBtn.addEventListener('click', handleMicClick);
  dom.copyBtn.addEventListener('click', handleCopyClick);
  dom.summarizeBtn.addEventListener('click', handleSummarizeClick);
  dom.summarizeBtnTranslate.addEventListener('click', handleSummarizeClick);
  document.getElementById('summary-btn-transcript-footer')?.addEventListener('click', handleSummarizeClick);
  document.getElementById('summary-btn-translate-footer')?.addEventListener('click', handleSummarizeClick);
  dom.copyBtnEn.addEventListener('click', () => handleTranslateCopy(dom.copyBtnEn, () => committedText + partialText));
  dom.copyBtnKo.addEventListener('click', () => handleTranslateCopy(dom.copyBtnKo, () => translationCommitted + translationPartial));
  dom.tabLive.addEventListener('click', () => setActiveTab('live'));
  dom.tabTranslate.addEventListener('click', () => setActiveTab('translate'));
  dom.tabInterpreter.addEventListener('click', () => setActiveTab('interpreter'));
  dom.tabInterview.addEventListener('click', () => setActiveTab('interview'));
  dom.tabInterview2.addEventListener('click', () => setActiveTab('interview2'));
  dom.tabMultilingual.addEventListener('click', () => setActiveTab('multilingual'));

  // Multilingual copy button
  dom.multilingualCopyBtn.addEventListener('click', () => {
    handleTranslateCopy(dom.multilingualCopyBtn, () => committedText + partialText);
  });

  // Audio source toggle buttons
  initAudioSourceToggle();

  // Interpreter scroll & controls
  initInterpreterControls();

  // Transcript / translate scroll behavior
  initTranscriptScrollBehavior();

  // Initial state
  setActiveTab('live');
  updateUI();
}

/**
 * Initialize audio source toggle buttons
 */
function initAudioSourceToggle() {
  const btns = document.querySelectorAll('.audio-src-btn');

  // Hide entire audio source selector on devices that don't support getDisplayMedia (mobile)
  const supportsDisplayMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia);
  if (!supportsDisplayMedia) {
    const selectorContainer = btns[0]?.closest('.flex.items-center.gap-1');
    if (selectorContainer) selectorContainer.style.display = 'none';
  }

  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.disabled) return;
      btns.forEach(b => b.classList.remove('audio-src-active'));
      btn.classList.add('audio-src-active');
      dom.audioSource.value = btn.dataset.source;
    });
  });
}

/**
 * Initialize interpreter scroll behavior and controls
 */
function initInterpreterControls() {
  // Jump to bottom button
  if (dom.interpreterJumpBtn) {
    dom.interpreterJumpBtn.addEventListener('click', () => {
      dom.interpreterScroll.scrollTop = dom.interpreterScroll.scrollHeight;
      interpreterAutoScroll = true;
    });
  }

  // Detect manual scroll → disable auto-scroll, show jump button
  if (dom.interpreterScroll) {
    dom.interpreterScroll.addEventListener('scroll', () => {
      const el = dom.interpreterScroll;
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
      if (!atBottom && interpreterAutoScroll) {
        interpreterAutoScroll = false;
        }
      if (atBottom && !interpreterAutoScroll) {
        interpreterAutoScroll = true;
        }
      // Show/hide jump button
      if (dom.interpreterJumpBtn) {
        dom.interpreterJumpBtn.classList.toggle('hidden', atBottom);
        if (!atBottom) dom.interpreterJumpBtn.style.display = 'flex';
      }
    });
  }

  // Copy interpreter English sentences
  if (dom.interpreterCopyEn) {
    dom.interpreterCopyEn.addEventListener('click', async () => {
      const rows = dom.interpreterRows.querySelectorAll('.interpreter-row:not(.interpreter-pending)');
      if (!rows.length) {
        showError('No sentences to copy');
        setTimeout(clearError, 2000);
        return;
      }
      const lines = [];
      rows.forEach((row) => {
        const src = row.querySelector('.interpreter-src')?.textContent || '';
        if (src) lines.push(src);
      });
      try {
        await navigator.clipboard.writeText(lines.join('\n'));
        const icon = dom.interpreterCopyEn.querySelector('.material-symbols-outlined');
        icon.textContent = 'check';
        setTimeout(() => { icon.textContent = 'content_copy'; }, 2000);
      } catch (e) {
        showError('Failed to copy');
        setTimeout(clearError, 2000);
      }
    });
  }

  // Copy interpreter Korean sentences
  if (dom.interpreterCopyKo) {
    dom.interpreterCopyKo.addEventListener('click', async () => {
      const rows = dom.interpreterRows.querySelectorAll('.interpreter-row:not(.interpreter-pending)');
      if (!rows.length) {
        showError('No sentences to copy');
        setTimeout(clearError, 2000);
        return;
      }
      const lines = [];
      rows.forEach((row) => {
        const tl = row.querySelector('.interpreter-tl')?.textContent || '';
        if (tl) lines.push(tl);
      });
      try {
        await navigator.clipboard.writeText(lines.join('\n'));
        const icon = dom.interpreterCopyKo.querySelector('.material-symbols-outlined');
        icon.textContent = 'check';
        setTimeout(() => { icon.textContent = 'content_copy'; }, 2000);
      } catch (e) {
        showError('Failed to copy');
        setTimeout(clearError, 2000);
      }
    });
  }
}

/**
 * Initialize transcript and translate scroll behavior (pause on manual scroll, jump button)
 */
function initTranscriptScrollBehavior() {
  // ── Live Transcript ──
  if (dom.transcriptBox) {
    dom.transcriptBox.addEventListener('scroll', () => {
      const el = dom.transcriptBox;
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
      if (!atBottom && transcriptAutoScroll) {
        transcriptAutoScroll = false;
      }
      if (atBottom && !transcriptAutoScroll) {
        transcriptAutoScroll = true;
      }
      if (dom.transcriptJumpBtn) {
        dom.transcriptJumpBtn.classList.toggle('hidden', atBottom);
        if (!atBottom) dom.transcriptJumpBtn.style.display = 'flex';
      }
    });
  }
  if (dom.transcriptJumpBtn) {
    dom.transcriptJumpBtn.addEventListener('click', () => {
      dom.transcriptBox.scrollTop = dom.transcriptBox.scrollHeight;
      transcriptAutoScroll = true;
      dom.transcriptJumpBtn.classList.add('hidden');
    });
  }

  // ── Live Translate ──
  const onTranslateScroll = (el) => {
    const atBottomEn = dom.translateScrollEn
      ? dom.translateScrollEn.scrollHeight - dom.translateScrollEn.scrollTop - dom.translateScrollEn.clientHeight < 60
      : true;
    const atBottomKo = dom.translateScrollKo
      ? dom.translateScrollKo.scrollHeight - dom.translateScrollKo.scrollTop - dom.translateScrollKo.clientHeight < 60
      : true;
    const atBottom = atBottomEn && atBottomKo;
    if (!atBottom && translateAutoScroll) {
      translateAutoScroll = false;
    }
    if (atBottom && !translateAutoScroll) {
      translateAutoScroll = true;
    }
    if (dom.translateJumpBtn) {
      dom.translateJumpBtn.classList.toggle('hidden', atBottom);
      if (!atBottom) dom.translateJumpBtn.style.display = 'flex';
    }
  };
  if (dom.translateScrollEn) dom.translateScrollEn.addEventListener('scroll', onTranslateScroll);
  if (dom.translateScrollKo) dom.translateScrollKo.addEventListener('scroll', onTranslateScroll);
  if (dom.translateJumpBtn) {
    dom.translateJumpBtn.addEventListener('click', () => {
      if (dom.translateScrollEn) dom.translateScrollEn.scrollTop = dom.translateScrollEn.scrollHeight;
      if (dom.translateScrollKo) dom.translateScrollKo.scrollTop = dom.translateScrollKo.scrollHeight;
      translateAutoScroll = true;
      dom.translateJumpBtn.classList.add('hidden');
    });
  }

  // ── Multilingual ASR ──
  if (dom.multilingualScroll) {
    dom.multilingualScroll.addEventListener('scroll', () => {
      const el = dom.multilingualScroll;
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
      if (!atBottom && multilingualAutoScroll) multilingualAutoScroll = false;
      if (atBottom && !multilingualAutoScroll) multilingualAutoScroll = true;
      if (dom.multilingualJumpBtn) {
        dom.multilingualJumpBtn.classList.toggle('hidden', atBottom);
        if (!atBottom) dom.multilingualJumpBtn.style.display = 'flex';
      }
    });
  }
  if (dom.multilingualJumpBtn) {
    dom.multilingualJumpBtn.addEventListener('click', () => {
      dom.multilingualScroll.scrollTop = dom.multilingualScroll.scrollHeight;
      multilingualAutoScroll = true;
      dom.multilingualJumpBtn.classList.add('hidden');
    });
  }
}

/**
 * Scroll interpreter to bottom if auto-scroll is on (Group A only)
 */
function interpreterScrollToBottom() {
  if (groupA.interpreterAutoScroll && dom.interpreterScroll) {
    dom.interpreterScroll.scrollTop = dom.interpreterScroll.scrollHeight;
  }
}

/**
 * Update audio track indicators in footer
 */
function updateTrackIndicators() {
  if (!dom.audioTrackIndicators) return;

  const micTracks = mediaStream ? mediaStream.getAudioTracks() : [];
  const sysTracks = displayStream ? displayStream.getAudioTracks() : [];
  const anyActive = micTracks.length > 0 || sysTracks.length > 0;

  // Show/hide the whole block
  dom.audioTrackIndicators.style.display = anyActive ? 'flex' : 'none';

  // Mic indicator
  if (dom.indicatorMic && dom.indicatorMicLabel) {
    if (micTracks.length > 0) {
      const t = micTracks[0];
      dom.indicatorMicLabel.textContent = t.readyState === 'live' ? 'Live' : t.readyState;
      dom.indicatorMicLabel.style.color = t.readyState === 'live' ? '#10b981' : '#ef4444';
      dom.indicatorMic.querySelector('.material-symbols-outlined').style.color =
        t.readyState === 'live' ? '#10b981' : '#ef4444';
      console.log('[mic]', t.label || 'unnamed', '| state:', t.readyState, '| settings:', t.getSettings());
    } else {
      dom.indicatorMicLabel.textContent = '--';
      dom.indicatorMicLabel.style.color = '';
      dom.indicatorMic.querySelector('.material-symbols-outlined').style.color = '';
    }
  }

  // System audio indicator
  if (dom.indicatorSys && dom.indicatorSysLabel) {
    if (sysTracks.length > 0) {
      const t = sysTracks[0];
      dom.indicatorSysLabel.textContent = t.readyState === 'live' ? 'Live' : t.readyState;
      dom.indicatorSysLabel.style.color = t.readyState === 'live' ? '#10b981' : '#ef4444';
      dom.indicatorSys.querySelector('.material-symbols-outlined').style.color =
        t.readyState === 'live' ? '#10b981' : '#ef4444';
      console.log('[sys]', t.label || 'unnamed', '| state:', t.readyState, '| settings:', t.getSettings());
    } else {
      dom.indicatorSysLabel.textContent = '--';
      dom.indicatorSysLabel.style.color = '';
      dom.indicatorSys.querySelector('.material-symbols-outlined').style.color = '';
    }
  }
}


/**
 * Tab switching - with tab group management
 */
function setActiveTab(tab) {
  const previousTabGroup = currentTabGroup;

  // Determine new tab group
  if (tab === 'multilingual') {
    currentTabGroup = 'groupB';
  } else {
    currentTabGroup = 'groupA';
  }

  // If switching between groups, stop current recording
  if (previousTabGroup !== currentTabGroup && currentState !== State.IDLE) {
    handleMicClick();
  }

  activeTab = tab;
  dom.transcriptPanel.classList.toggle('hidden', tab !== 'live');
  dom.translateBox.classList.toggle('hidden', tab !== 'translate');
  dom.interpreterBox.classList.toggle('hidden', tab !== 'interpreter');
  dom.interviewBox.classList.toggle('hidden', tab !== 'interview');
  dom.interview2Box.classList.toggle('hidden', tab !== 'interview2');
  dom.multilingualBox.classList.toggle('hidden', tab !== 'multilingual');

  updateTabButton(dom.tabLive, tab === 'live');
  updateTabButton(dom.tabTranslate, tab === 'translate');
  updateTabButton(dom.tabInterpreter, tab === 'interpreter');
  updateTabButton(dom.tabInterview, tab === 'interview');
  updateTabButton(dom.tabInterview2, tab === 'interview2');
  updateTabButton(dom.tabMultilingual, tab === 'multilingual');
}

function updateTabButton(button, isActive) {
  button.classList.toggle('tab-active', isActive);
  button.classList.toggle('text-primary', isActive);
  button.classList.toggle('text-slate-500', !isActive);
}

/**
 * Connect to WebSocket server and immediately start recording
 */
async function connectAndStart() {
  setState(State.CONNECTING);
  clearError();

  try {
    // Step 1: Connect WebSocket and wait for session_start
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Connection timeout')), 10000);

      ws = new WebSocket(SERVER_URL);

      ws.onopen = () => {
        console.log('WebSocket connected');
      };

      ws.onmessage = (event) => {
        let data;
        try { data = JSON.parse(event.data); } catch (e) { return; }
        if (data.type === 'session_start') {
          sessionId = data.session_id;
          console.log('Session started:', sessionId);
          clearTimeout(timeout);

          // Set up persistent handlers for the connection lifetime
          ws.onmessage = handleWebSocketMessage;
          ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            showError(i18n.errorConnection || '[Connection error] Server is not running');
            setState(State.IDLE);
          };
          ws.onclose = () => {
            console.log('WebSocket closed');
            if (currentState !== State.IDLE) {
              setState(State.IDLE);
              showError(i18n.errorDisconnected || 'Disconnected from server');
            }
            cleanup();
          };

          resolve();
        }
      };

      ws.onerror = (error) => {
        clearTimeout(timeout);
        console.error('WebSocket connection error:', error);
        reject(new Error('Connection failed'));
      };

      ws.onclose = () => {
        clearTimeout(timeout);
        reject(new Error('Connection closed'));
      };
    });

    // Step 2: Connected — now start recording
    setState(State.CONNECTED);
    await startRecording();

    // If still CONNECTED after startRecording, it means recording failed
    // (success sets state to RECORDING). Clean up by disconnecting.
    if (currentState === State.CONNECTED) {
      disconnectWebSocket();
    }

  } catch (error) {
    console.error('Failed to connect and start:', error);
    showError(i18n.errorConnection || 'Failed to connect to server');
    if (ws) { ws.close(); ws = null; }
    setState(State.IDLE);
    cleanup();
  }
}

/**
 * Disconnect from WebSocket
 */
function disconnectWebSocket() {
  if (ws) {
    ws.close();
    ws = null;
  }
  setState(State.IDLE);
  // Re-render transcript so full punctuation is restored after streaming ends
  updateTranscript();
}

/**
 * Handle WebSocket messages from server
 */
function handleWebSocketMessage(event) {
  // Parse JSON message
  let data;
  try {
    data = JSON.parse(event.data);
  } catch (e) {
    console.error('Failed to parse message:', e);
    return;
  }

  const msgType = data.type;
  console.log('Received:', msgType, JSON.stringify(data).substring(0, 200));

  switch (msgType) {
    case 'session_start':
      sessionId = data.session_id;
      console.log('Session started:', sessionId);
      setState(State.CONNECTED);
      break;

    case 'ack':
      console.log('Server acknowledged start');
      readyToSendAudio = true;
      if (ackResolver) {
        ackResolver();
        ackResolver = null;
      }
      break;

    case 'committed':
      handleCommittedText(data);
      break;

    case 'committed_formatted':
      handleCommittedFormatted(data);
      break;

    case 'partial':
      handlePartialText(data);
      break;

    case 'committed_translation':
      handleCommittedTranslation(data);
      break;

    case 'partial_translation':
      handlePartialTranslation(data);
      break;

    case 'final_translation':
      handleFinalTranslation(data);
      break;

    case 'language_detected':
      handleLanguageDetected(data);
      break;

    case 'ping':
      handlePing(data);
      break;

    case 'backpressure':
      handleBackpressure(data);
      break;

    case 'error':
      showError(data.message || 'Server error');
      break;

    case 'final':
      handleFinalResult(data);
      break;

    case 'summary':
      handleSummaryResult(data);
      break;

    default:
      console.warn('Unknown message type:', msgType);
  }
}

/**
 * Handle committed text from server
 */
function handleCommittedText(data) {
  const newText = data.text || '';

  if (currentTabGroup === 'groupA') {
    groupA._rawCommitted += newText;
    groupA.committedText = groupA_computeCommittedText();
    groupA.partialText = '';
    updateTranscript();

    const wordCount = groupA.committedText.split(/\s+/).filter(Boolean).length;
    if (!groupA.summarizationInFlight && wordCount - groupA.lastSummarizedWordCount >= 200) {
      requestSummarization(groupA.committedText);
    }
  } else if (currentTabGroup === 'groupB') {
    groupB._rawCommitted += newText;
    groupB.committedText = groupB_computeCommittedText();
    groupB.partialText = '';
    updateMultilingualTranscript();

    const wordCount = groupB.committedText.split(/\s+/).filter(Boolean).length;
    if (!groupB.summarizationInFlight && wordCount - groupB.lastSummarizedWordCount >= 200) {
      requestSummarization(groupB.committedText);
    }
  }
}

/**
 * Handle committed_formatted text from server (ITN + punctuation + capitalization).
 * The server sends the formatted version of ALL accumulated committed text so far.
 * raw_length tells us how many chars of raw text were covered by this formatting.
 * Display = formattedPrefix + raw tail (any raw chunks that arrived after formatting).
 */
function handleCommittedFormatted(data) {
  const formatted = data.text || '';
  const rawLength = data.raw_length || 0;

  if (currentTabGroup === 'groupA') {
    if (!formatted || rawLength <= groupA._formattedRawLength) return;
    groupA._formattedPrefix = formatted;
    groupA._formattedRawLength = rawLength;
    groupA.committedText = groupA_computeCommittedText();
    updateTranscript();
  } else if (currentTabGroup === 'groupB') {
    if (!formatted || rawLength <= groupB._formattedRawLength) return;
    groupB._formattedPrefix = formatted;
    groupB._formattedRawLength = rawLength;
    groupB.committedText = groupB_computeCommittedText();
    updateMultilingualTranscript();
  }
}

/**
 * Handle committed translation (finalized sentence translation) - Group A only
 */
function handleCommittedTranslation(data) {
  // Only Group A (fastconformer backend) uses translation
  if (currentTabGroup !== 'groupA') return;

  const tl = data.text || '';
  const source = data.source || '';
  groupA.translationCommitted += tl + ' ';
  groupA.translationPartial = '';
  updateTranscript();

  // Interpreter+ tab: use source from translation event (properly segmented sentences)
  groupA.interpreterBuffer.text += (groupA.interpreterBuffer.text ? ' ' : '') + tl;
  groupA.interpreterBuffer.source += (groupA.interpreterBuffer.source ? ' ' : '') + source;

  if (data.sentence_end) {
    addInterpreterRow(groupA.interpreterBuffer.source, groupA.interpreterBuffer.text);
    groupA.interpreterLastFlushLen = groupA.committedText.length;
    groupA.interpreterBuffer = { source: '', text: '' };
  } else {
    updateInterpreterBuffering(groupA.interpreterBuffer.source, groupA.interpreterBuffer.text);
  }
}

/**
 * Handle partial translation (in-progress preview) - Group A only
 */
function handlePartialTranslation(data) {
  // Only Group A (fastconformer backend) uses translation
  if (currentTabGroup !== 'groupA') return;

  groupA.translationPartial = data.translation || '';
  updateTranscript();

  // Interpreter+ tab: pending row uses translation source for English
  const partialSource = data.source || '';
  const combinedSrc = [groupA.interpreterBuffer.source, partialSource].filter(Boolean).join(' ');
  const combinedTl = [groupA.interpreterBuffer.text, groupA.translationPartial].filter(Boolean).join(' ');
  updateInterpreterPending(combinedSrc, combinedTl);
}

/**
 * Handle final_translation (full accumulated translation at end of session) - Group A only
 */
function handleFinalTranslation(data) {
  // Only Group A (fastconformer backend) uses translation
  if (currentTabGroup !== 'groupA') return;

  groupA.translationCommitted = data.translation || '';
  groupA.translationPartial = '';
  updateTranscript();
}

/**
 * Handle partial text from server
 */
function handlePartialText(data) {
  if (currentTabGroup === 'groupA') {
    groupA.partialText = data.text || '';
    if (data.translation !== undefined) {
      groupA.translationPartial = data.translation;
    }
    updateTranscript();
  } else if (currentTabGroup === 'groupB') {
    groupB.partialText = data.text || '';
    updateMultilingualTranscript();
  }

  // Update stats (shared)
  if (data.triton_call_ms !== undefined) {
    latestLatency = data.triton_call_ms;
  }
  if (data.buffer_fill_pct !== undefined) {
    bufferFillPct = data.buffer_fill_pct;
  }

  updateStats();
}

/**
 * Handle ping from server
 */
function handlePing(data) {
  const pongMsg = {
    type: 'pong',
    timestamp: Date.now() / 1000,
    client_timestamp: data.timestamp,
  };
  ws.send(JSON.stringify(pongMsg));
}

/**
 * Handle backpressure message
 */
function handleBackpressure(data) {
  const action = data.action;
  const fillPct = data.fill_pct;
  console.warn(`Backpressure: ${action}, buffer at ${fillPct}%`);
}

/**
 * Handle language_detected message from Qwen3-ASR (Group B only)
 */
function handleLanguageDetected(data) {
  if (currentTabGroup !== 'groupB') return;

  groupB.detectedLanguage = data.language || '';
  console.log('Language detected:', groupB.detectedLanguage);

  // Update Multilingual tab badge
  if (dom.multilingualLangBadge) {
    dom.multilingualLangBadge.textContent = groupB.detectedLanguage;
    dom.multilingualLangBadge.classList.remove('hidden');
  }
}

/**
 * Handle final result from server
 */
function handleFinalResult(data) {
  const finalText = data.transcription || '';

  if (currentTabGroup === 'groupA') {
    groupA.committedText = finalText;
    groupA.partialText = '';
    groupA.translationPartial = '';
    updateTranscript();
  } else if (currentTabGroup === 'groupB') {
    groupB.committedText = finalText;
    groupB.partialText = '';
    updateMultilingualTranscript();
  }

  setState(State.CONNECTED);
  cleanup();
}

/**
 * Handle mic button click (single button: start / stop+disconnect)
 */
async function handleMicClick() {
  if (currentState === State.IDLE) {
    await connectAndStart();
  } else if (currentState === State.RECORDING) {
    await stopRecording();
    disconnectWebSocket();
  }
}

/**
 * Handle copy button click (Group A only)
 */
async function handleCopyClick() {
  const fullText = groupA.committedText + groupA.partialText;

  if (!fullText) {
    showError(i18n.errorNoText || 'No transcription to copy');
    setTimeout(clearError, 2000);
    return;
  }

  try {
    await navigator.clipboard.writeText(fullText);

    const icon = dom.copyBtn.querySelector('.material-symbols-outlined');
    icon.textContent = 'check';
    setTimeout(() => { icon.textContent = 'content_copy'; }, 2000);
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    showError(i18n.errorCopyFailed || 'Failed to copy to clipboard');
    setTimeout(clearError, 2000);
  }
}

/**
 * Handle copy for translate panel buttons
 */
async function handleTranslateCopy(btn, getText) {
  const text = getText();
  if (!text.trim()) {
    showError(i18n.errorNoText || 'No text to copy');
    setTimeout(clearError, 2000);
    return;
  }

  try {
    await navigator.clipboard.writeText(text.trim());

    const icon = btn.querySelector('.material-symbols-outlined');
    icon.textContent = 'check';
    setTimeout(() => { icon.textContent = 'content_copy'; }, 2000);
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    showError(i18n.errorCopyFailed || 'Failed to copy to clipboard');
    setTimeout(clearError, 2000);
  }
}

/**
 * Wait for server ack with timeout
 */
function waitForAck(timeoutMs) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      ackResolver = null;
      reject(new Error('Timeout waiting for server ack'));
    }, timeoutMs);

    ackResolver = () => {
      clearTimeout(timer);
      resolve();
    };
  });
}

/**
 * Get the ASR backend based on the active tab
 * @returns {'fastconformer'|'qwen3'}
 */
function getAsrBackend() {
  if (activeTab === 'multilingual') {
    return 'qwen3';
  }
  return 'fastconformer';
}

/**
 * Get the selected audio source mode
 * @returns {'mic'|'system'|'both'}
 */
function getAudioSource() {
  return dom.audioSource ? dom.audioSource.value : 'mic';
}

/**
 * Start recording
 */
async function startRecording() {
  try {
    setState(State.RECORDING);
    clearError();
    readyToSendAudio = false;

    // Reset state based on current tab group
    if (currentTabGroup === 'groupA') {
      groupA.committedText = '';
      groupA.partialText = '';
      groupA._rawCommitted = '';
      groupA._formattedPrefix = '';
      groupA._formattedRawLength = 0;
      groupA.translationCommitted = '';
      groupA.translationPartial = '';
      groupA.lastSummarizedWordCount = 0;
      groupA.summarizationInFlight = false;

      // Hide summary footers for Group A
      dom.summaryContentTranscript.classList.add('hidden');
      dom.summaryFooterTranscript.classList.add('hidden');
      dom.summaryContentTranslate.classList.add('hidden');
      dom.summaryFooterTranslate.classList.add('hidden');

      updateTranscript();

      // Reset scroll state for Group A
      groupA.transcriptAutoScroll = true;
      groupA.translateAutoScroll = true;
      if (dom.transcriptJumpBtn) dom.transcriptJumpBtn.classList.add('hidden');
      if (dom.translateJumpBtn) dom.translateJumpBtn.classList.add('hidden');

      // Reset interpreter
      groupA.interpreterSentenceCount = 0;
      groupA.interpreterAutoScroll = true;
      groupA.interpreterBuffer = { source: '', text: '' };
      groupA.interpreterLastFlushLen = 0;
      if (dom.interpreterRows) dom.interpreterRows.innerHTML = '';
      if (dom.interpreterPlaceholder) dom.interpreterPlaceholder.style.display = 'flex';
    } else if (currentTabGroup === 'groupB') {
      groupB.committedText = '';
      groupB.partialText = '';
      groupB._rawCommitted = '';
      groupB._formattedPrefix = '';
      groupB._formattedRawLength = 0;
      groupB.detectedLanguage = '';
      groupB.lastSummarizedWordCount = 0;
      groupB.summarizationInFlight = false;

      updateMultilingualTranscript();

      // Reset scroll state for Group B
      groupB.multilingualAutoScroll = true;
      if (dom.multilingualLangBadge) dom.multilingualLangBadge.classList.add('hidden');
      if (dom.multilingualJumpBtn) dom.multilingualJumpBtn.classList.add('hidden');
    }

    const audioSource = getAudioSource();
    const needMic = audioSource === 'mic' || audioSource === 'both';
    const needSystem = audioSource === 'system' || audioSource === 'both';

    // Get system audio first if needed
    if (needSystem) {
      try {
        displayStream = await navigator.mediaDevices.getDisplayMedia({
          audio: true,
          video: true,
        });
        displayStream.getVideoTracks().forEach((t) => t.stop());

        if (!displayStream.getAudioTracks().length) {
          showError('No system audio track available. Make sure to check "Share audio" in the picker.');
          setState(State.CONNECTED);
          cleanup();
          return;
        }
      } catch (e) {
        console.warn('System audio capture cancelled or denied:', e.name);
        if (audioSource === 'system') {
          showError('System audio capture was cancelled.');
          setState(State.CONNECTED);
          cleanup();
          return;
        }
        displayStream = null;
      }
    }

    // Get microphone if needed
    if (needMic) {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
    }

    // Log and show track diagnostics
    updateTrackIndicators();

    // Create AudioContext
    audioContext = new (window.AudioContext || window.webkitAudioContext)();

    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    // Load AudioWorklet module
    const baseUrl = document.body.dataset.base || '';
    const workletUrl = `${baseUrl}/demo/audio-resampler-worklet.js`;
    await audioContext.audioWorklet.addModule(workletUrl);

    // Create worklet node
    workletNode = new AudioWorkletNode(audioContext, 'resampler-worklet');

    // Send start message
    const startMsg = {
      type: 'start',
      request_id: crypto.randomUUID(),
      sample_rate: 16000,
      channels: 1,
      bytes_per_sample: 2,
      client_t0_ns: Math.round(performance.now() * 1e6),
      asr_backend: getAsrBackend(),
      source_language: dom.multilingualSourceLang ? dom.multilingualSourceLang.value : 'auto',
    };
    ws.send(JSON.stringify(startMsg));

    // Wait for server ack
    await waitForAck(5000);

    // Handle resampled audio chunks
    let chunkCount = 0;
    workletNode.port.onmessage = (event) => {
      if (!readyToSendAudio) return;

      const pcm16Buffer = event.data;
      chunkCount++;

      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(pcm16Buffer);
      }
    };

    // Connect audio pipeline
    if (mediaStream) {
      const micSource = audioContext.createMediaStreamSource(mediaStream);
      micSource.connect(workletNode);
    }

    if (displayStream && displayStream.getAudioTracks().length > 0) {
      const systemSource = audioContext.createMediaStreamSource(displayStream);
      systemSource.connect(workletNode);
    }

    workletNode.connect(audioContext.destination);

    recordingStartTime = Date.now();
    updateStats();
  } catch (error) {
    console.error('Failed to start recording:', error);
    showError(i18n.errorMicAccess || 'Audio capture denied or server not ready');
    setState(State.CONNECTED);
    cleanup();
  }
}

/**
 * Stop recording
 */
async function stopRecording() {
  setState(State.STOPPING);

  try {
    if (workletNode) {
      workletNode.port.postMessage('stop');
      workletNode.disconnect();
      workletNode = null;
    }

    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
    if (displayStream) {
      displayStream.getTracks().forEach((track) => track.stop());
      displayStream = null;
    }

    if (audioContext) {
      await audioContext.close();
      audioContext = null;
    }

    const endMsg = {
      type: 'end',
      client_send_audio_start_ns: Math.round((recordingStartTime || Date.now()) * 1e6),
      client_send_audio_end_ns: Math.round(performance.now() * 1e6),
    };
    ws.send(JSON.stringify(endMsg));
  } catch (error) {
    console.error('Error stopping recording:', error);
    setState(State.CONNECTED);
    cleanup();
  }
}

/**
 * Cleanup audio resources
 */
function cleanup() {
  if (workletNode) {
    try {
      workletNode.port.postMessage('stop');
      workletNode.disconnect();
    } catch (e) {
      // Ignore
    }
    workletNode = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (displayStream) {
    displayStream.getTracks().forEach((track) => track.stop());
    displayStream = null;
  }

  if (audioContext) {
    audioContext.close().catch(() => { });
    audioContext = null;
  }

  recordingStartTime = null;
  latestLatency = null;
  bufferFillPct = null;
  readyToSendAudio = false;
  ackResolver = null;

  // Clear track indicators
  if (dom.audioTrackIndicators) dom.audioTrackIndicators.style.display = 'none';
}

/**
 * Update UI based on current state
 */
function setState(newState) {
  currentState = newState;
  updateUI();
}

/**
 * Update UI elements
 */
function updateUI() {
  // Status dot in header
  dom.statusDot.className = 'asr-status-dot';
  switch (currentState) {
    case State.IDLE:
      dom.statusDot.classList.add('idle');
      dom.statusText.textContent = i18n.statusIdle || 'Disconnected';
      break;
    case State.CONNECTING:
      dom.statusDot.classList.add('connecting');
      dom.statusText.textContent = i18n.statusConnecting || 'Connecting...';
      break;
    case State.CONNECTED:
      dom.statusDot.classList.add('connected');
      dom.statusText.textContent = i18n.statusConnected || 'Connected';
      break;
    case State.RECORDING:
      dom.statusDot.classList.add('connected');
      dom.statusText.textContent = i18n.statusRecording || 'Listening';
      break;
    case State.STOPPING:
      dom.statusDot.classList.add('connecting');
      dom.statusText.textContent = i18n.statusStopping || 'Stopping...';
      break;
  }

  // Mic button — single button (start / stop+disconnect)
  dom.micBtn.disabled = currentState === State.STOPPING;

  if (currentState === State.RECORDING) {
    dom.micBtn.classList.add('recording');
    dom.micIcon.textContent = 'stop';
    dom.micLabel.textContent = i18n.micLabelStop || 'Stop';
    dom.micReadyDot.className = 'w-2 h-2 rounded-full bg-red-500 animate-pulse';
    dom.micReadyText.textContent = 'Listening';
    dom.waveform.classList.add('waveform-active');
  } else if (currentState === State.CONNECTING) {
    // Keep showing Start while connecting — no visual change on the button
    dom.micBtn.classList.remove('recording');
    dom.micIcon.textContent = 'power_settings_new';
    dom.micLabel.textContent = i18n.micLabelStart || 'Start';
    dom.micReadyDot.className = 'w-2 h-2 rounded-full bg-yellow-500 animate-pulse';
    dom.micReadyText.textContent = 'Connecting...';
    dom.waveform.classList.remove('waveform-active');
  } else {
    dom.micBtn.classList.remove('recording');
    dom.micIcon.textContent = 'power_settings_new';
    dom.micLabel.textContent = i18n.micLabelStart || 'Start';
    dom.waveform.classList.remove('waveform-active');
    if (currentState === State.CONNECTED) {
      dom.micReadyDot.className = 'w-2 h-2 rounded-full bg-green-500 animate-pulse';
      dom.micReadyText.textContent = 'Ready';
    } else {
      dom.micReadyDot.className = 'w-2 h-2 rounded-full bg-slate-600';
      dom.micReadyText.textContent = 'Mic: Standby';
    }
  }

  // Cursor visibility
  if (currentState === State.RECORDING) {
    dom.cursor.style.display = 'inline-block';
  } else {
    dom.cursor.style.display = 'none';
  }

  // Footer connection bars
  if (dom.statusFooterDot) {
    const bars = dom.statusFooterDot.querySelectorAll('div');
    const connected = currentState === State.CONNECTED || currentState === State.RECORDING;
    bars.forEach(b => {
      b.style.background = connected ? '#10b981' : '#4b5563';
    });
  }

  // Disable audio source selector while recording
  const srcDisabled = currentState === State.RECORDING || currentState === State.STOPPING;
  if (dom.audioSource) {
    dom.audioSource.disabled = srcDisabled;
  }
  // Also disable toggle buttons
  document.querySelectorAll('.audio-src-btn').forEach(btn => {
    btn.disabled = srcDisabled;
  });

}

/**
 * Update transcript display (GROUP A only: Transcript, Interpreter, Interpreter+)
 */
function updateTranscript() {
  // Only update if we're in Group A
  if (currentTabGroup !== 'groupA') return;

  // While streaming, hide trailing punctuation — it looks premature to the user
  const isStreaming = currentState === State.RECORDING || currentState === State.CONNECTED;
  const displayCommitted = isStreaming ? groupA.committedText.replace(/[.!?]+\s*$/, '') : groupA.committedText;

  dom.committedSpan.textContent = displayCommitted;
  dom.partialSpan.textContent = groupA.partialText;

  // Toggle placeholder
  if (dom.placeholder) {
    dom.placeholder.style.display = (groupA.committedText || groupA.partialText) ? 'none' : 'flex';
  }

  // Auto-scroll to bottom (paused when user has scrolled up)
  if (groupA.transcriptAutoScroll) {
    dom.transcriptBox.scrollTop = dom.transcriptBox.scrollHeight;
  }

  // Update translate tab mirrors
  dom.translateCommittedEn.textContent = displayCommitted;
  dom.translatePartialEn.textContent = groupA.partialText;
  dom.translateCommittedKo.textContent = groupA.translationCommitted;
  dom.translatePartialKo.textContent = groupA.translationPartial;

  // Toggle translate placeholder
  if (dom.translatePlaceholder) {
    dom.translatePlaceholder.style.display = (groupA.committedText || groupA.partialText) ? 'none' : 'flex';
  }

  // Auto-scroll translate panels (paused when user has scrolled up)
  if (groupA.translateAutoScroll) {
    dom.translateScrollEn.scrollTop = dom.translateScrollEn.scrollHeight;
    dom.translateScrollKo.scrollTop = dom.translateScrollKo.scrollHeight;
  }
}

/**
 * Update transcript display (GROUP B only: Multilingual)
 */
function updateMultilingualTranscript() {
  // Only update if we're in Group B
  if (currentTabGroup !== 'groupB') return;

  const isStreaming = currentState === State.RECORDING || currentState === State.CONNECTED;
  const displayCommitted = isStreaming ? groupB.committedText.replace(/[.!?]+\s*$/, '') : groupB.committedText;

  if (dom.multilingualCommitted) {
    dom.multilingualCommitted.textContent = displayCommitted;
    dom.multilingualPartial.textContent = groupB.partialText;
    if (dom.multilingualCursor) {
      dom.multilingualCursor.style.display = currentState === State.RECORDING ? 'inline-block' : 'none';
    }
    if (dom.multilingualPlaceholder) {
      dom.multilingualPlaceholder.style.display = (groupB.committedText || groupB.partialText) ? 'none' : 'flex';
    }
    if (groupB.multilingualAutoScroll && dom.multilingualScroll) {
      dom.multilingualScroll.scrollTop = dom.multilingualScroll.scrollHeight;
    }
  }
}

/**
 * Update stats display
 */
function updateStats() {
  // Latency
  if (latestLatency !== null) {
    dom.statLatency.textContent = `${latestLatency.toFixed(0)}ms`;
  } else {
    dom.statLatency.textContent = '-';
  }

  // Duration
  if (recordingStartTime) {
    const elapsed = (Date.now() - recordingStartTime) / 1000;
    dom.statDuration.textContent = `${elapsed.toFixed(1)}s`;
  } else {
    dom.statDuration.textContent = '-';
  }

  // Buffer fill
  if (bufferFillPct !== null) {
    dom.statBuffer.textContent = `${bufferFillPct.toFixed(0)}%`;
  } else {
    dom.statBuffer.textContent = '-';
  }
}

/**
 * Show error message
 */
function showError(message) {
  dom.errorMsg.textContent = message;
  dom.errorMsg.classList.remove('hidden');
  dom.errorMsg.classList.add('show');
}

/**
 * Clear error message
 */
function clearError() {
  dom.errorMsg.textContent = '';
  dom.errorMsg.classList.add('hidden');
  dom.errorMsg.classList.remove('show');
}

/**
 * Handle manual summarize button click
 */
async function handleSummarizeClick() {
  const text = committedText + partialText;
  console.log('[summarize] button clicked, text length:', text.length, 'ws state:', ws?.readyState);
  if (!text.trim()) {
    showError('No transcript to summarize');
    setTimeout(clearError, 2000);
    return;
  }
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    showError('Not connected — connect to server first');
    setTimeout(clearError, 2000);
    return;
  }
  requestSummarization(text);
}

/**
 * Request summarization via WebSocket
 */
function requestSummarization(text) {
  const currentGroup = currentTabGroup === 'groupA' ? groupA : groupB;
  if (!currentGroup) {
    console.log('[summarize] skipped — invalid group');
    return;
  }

  if (currentGroup.summarizationInFlight) {
    console.log('[summarize] skipped — already in flight');
    return;
  }
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.log('[summarize] skipped — WebSocket not open');
    return;
  }

  currentGroup.summarizationInFlight = true;
  const wordCount = text.split(/\s+/).filter(Boolean).length;
  console.log('[summarize] sending request, wordCount:', wordCount);

  ws.send(JSON.stringify({
    type: 'summarize',
    text: text,
    word_count: wordCount,
  }));
}

/**
 * Handle summary response from server
 */
function handleSummaryResult(data) {
  if (currentTabGroup === 'groupA') {
    groupA.summarizationInFlight = false;
  } else if (currentTabGroup === 'groupB') {
    groupB.summarizationInFlight = false;
  } else {
    return;
  }

  const summary = data.summary || '';
  const koreanSummary = data.korean_summary || '';
  const wordCount = data.word_count || 0;
  console.log('[summarize] received summary, length:', summary.length, 'wordCount:', wordCount);

  if (summary) {
    insertSummaryBlock(summary, koreanSummary, wordCount);
    if (currentTabGroup === 'groupA') {
      groupA.lastSummarizedWordCount = wordCount;
    } else if (currentTabGroup === 'groupB') {
      groupB.lastSummarizedWordCount = wordCount;
    }
  } else {
    console.warn('[summarize] empty summary received');
  }
}

/**
 * Display summary in the footer of transcript and translate panels
 */
function insertSummaryBlock(summary, koreanSummary, wordCount) {
  // Transcript tab: English summary
  dom.summaryContentTranscript.textContent = summary;
  dom.summaryContentTranscript.classList.remove('hidden');
  dom.summaryFooterTranscript.classList.remove('hidden');

  // Interpreter tab: Korean summary (fallback to English if translation failed)
  dom.summaryContentTranslate.textContent = koreanSummary || summary;
  dom.summaryContentTranslate.classList.remove('hidden');
  dom.summaryFooterTranslate.classList.remove('hidden');
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Add a committed sentence row to the interpreter panel
 */
function addInterpreterRow(source, translation) {
  // Remove any existing pending row
  const pending = dom.interpreterRows.querySelector('.interpreter-pending');
  if (pending) pending.remove();

  interpreterSentenceCount++;

  const row = document.createElement('div');
  row.className = 'interpreter-row';
  row.innerHTML = `
    <div class="interp-col-en">
      <span class="interpreter-src">${escapeHtml(source)}</span>
    </div>
    <div class="interp-col-ko">
      <span class="interpreter-tl">${escapeHtml(translation)}</span>
    </div>`;

  dom.interpreterRows.appendChild(row);

  if (dom.interpreterPlaceholder) dom.interpreterPlaceholder.style.display = 'none';

  interpreterScrollToBottom();
}

/**
 * Show buffered-but-not-yet-sentence-end content as an intermediate row
 */
function updateInterpreterBuffering(source, translation) {
  let pending = dom.interpreterRows.querySelector('.interpreter-pending');

  if (!pending) {
    pending = document.createElement('div');
    pending.className = 'interpreter-row interpreter-pending';
    dom.interpreterRows.appendChild(pending);
  }

  pending.innerHTML = `
    <div class="interp-col-en">
      <span class="interp-text-dim">${escapeHtml(source)}</span>
    </div>
    <div class="interp-col-ko">
      <span class="interp-text-ko-dim">${escapeHtml(translation)}</span>
    </div>`;

  if (dom.interpreterPlaceholder) dom.interpreterPlaceholder.style.display = 'none';

  interpreterScrollToBottom();
}

/**
 * Update the pending (in-progress) row in the interpreter panel
 */
function updateInterpreterPending(source, translation) {
  let pending = dom.interpreterRows.querySelector('.interpreter-pending');

  if (!source && !translation) {
    if (pending) pending.remove();
    return;
  }

  if (!pending) {
    pending = document.createElement('div');
    pending.className = 'interpreter-row interpreter-pending';
    dom.interpreterRows.appendChild(pending);
  }

  pending.innerHTML = `
    <div class="interp-col-en">
      <span class="interp-text-partial">${escapeHtml(source)}</span>
    </div>
    <div class="interp-col-ko">
      <span class="interp-text-ko-partial">${escapeHtml(translation)}</span>
    </div>`;

  if (dom.interpreterPlaceholder) dom.interpreterPlaceholder.style.display = 'none';

  interpreterScrollToBottom();
}

// Start stats update loop (every 100ms)
setInterval(() => {
  if (currentState === State.RECORDING) {
    updateStats();
  }
}, 100);

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

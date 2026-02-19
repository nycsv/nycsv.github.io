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

// Transcription state
let committedText = '';
let partialText = '';

// Translation state
let translationCommitted = '';
let translationPartial = '';

// Interpreter state
let interpreterSentenceCount = 0;
let interpreterBuffer = { source: '', text: '' };

// Summarization state
let lastSummarizedWordCount = 0;
let summarizationInFlight = false;

// Audio gating: only send audio after server ack
let readyToSendAudio = false;
let ackResolver = null;

// Stats
let recordingStartTime = null;
let latestLatency = null;
let bufferFillPct = null;

// DOM elements
let dom = {};

// i18n strings (injected by template)
let i18n = {};

// Interpreter auto-scroll state
let interpreterAutoScroll = true;

/**
 * Initialize application
 */
function init() {
  // Get i18n strings from global (set by template)
  i18n = window.ASR_I18N || {};

  // Cache DOM elements
  dom = {
    connectBtn: document.getElementById('connect-btn'),
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
    sessionMarker: document.getElementById('session-marker'),
    sessionMarkerTime: document.getElementById('session-marker-time'),
    errorMsg: document.getElementById('error-message'),
    statLatency: document.getElementById('stat-latency'),
    statDuration: document.getElementById('stat-duration'),
    statStatus: document.getElementById('stat-status'),
    statBuffer: document.getElementById('stat-buffer'),
    statusFooterDot: document.getElementById('status-footer-dot'),
    audioSource: document.getElementById('audio-source'),
    copyBtn: document.getElementById('copy-btn'),
    summarizeBtn: document.getElementById('summarize-btn'),
    summarizeBtnTranslate: document.getElementById('summarize-btn-translate'),
    summarizeBtnInterpreter: document.getElementById('summarize-btn-interpreter'),
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
    interpreterPlaceholder: document.getElementById('interpreter-placeholder'),
    interpreterPinBtn: document.getElementById('interpreter-pin-btn'),
    interpreterJumpBtn: document.getElementById('interpreter-jump-btn'),
    interpreterCountBadge: document.getElementById('interpreter-count-badge'),
    interpreterCopyBtn: document.getElementById('interpreter-copy-btn'),
    tabInterview: document.getElementById('tab-interview'),
    interviewBox: document.getElementById('interview-box'),
    tabInterview2: document.getElementById('tab-interview2'),
    interview2Box: document.getElementById('interview2-box'),
    audioTrackIndicators: document.getElementById('audio-track-indicators'),
    indicatorMic: document.getElementById('indicator-mic'),
    indicatorMicLabel: document.getElementById('indicator-mic-label'),
    indicatorSys: document.getElementById('indicator-sys'),
    indicatorSysLabel: document.getElementById('indicator-sys-label'),
  };

  // Attach event listeners
  dom.connectBtn.addEventListener('click', handleConnect);
  dom.micBtn.addEventListener('click', handleMicClick);
  dom.copyBtn.addEventListener('click', handleCopyClick);
  dom.summarizeBtn.addEventListener('click', handleSummarizeClick);
  dom.summarizeBtnTranslate.addEventListener('click', handleSummarizeClick);
  dom.summarizeBtnInterpreter.addEventListener('click', handleSummarizeClick);
  dom.copyBtnEn.addEventListener('click', () => handleTranslateCopy(dom.copyBtnEn, () => committedText + partialText));
  dom.copyBtnKo.addEventListener('click', () => handleTranslateCopy(dom.copyBtnKo, () => translationCommitted + translationPartial));
  dom.tabLive.addEventListener('click', () => setActiveTab('live'));
  dom.tabTranslate.addEventListener('click', () => setActiveTab('translate'));
  dom.tabInterpreter.addEventListener('click', () => setActiveTab('interpreter'));
  dom.tabInterview.addEventListener('click', () => setActiveTab('interview'));
  dom.tabInterview2.addEventListener('click', () => setActiveTab('interview2'));

  // Audio source toggle buttons
  initAudioSourceToggle();

  // Interpreter scroll & controls
  initInterpreterControls();

  // Initial state
  setActiveTab('live');
  updateUI();
}

/**
 * Initialize audio source toggle buttons
 */
function initAudioSourceToggle() {
  const btns = document.querySelectorAll('.audio-src-btn');
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
  // Pin/unpin auto-scroll
  if (dom.interpreterPinBtn) {
    dom.interpreterPinBtn.addEventListener('click', () => {
      interpreterAutoScroll = !interpreterAutoScroll;
      updateInterpreterPinUI();
      if (interpreterAutoScroll) {
        dom.interpreterScroll.scrollTop = dom.interpreterScroll.scrollHeight;
      }
    });
  }

  // Jump to bottom button
  if (dom.interpreterJumpBtn) {
    dom.interpreterJumpBtn.addEventListener('click', () => {
      dom.interpreterScroll.scrollTop = dom.interpreterScroll.scrollHeight;
      interpreterAutoScroll = true;
      updateInterpreterPinUI();
    });
  }

  // Detect manual scroll → disable auto-scroll, show jump button
  if (dom.interpreterScroll) {
    dom.interpreterScroll.addEventListener('scroll', () => {
      const el = dom.interpreterScroll;
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
      if (!atBottom && interpreterAutoScroll) {
        interpreterAutoScroll = false;
        updateInterpreterPinUI();
      }
      if (atBottom && !interpreterAutoScroll) {
        interpreterAutoScroll = true;
        updateInterpreterPinUI();
      }
      // Show/hide jump button
      if (dom.interpreterJumpBtn) {
        dom.interpreterJumpBtn.classList.toggle('hidden', atBottom);
        if (!atBottom) dom.interpreterJumpBtn.style.display = 'flex';
      }
    });
  }

  // Copy all interpreter sentences
  if (dom.interpreterCopyBtn) {
    dom.interpreterCopyBtn.addEventListener('click', async () => {
      const rows = dom.interpreterRows.querySelectorAll('.interpreter-row:not(.interpreter-pending)');
      if (!rows.length) {
        showError('No sentences to copy');
        setTimeout(clearError, 2000);
        return;
      }
      const lines = [];
      rows.forEach((row, i) => {
        const src = row.querySelector('.interpreter-src')?.textContent || '';
        const tl = row.querySelector('.interpreter-tl')?.textContent || '';
        lines.push(`${i + 1}. ${src}\n   ${tl}`);
      });
      try {
        await navigator.clipboard.writeText(lines.join('\n\n'));
        const icon = dom.interpreterCopyBtn.querySelector('.material-symbols-outlined');
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
 * Update interpreter pin button UI
 */
function updateInterpreterPinUI() {
  if (!dom.interpreterPinBtn) return;
  if (interpreterAutoScroll) {
    dom.interpreterPinBtn.classList.remove('interpreter-pin-inactive');
    dom.interpreterPinBtn.classList.add('interpreter-pin-active', 'bg-primary/10', 'text-primary', 'border-primary/20');
    dom.interpreterPinBtn.classList.remove('bg-transparent', 'text-slate-500', 'border-transparent');
  } else {
    dom.interpreterPinBtn.classList.add('interpreter-pin-inactive');
    dom.interpreterPinBtn.classList.remove('interpreter-pin-active', 'bg-primary/10', 'text-primary', 'border-primary/20');
  }
}

/**
 * Scroll interpreter to bottom if auto-scroll is on
 */
function interpreterScrollToBottom() {
  if (interpreterAutoScroll && dom.interpreterScroll) {
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
 * Update interpreter sentence count badge
 */
function updateInterpreterCountBadge() {
  if (dom.interpreterCountBadge) {
    if (interpreterSentenceCount > 0) {
      dom.interpreterCountBadge.textContent = interpreterSentenceCount;
      dom.interpreterCountBadge.classList.remove('hidden');
    } else {
      dom.interpreterCountBadge.classList.add('hidden');
    }
  }
}

/**
 * Tab switching
 */
function setActiveTab(tab) {
  dom.transcriptPanel.classList.toggle('hidden', tab !== 'live');
  dom.translateBox.classList.toggle('hidden', tab !== 'translate');
  dom.interpreterBox.classList.toggle('hidden', tab !== 'interpreter');
  dom.interviewBox.classList.toggle('hidden', tab !== 'interview');
  dom.interview2Box.classList.toggle('hidden', tab !== 'interview2');

  updateTabButton(dom.tabLive, tab === 'live');
  updateTabButton(dom.tabTranslate, tab === 'translate');
  updateTabButton(dom.tabInterpreter, tab === 'interpreter');
  updateTabButton(dom.tabInterview, tab === 'interview');
  updateTabButton(dom.tabInterview2, tab === 'interview2');
}

function updateTabButton(button, isActive) {
  button.classList.toggle('tab-active', isActive);
  button.classList.toggle('text-primary', isActive);
  button.classList.toggle('text-slate-500', !isActive);
}

/**
 * Handle connect button click
 */
async function handleConnect() {
  if (currentState === State.IDLE) {
    await connectWebSocket(SERVER_URL);
  } else if (currentState === State.CONNECTED) {
    disconnectWebSocket();
  }
}

/**
 * Connect to WebSocket server
 */
async function connectWebSocket(url) {
  setState(State.CONNECTING);
  clearError();

  try {
    ws = new WebSocket(url);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

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
  } catch (error) {
    console.error('Failed to connect:', error);
    showError(i18n.errorConnection || 'Failed to connect');
    setState(State.IDLE);
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
  committedText += newText;
  partialText = '';  // Clear partial — committed text replaces it
  updateTranscript();

  // Auto-summarize every ~200 words
  const wordCount = committedText.split(/\s+/).filter(Boolean).length;
  if (!summarizationInFlight && wordCount - lastSummarizedWordCount >= 200) {
    requestSummarization(committedText);
  }
}

/**
 * Handle committed translation (finalized sentence translation)
 */
function handleCommittedTranslation(data) {
  const tl = data.text || '';
  const src = data.source || '';
  translationCommitted += tl + ' ';
  translationPartial = '';
  updateTranscript();

  // Interpreter tab: accumulate into buffer, flush to card only on sentence_end
  interpreterBuffer.source += (interpreterBuffer.source ? ' ' : '') + src;
  interpreterBuffer.text += (interpreterBuffer.text ? ' ' : '') + tl;

  if (data.sentence_end) {
    addInterpreterRow(interpreterBuffer.source, interpreterBuffer.text);
    interpreterBuffer = { source: '', text: '' };
  } else {
    updateInterpreterBuffering(interpreterBuffer.source, interpreterBuffer.text);
  }
}

/**
 * Handle partial translation (in-progress preview)
 */
function handlePartialTranslation(data) {
  translationPartial = data.translation || '';
  const partialSrc = data.source || '';
  updateTranscript();

  // Interpreter tab: pending row shows buffer + current partial
  const combinedSrc = [interpreterBuffer.source, partialSrc].filter(Boolean).join(' ');
  const combinedTl = [interpreterBuffer.text, translationPartial].filter(Boolean).join(' ');
  updateInterpreterPending(combinedSrc, combinedTl);
}

/**
 * Handle final_translation (full accumulated translation at end of session)
 */
function handleFinalTranslation(data) {
  translationCommitted = data.translation || '';
  translationPartial = '';
  updateTranscript();
}

/**
 * Handle partial text from server
 */
function handlePartialText(data) {
  partialText = data.text || '';

  if (data.translation !== undefined) {
    translationPartial = data.translation;
  }

  // Update stats
  if (data.triton_call_ms !== undefined) {
    latestLatency = data.triton_call_ms;
  }
  if (data.buffer_fill_pct !== undefined) {
    bufferFillPct = data.buffer_fill_pct;
  }

  updateTranscript();
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
 * Handle final result from server
 */
function handleFinalResult(data) {
  const finalText = data.transcription || '';
  committedText = finalText;
  partialText = '';
  translationPartial = '';
  updateTranscript();
  setState(State.CONNECTED);
  cleanup();
}

/**
 * Handle mic button click
 */
async function handleMicClick() {
  if (currentState === State.CONNECTED) {
    await startRecording();
  } else if (currentState === State.RECORDING) {
    await stopRecording();
  }
}

/**
 * Handle copy button click
 */
async function handleCopyClick() {
  const fullText = committedText + partialText;

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

    // Reset transcription and translation
    committedText = '';
    partialText = '';
    translationCommitted = '';
    translationPartial = '';
    lastSummarizedWordCount = 0;
    summarizationInFlight = false;
    updateTranscript();

    // Reset interpreter
    interpreterSentenceCount = 0;
    interpreterAutoScroll = true;
    interpreterBuffer = { source: '', text: '' };
    if (dom.interpreterRows) dom.interpreterRows.innerHTML = '';
    if (dom.interpreterPlaceholder) dom.interpreterPlaceholder.style.display = 'flex';
    updateInterpreterCountBadge();
    updateInterpreterPinUI();

    // Show session start marker
    if (dom.sessionMarker) {
      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      if (dom.sessionMarkerTime) dom.sessionMarkerTime.textContent = `Session Started at ${timeStr}`;
      dom.sessionMarker.classList.remove('hidden');
      dom.sessionMarker.classList.add('flex');
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
      dom.statusText.textContent = i18n.statusRecording || 'Recording';
      break;
    case State.STOPPING:
      dom.statusDot.classList.add('connecting');
      dom.statusText.textContent = i18n.statusStopping || 'Stopping...';
      break;
  }

  // Connect button
  const btnTextSpan = dom.connectBtn.querySelector('span:last-child');
  if (currentState === State.IDLE || currentState === State.CONNECTING) {
    dom.connectBtn.querySelector('.material-symbols-outlined').textContent = 'power_settings_new';
    if (btnTextSpan) btnTextSpan.textContent = i18n.btnConnect || 'Connect';
    dom.connectBtn.disabled = currentState === State.CONNECTING;
    dom.connectBtn.classList.remove('btn-danger');
    dom.connectBtn.classList.add('btn-primary');
  } else {
    dom.connectBtn.querySelector('.material-symbols-outlined').textContent = 'power_settings_new';
    if (btnTextSpan) btnTextSpan.textContent = i18n.btnDisconnect || 'Disconnect';
    dom.connectBtn.disabled = currentState === State.RECORDING || currentState === State.STOPPING;
    dom.connectBtn.classList.remove('btn-primary');
    dom.connectBtn.classList.add('btn-danger');
  }

  // Mic button
  const micEnabled = currentState === State.CONNECTED || currentState === State.RECORDING;
  dom.micBtn.disabled = !micEnabled;

  if (currentState === State.RECORDING) {
    dom.micBtn.classList.add('recording');
    dom.micIcon.textContent = 'stop';
    dom.micLabel.textContent = i18n.micLabelStop || 'Click to stop';
    dom.micReadyDot.className = 'w-2 h-2 rounded-full bg-red-500 animate-pulse';
    dom.micReadyText.textContent = 'Recording';
    dom.waveform.classList.add('waveform-active');
  } else {
    dom.micBtn.classList.remove('recording');
    dom.micIcon.textContent = 'mic';
    dom.waveform.classList.remove('waveform-active');
    if (currentState === State.CONNECTED) {
      dom.micLabel.textContent = i18n.micLabelStart || 'Click to start transcribing';
      dom.micReadyDot.className = 'w-2 h-2 rounded-full bg-green-500 animate-pulse';
      dom.micReadyText.textContent = 'Microphone Ready';
    } else {
      dom.micLabel.textContent = i18n.micLabelConnect || 'Connect first';
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

  // Status text in footer
  dom.statStatus.textContent = currentState;
}

/**
 * Update transcript display
 */
function updateTranscript() {
  dom.committedSpan.textContent = committedText;
  dom.partialSpan.textContent = partialText;

  // Toggle placeholder
  if (dom.placeholder) {
    dom.placeholder.style.display = (committedText || partialText) ? 'none' : 'flex';
  }

  // Auto-scroll to bottom
  dom.transcriptBox.scrollTop = dom.transcriptBox.scrollHeight;

  // Update translate tab mirrors
  dom.translateCommittedEn.textContent = committedText;
  dom.translatePartialEn.textContent = partialText;
  dom.translateCommittedKo.textContent = translationCommitted;
  dom.translatePartialKo.textContent = translationPartial;

  // Auto-scroll translate panels
  dom.translateScrollEn.scrollTop = dom.translateScrollEn.scrollHeight;
  dom.translateScrollKo.scrollTop = dom.translateScrollKo.scrollHeight;
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

  // Status
  dom.statStatus.textContent = currentState;

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
  if (summarizationInFlight) {
    console.log('[summarize] skipped — already in flight');
    return;
  }
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.log('[summarize] skipped — WebSocket not open');
    return;
  }

  summarizationInFlight = true;
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
  summarizationInFlight = false;
  const summary = data.summary || '';
  const wordCount = data.word_count || 0;
  console.log('[summarize] received summary, length:', summary.length, 'wordCount:', wordCount);

  if (summary) {
    insertSummaryBlock(summary, wordCount);
    lastSummarizedWordCount = wordCount;
  } else {
    console.warn('[summarize] empty summary received');
  }
}

/**
 * Insert a summary block at the bottom of the transcript and translate panels,
 * replacing any previously generated summary block.
 */
function insertSummaryBlock(summary, wordCount) {
  const label = `Summary at ${wordCount} words`;

  function makeBlock() {
    const block = document.createElement('div');
    block.className = 'summary-block';
    block.innerHTML = `
      <div class="summary-block-label">${escapeHtml(label)}</div>
      <div class="summary-block-text">${escapeHtml(summary)}</div>`;
    return block;
  }

  function replaceSummary(container, scrollEl) {
    const existing = container.querySelector('.summary-block');
    if (existing) existing.remove();
    container.appendChild(makeBlock());
    scrollEl.scrollTop = scrollEl.scrollHeight;
  }

  // Replace summary in Live Transcript tab
  replaceSummary(dom.transcriptContent, dom.transcriptBox);

  // Replace summary in Live Translate tab (English side only — no summary on Korean side)
  replaceSummary(dom.translateContentEn, dom.translateScrollEn);
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
      <span class="interpreter-row-num">${interpreterSentenceCount}</span><span class="interpreter-src">${escapeHtml(source)}</span>
    </div>
    <div class="interp-col-ko">
      <span class="interpreter-tl">${escapeHtml(translation)}</span>
    </div>`;

  dom.interpreterRows.appendChild(row);

  if (dom.interpreterPlaceholder) dom.interpreterPlaceholder.style.display = 'none';

  updateInterpreterCountBadge();
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
      <span class="interpreter-row-num interpreter-row-num--pending">…</span><span class="interp-text-dim">${escapeHtml(source)}</span>
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
      <span class="interpreter-row-num interpreter-row-num--pending">~</span><span class="interp-text-partial">${escapeHtml(source)}</span>
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

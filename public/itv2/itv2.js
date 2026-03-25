(function () {
  'use strict';

  // ──────────────────────────────────────────────
  // Constants
  // ──────────────────────────────────────────────
  const DEFAULT_SERVER_URL = 'https://localhost';
  const LS_SERVER_URL = 'itv2_server_url';

  // ──────────────────────────────────────────────
  // DOM refs (populated on init)
  // ──────────────────────────────────────────────
  let el = {};

  // ──────────────────────────────────────────────
  // State
  // ──────────────────────────────────────────────
  let analyzing = false;

  // ──────────────────────────────────────────────
  // Init (called after DOM is ready)
  // ──────────────────────────────────────────────
  function init() {
    el = {
      serverUrl:       document.getElementById('itv2-server-url'),
      analyzeBtn:      document.getElementById('itv2-analyze-btn'),
      analyzeIcon:     document.getElementById('itv2-analyze-icon'),
      analyzeLabel:    document.getElementById('itv2-analyze-label'),
      clearBtn:        document.getElementById('itv2-clear-btn'),
      placeholder:     document.getElementById('itv2-placeholder'),
      thinking:        document.getElementById('itv2-thinking'),
      responseContent: document.getElementById('itv2-response-content'),
      transcriptText:  document.getElementById('itv2-transcript-text'),
      transcriptPart:  document.getElementById('itv2-transcript-partial'),
      transcriptPh:    document.getElementById('itv2-transcript-placeholder'),
      transcriptScroll:document.getElementById('itv2-transcript-scroll'),
    };

    // Restore saved server URL
    const saved = localStorage.getItem(LS_SERVER_URL);
    if (saved) el.serverUrl.value = saved;
    else el.serverUrl.value = DEFAULT_SERVER_URL;

    // Save URL on change
    el.serverUrl.addEventListener('input', () => {
      localStorage.setItem(LS_SERVER_URL, el.serverUrl.value.trim());
    });

    el.analyzeBtn.addEventListener('click', handleAnalyze);
    el.clearBtn.addEventListener('click', clearResults);

    startTranscriptMirror();
  }

  // ──────────────────────────────────────────────
  // Transcript mirroring
  // ──────────────────────────────────────────────
  function startTranscriptMirror() {
    const committed = document.getElementById('transcript-committed');
    const partial   = document.getElementById('transcript-partial');

    if (!committed || !partial) return;

    const observer = new MutationObserver(() => {
      syncTranscript(committed.textContent, partial.textContent);
    });

    observer.observe(committed, { characterData: true, childList: true, subtree: true });
    observer.observe(partial,   { characterData: true, childList: true, subtree: true });

    // Initial sync
    syncTranscript(committed.textContent, partial.textContent);
  }

  function syncTranscript(committed, partial) {
    const hasContent = committed.trim() || partial.trim();

    el.transcriptText.textContent = committed;
    el.transcriptPart.textContent = partial;
    el.transcriptPh.classList.toggle('hidden', !!hasContent);

    // Auto-scroll to bottom
    el.transcriptScroll.scrollTop = el.transcriptScroll.scrollHeight;
  }

  // ──────────────────────────────────────────────
  // Analyze handler
  // ──────────────────────────────────────────────
  async function handleAnalyze() {
    if (analyzing) return;

    const serverUrl = (el.serverUrl.value || '').trim();
    if (!serverUrl) {
      showError('Please enter the local server URL.');
      return;
    }

    const transcript = getFullTranscript();
    if (!transcript.trim()) {
      showError('No transcript yet. Start recording and describe the problem first.');
      return;
    }

    setAnalyzing(true);
    showThinking();

    try {
      const result = await callLocalServer(serverUrl, transcript);
      renderResult(result);
    } catch (err) {
      showError(err.message || 'Unknown error calling local server.');
    } finally {
      setAnalyzing(false);
    }
  }

  function getFullTranscript() {
    const committed = document.getElementById('transcript-committed');
    const partial   = document.getElementById('transcript-partial');
    return ((committed ? committed.textContent : '') + ' ' +
            (partial   ? partial.textContent   : '')).trim();
  }

  async function callLocalServer(serverUrl, transcript) {
    const url = serverUrl.replace(/\/+$/, '') + '/interview';

    const body = {
      transcript: transcript,
      max_new_tokens: 4096,
      temperature: 0.7,
    };

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      let errMsg = `Server error ${response.status}`;
      try {
        const errBody = await response.json();
        errMsg = errBody?.detail || errMsg;
      } catch (_) {}
      throw new Error(errMsg);
    }

    return await response.json();
  }

  // ──────────────────────────────────────────────
  // UI state helpers
  // ──────────────────────────────────────────────
  function setAnalyzing(on) {
    analyzing = on;
    el.analyzeBtn.disabled = on;
    el.analyzeIcon.textContent  = on ? 'hourglass_top' : 'psychology';
    el.analyzeLabel.textContent = on ? 'Analyzing...'  : 'Analyze';
  }

  function showThinking() {
    el.placeholder.classList.add('hidden');
    el.thinking.classList.remove('hidden');
    el.responseContent.classList.add('hidden');
  }

  function clearResults() {
    el.responseContent.innerHTML = '';
    el.responseContent.classList.add('hidden');
    el.thinking.classList.add('hidden');
    el.placeholder.classList.remove('hidden');
  }

  // ──────────────────────────────────────────────
  // Error display
  // ──────────────────────────────────────────────
  function showError(msg) {
    el.thinking.classList.add('hidden');
    el.placeholder.classList.add('hidden');
    el.responseContent.innerHTML = `<div class="itv-error">${escapeHtml(msg)}</div>`;
    el.responseContent.classList.remove('hidden');
  }

  function renderResult(r) {
    el.thinking.classList.add('hidden');
    el.placeholder.classList.add('hidden');

    const badgeClass = getBadgeClass(r.problem_type);

    const html = `
      <!-- Problem Statement -->
      <div class="itv-section">
        <div class="itv-section-header">
          <span class="material-symbols-outlined itv-section-icon text-primary">assignment</span>
          <span class="itv-section-title">Problem</span>
        </div>
        <div class="itv-section-body">
          <div class="itv-problem-type-badge ${badgeClass}">
            ${escapeHtml(r.problem_type || 'Unknown')}
          </div>
          <p>${escapeHtml(r.problem_statement || '')}</p>
        </div>
      </div>

      <!-- Approach -->
      <div class="itv-section">
        <div class="itv-section-header">
          <span class="material-symbols-outlined itv-section-icon text-amber-400">lightbulb</span>
          <span class="itv-section-title">Approach</span>
        </div>
        <div class="itv-section-body itv-approach-text">
          ${formatApproach(r.approach || '')}
        </div>
      </div>

      <!-- Solution Code -->
      <div class="itv-section">
        <div class="itv-section-header">
          <span class="material-symbols-outlined itv-section-icon text-emerald-400">code</span>
          <span class="itv-section-title">Solution — ${escapeHtml(r.language || 'python')}</span>
        </div>
        <div class="itv-section-body" style="padding:0;">
          ${renderCodeBlock(r.solution_code || '', r.language || 'python')}
        </div>
      </div>

      <!-- Complexity -->
      <div class="itv-section">
        <div class="itv-section-header">
          <span class="material-symbols-outlined itv-section-icon text-cyan-400">query_stats</span>
          <span class="itv-section-title">Complexity</span>
        </div>
        <div class="itv-section-body">
          <div class="itv-complexity-row">
            <div class="itv-complexity-item">
              <span class="itv-complexity-label">Time</span>
              <span class="itv-complexity-value">${escapeHtml(r.time_complexity || '—')}</span>
            </div>
            <div class="itv-complexity-item">
              <span class="itv-complexity-label">Space</span>
              <span class="itv-complexity-value">${escapeHtml(r.space_complexity || '—')}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Key Insights -->
      ${renderList(r.key_insights, 'Key Insights', 'tips_and_updates', 'text-violet-400', 'itv-insights-list')}

      <!-- Follow-up Questions -->
      ${renderList(r.follow_ups, 'Follow-up Questions', 'quiz', 'text-slate-400', 'itv-followup-list')}
    `;

    el.responseContent.innerHTML = html;
    el.responseContent.classList.remove('hidden');

    // Wire up copy buttons for code blocks
    el.responseContent.querySelectorAll('.itv-code-copy-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const pre = btn.closest('.itv-code-block').querySelector('pre');
        navigator.clipboard.writeText(pre.textContent).then(() => {
          btn.querySelector('span.label').textContent = 'Copied!';
          setTimeout(() => { btn.querySelector('span.label').textContent = 'Copy'; }, 1500);
        });
      });
    });

    // Scroll response to top
    el.responseContent.parentElement.scrollTop = 0;
  }

  function renderCodeBlock(code, lang) {
    const highlighted = syntaxHighlight(code, lang);
    return `
      <div class="itv-code-block">
        <div class="itv-code-header">
          <span class="itv-code-lang">${escapeHtml(lang)}</span>
          <button class="itv-code-copy-btn">
            <span class="material-symbols-outlined">content_copy</span>
            <span class="label">Copy</span>
          </button>
        </div>
        <pre>${highlighted}</pre>
      </div>
    `;
  }

  function renderList(items, title, icon, iconColor, listClass) {
    if (!Array.isArray(items) || items.length === 0) return '';
    const lis = items.map(item => `<li>${escapeHtml(item)}</li>`).join('');
    return `
      <div class="itv-section">
        <div class="itv-section-header">
          <span class="material-symbols-outlined itv-section-icon ${iconColor}">${icon}</span>
          <span class="itv-section-title">${title}</span>
        </div>
        <div class="itv-section-body">
          <ul class="${listClass}">${lis}</ul>
        </div>
      </div>
    `;
  }

  const LANG_KEYWORDS = {
    python: {
      kw: new Set('def,return,if,elif,else,for,while,in,not,and,or,import,from,class,pass,break,continue,yield,lambda,with,as,try,except,finally,raise,None,True,False,self,is,del,global,nonlocal,assert'.split(',')),
      bi: new Set('print,len,range,int,str,float,list,dict,set,tuple,bool,type,enumerate,zip,map,filter,sorted,min,max,sum,abs,round,open,any,all,isinstance,hasattr,getattr,setattr,append,extend,pop,insert,remove,keys,values,items,get,update'.split(',')),
    },
    javascript: {
      kw: new Set('function,return,if,else,for,while,let,const,var,new,this,class,extends,import,export,from,default,async,await,try,catch,finally,throw,typeof,instanceof,in,of,null,undefined,true,false,break,continue'.split(',')),
    },
    java: {
      kw: new Set('public,private,protected,static,void,int,long,double,float,boolean,char,byte,short,class,interface,extends,implements,new,return,if,else,for,while,do,try,catch,finally,throw,throws,import,package,this,super,null,true,false,break,continue,final,abstract,enum'.split(',')),
    },
    cpp: {
      kw: new Set('int,long,double,float,bool,char,void,auto,const,static,return,if,else,for,while,do,try,catch,throw,class,struct,namespace,using,new,delete,nullptr,true,false,break,continue,public,private,protected,virtual,override,template,typename'.split(',')),
    },
  };

  const TOKEN_RE = /"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\/\*[\s\S]*?\*\/|\/\/[^\n]*|#[^\n]*|@\w+|\b[a-zA-Z_]\w*\b|\b\d+(?:\.\d+)?\b|[^\s]/g;

  function syntaxHighlight(code, lang) {
    const langDef = LANG_KEYWORDS[lang] || {};
    const kwSet = langDef.kw || new Set();
    const biSet = langDef.bi || new Set();

    const parts = [];
    let match;
    let lastIndex = 0;

    TOKEN_RE.lastIndex = 0;

    while ((match = TOKEN_RE.exec(code)) !== null) {
      if (match.index > lastIndex) {
        parts.push(escapeHtml(code.slice(lastIndex, match.index)));
      }
      lastIndex = TOKEN_RE.lastIndex;

      const tok = match[0];
      const first = tok[0];
      const first2 = tok.slice(0, 2);
      const first3 = tok.slice(0, 3);

      if (first3 === '"""' || first3 === "'''") {
        parts.push('<span class="sh-keyword">' + escapeHtml(tok) + '</span>');
      } else if ((first === '"' || first === "'") && tok.length > 1) {
        parts.push('<span class="sh-string">' + escapeHtml(tok) + '</span>');
      } else if (first2 === '/*') {
        parts.push('<span class="sh-comment">' + escapeHtml(tok) + '</span>');
      } else if (first2 === '//' || first === '#') {
        parts.push('<span class="sh-comment">' + escapeHtml(tok) + '</span>');
      } else if (first === '@') {
        parts.push('<span class="sh-decorator">' + escapeHtml(tok) + '</span>');
      } else if (/^[a-zA-Z_]/.test(first)) {
        if (kwSet.has(tok)) {
          parts.push('<span class="sh-keyword">' + escapeHtml(tok) + '</span>');
        } else if (biSet.has(tok)) {
          parts.push('<span class="sh-builtin">' + escapeHtml(tok) + '</span>');
        } else {
          parts.push(escapeHtml(tok));
        }
      } else if (/^\d/.test(first)) {
        parts.push('<span class="sh-number">' + escapeHtml(tok) + '</span>');
      } else {
        parts.push(escapeHtml(tok));
      }
    }

    if (lastIndex < code.length) {
      parts.push(escapeHtml(code.slice(lastIndex)));
    }

    return parts.join('');
  }

  // ──────────────────────────────────────────────
  // Helpers
  // ──────────────────────────────────────────────
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&#34;')
      .replace(/'/g, '&#39;');
  }

  function formatApproach(text) {
    return text.split(/\n\n+/)
      .map(p => `<p>${escapeHtml(p.trim())}</p>`)
      .join('');
  }

  function getBadgeClass(type) {
    if (!type) return 'itv-badge-other';
    const t = type.toLowerCase();
    if (t.includes('ml') || t.includes('machine')) return 'itv-badge-ml';
    if (t.includes('system'))                        return 'itv-badge-system';
    if (t.includes('algo') || t.includes('data'))   return 'itv-badge-algo';
    return 'itv-badge-other';
  }

  // ──────────────────────────────────────────────
  // Boot
  // ──────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();

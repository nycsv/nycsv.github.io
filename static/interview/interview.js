/**
 * Interview Assistant Module
 *
 * Reads the live ASR transcript and uses the Claude API to analyze coding /
 * ML-design interview problems, then renders structured solutions in the
 * right-hand panel.
 */

(function () {
  'use strict';

  // ──────────────────────────────────────────────
  // Constants
  // ──────────────────────────────────────────────
  const CLAUDE_API_URL = 'https://api.anthropic.com/v1/messages';
  const ANTHROPIC_VERSION = '2023-06-01';
  const LS_API_KEY = 'interview_anthropic_api_key';

  const SYSTEM_PROMPT = `You are an expert coding interview assistant specializing in algorithms, data structures, and ML system design. \
When given a transcript of an interview question, extract and solve it.

Respond with ONLY valid JSON (no markdown fences, no explanation outside JSON) in this exact schema:
{
  "problem_type": "Algorithm" | "Data Structure" | "ML Design" | "System Design" | "Other",
  "problem_statement": "<concise restatement of the problem>",
  "approach": "<step-by-step explanation of the solution strategy, 3-6 sentences>",
  "solution_code": "<complete, runnable code>",
  "language": "python" | "javascript" | "java" | "cpp" | "other",
  "time_complexity": "O(...) — one-line explanation",
  "space_complexity": "O(...) — one-line explanation",
  "key_insights": ["<insight 1>", "<insight 2>", "..."],
  "follow_ups": ["<follow-up question 1>", "<follow-up question 2>", "..."]
}

Rules:
- For ML Design questions, put the architecture / pipeline in solution_code as pseudocode or a structured description.
- key_insights: 2-4 items max.
- follow_ups: 2-3 items max.
- Never include markdown code fences in solution_code — just the raw source.
- Always prefer Python unless another language is explicitly requested.`;

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
      apiKey:          document.getElementById('interview-api-key'),
      model:           document.getElementById('interview-model'),
      analyzeBtn:      document.getElementById('interview-analyze-btn'),
      analyzeIcon:     document.getElementById('interview-analyze-icon'),
      analyzeLabel:    document.getElementById('interview-analyze-label'),
      clearBtn:        document.getElementById('interview-clear-btn'),
      placeholder:     document.getElementById('interview-placeholder'),
      thinking:        document.getElementById('interview-thinking'),
      responseContent: document.getElementById('interview-response-content'),
      transcriptText:  document.getElementById('interview-transcript-text'),
      transcriptPart:  document.getElementById('interview-transcript-partial'),
      transcriptPh:    document.getElementById('interview-transcript-placeholder'),
      transcriptScroll:document.getElementById('interview-transcript-scroll'),
    };

    // Restore saved API key
    const saved = localStorage.getItem(LS_API_KEY);
    if (saved) el.apiKey.value = saved;

    // Save key on change
    el.apiKey.addEventListener('input', () => {
      localStorage.setItem(LS_API_KEY, el.apiKey.value.trim());
    });

    el.analyzeBtn.addEventListener('click', handleAnalyze);
    el.clearBtn.addEventListener('click', clearResults);

    // Mirror transcript updates into interview panel
    startTranscriptMirror();
  }

  // ──────────────────────────────────────────────
  // Transcript mirroring
  // Mirror the live transcript DOM into the interview left pane
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

    const apiKey = (el.apiKey.value || '').trim();
    if (!apiKey) {
      showError('Please enter your Anthropic API key (sk-ant-...).');
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
      const result = await callClaude(apiKey, el.model.value, transcript);
      renderResult(result);
    } catch (err) {
      showError(err.message || 'Unknown error calling Claude API.');
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

  // ──────────────────────────────────────────────
  // Claude API call
  // ──────────────────────────────────────────────
  async function callClaude(apiKey, model, transcript) {
    const userMessage = `Here is the live transcript of an interview session. Please analyze it and extract the coding/ML problem being asked, then solve it:\n\n---\n${transcript}\n---`;

    const body = {
      model: model,
      max_tokens: 4096,
      system: SYSTEM_PROMPT,
      messages: [
        { role: 'user', content: userMessage }
      ],
    };

    const response = await fetch(CLAUDE_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': ANTHROPIC_VERSION,
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      let errMsg = `API error ${response.status}`;
      try {
        const errBody = await response.json();
        errMsg = errBody?.error?.message || errMsg;
      } catch (_) {}
      throw new Error(errMsg);
    }

    const data = await response.json();
    const rawText = data?.content?.[0]?.text || '';

    // Parse JSON from Claude's response
    let parsed;
    try {
      // Strip any accidental markdown fences
      const clean = rawText.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
      parsed = JSON.parse(clean);
    } catch (_) {
      throw new Error('Failed to parse Claude response as JSON. Raw response:\n' + rawText.slice(0, 300));
    }

    return parsed;
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
    el.responseContent.innerHTML = `<div class="interview-error">${escapeHtml(msg)}</div>`;
    el.responseContent.classList.remove('hidden');
  }

  // ──────────────────────────────────────────────
  // Result rendering
  // ──────────────────────────────────────────────
  function renderResult(r) {
    el.thinking.classList.add('hidden');
    el.placeholder.classList.add('hidden');

    const badgeClass = getBadgeClass(r.problem_type);

    const html = `
      <!-- Problem Statement -->
      <div class="interview-section">
        <div class="interview-section-header">
          <span class="material-symbols-outlined interview-section-icon text-primary">assignment</span>
          <span class="interview-section-title">Problem</span>
        </div>
        <div class="interview-section-body">
          <div class="interview-problem-type-badge ${badgeClass}">
            ${escapeHtml(r.problem_type || 'Unknown')}
          </div>
          <p>${escapeHtml(r.problem_statement || '')}</p>
        </div>
      </div>

      <!-- Approach -->
      <div class="interview-section">
        <div class="interview-section-header">
          <span class="material-symbols-outlined interview-section-icon text-amber-400">lightbulb</span>
          <span class="interview-section-title">Approach</span>
        </div>
        <div class="interview-section-body interview-approach-text">
          ${formatApproach(r.approach || '')}
        </div>
      </div>

      <!-- Solution Code -->
      <div class="interview-section">
        <div class="interview-section-header">
          <span class="material-symbols-outlined interview-section-icon text-emerald-400">code</span>
          <span class="interview-section-title">Solution — ${escapeHtml(r.language || 'python')}</span>
        </div>
        <div class="interview-section-body" style="padding:0;">
          ${renderCodeBlock(r.solution_code || '', r.language || 'python')}
        </div>
      </div>

      <!-- Complexity -->
      <div class="interview-section">
        <div class="interview-section-header">
          <span class="material-symbols-outlined interview-section-icon text-cyan-400">query_stats</span>
          <span class="interview-section-title">Complexity</span>
        </div>
        <div class="interview-section-body">
          <div class="interview-complexity-row">
            <div class="interview-complexity-item">
              <span class="interview-complexity-label">Time</span>
              <span class="interview-complexity-value">${escapeHtml(r.time_complexity || '—')}</span>
            </div>
            <div class="interview-complexity-item">
              <span class="interview-complexity-label">Space</span>
              <span class="interview-complexity-value">${escapeHtml(r.space_complexity || '—')}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Key Insights -->
      ${renderList(r.key_insights, 'Key Insights', 'tips_and_updates', 'text-violet-400', 'interview-insights-list')}

      <!-- Follow-up Questions -->
      ${renderList(r.follow_ups, 'Follow-up Questions', 'quiz', 'text-slate-400', 'interview-followup-list')}
    `;

    el.responseContent.innerHTML = html;
    el.responseContent.classList.remove('hidden');

    // Wire up copy buttons for code blocks
    el.responseContent.querySelectorAll('.interview-code-copy-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const pre = btn.closest('.interview-code-block').querySelector('pre');
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
      <div class="interview-code-block">
        <div class="interview-code-header">
          <span class="interview-code-lang">${escapeHtml(lang)}</span>
          <button class="interview-code-copy-btn">
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
      <div class="interview-section">
        <div class="interview-section-header">
          <span class="material-symbols-outlined interview-section-icon ${iconColor}">${icon}</span>
          <span class="interview-section-title">${title}</span>
        </div>
        <div class="interview-section-body">
          <ul class="${listClass}">${lis}</ul>
        </div>
      </div>
    `;
  }

  // ──────────────────────────────────────────────
  // Tokenizer-based syntax highlighter
  // Works on RAW (unescaped) code to avoid HTML entity conflicts.
  // Tokenizes into strings, comments, decorators, keywords, numbers, and plain code,
  // then escapes each token individually before wrapping with <span>.
  // ──────────────────────────────────────────────

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

  // Master tokenizer regex — order matters: earlier alternatives win.
  // Groups: 1=triple-dq-string, 2=triple-sq-string, 3=dq-string, 4=sq-string,
  //         5=block-comment, 6=line-comment(// or #), 7=decorator,
  //         8=word, 9=number, 10=other
  const TOKEN_RE = /"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\/\*[\s\S]*?\*\/|\/\/[^\n]*|#[^\n]*|@\w+|\b[a-zA-Z_]\w*\b|\b\d+(?:\.\d+)?\b|[^\s]/g;

  function syntaxHighlight(code, lang) {
    const langDef = LANG_KEYWORDS[lang] || {};
    const kwSet = langDef.kw || new Set();
    const biSet = langDef.bi || new Set();

    const parts = [];

    let match;
    let lastIndex = 0;

    // Reset regex state
    TOKEN_RE.lastIndex = 0;

    while ((match = TOKEN_RE.exec(code)) !== null) {
      // Emit any whitespace / gap between tokens
      if (match.index > lastIndex) {
        parts.push(escapeHtml(code.slice(lastIndex, match.index)));
      }
      lastIndex = TOKEN_RE.lastIndex;

      const tok = match[0];
      const first = tok[0];
      const first2 = tok.slice(0, 2);
      const first3 = tok.slice(0, 3);

      // Triple-quoted strings
      if (first3 === '"""' || first3 === "'''") {
        parts.push('<span class="sh-string">' + escapeHtml(tok) + '</span>');
      }
      // Double/single-quoted strings
      else if ((first === '"' || first === "'") && tok.length > 1) {
        parts.push('<span class="sh-string">' + escapeHtml(tok) + '</span>');
      }
      // Block comments
      else if (first2 === '/*') {
        parts.push('<span class="sh-comment">' + escapeHtml(tok) + '</span>');
      }
      // Line comments (// or #)
      else if (first2 === '//' || first === '#') {
        parts.push('<span class="sh-comment">' + escapeHtml(tok) + '</span>');
      }
      // Decorators
      else if (first === '@') {
        parts.push('<span class="sh-decorator">' + escapeHtml(tok) + '</span>');
      }
      // Words (identifiers / keywords)
      else if (/^[a-zA-Z_]/.test(first)) {
        if (kwSet.has(tok)) {
          parts.push('<span class="sh-keyword">' + escapeHtml(tok) + '</span>');
        } else if (biSet.has(tok)) {
          parts.push('<span class="sh-builtin">' + escapeHtml(tok) + '</span>');
        } else {
          parts.push(escapeHtml(tok));
        }
      }
      // Numbers
      else if (/^\d/.test(first)) {
        parts.push('<span class="sh-number">' + escapeHtml(tok) + '</span>');
      }
      // Everything else (operators, punctuation)
      else {
        parts.push(escapeHtml(tok));
      }
    }

    // Trailing whitespace
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
    // Split on double-newline or numbered steps into paragraphs
    return text.split(/\n\n+/)
      .map(p => `<p>${escapeHtml(p.trim())}</p>`)
      .join('');
  }

  function getBadgeClass(type) {
    if (!type) return 'interview-badge-other';
    const t = type.toLowerCase();
    if (t.includes('ml') || t.includes('machine')) return 'interview-badge-ml';
    if (t.includes('system'))                        return 'interview-badge-system';
    if (t.includes('algo') || t.includes('data'))   return 'interview-badge-algo';
    return 'interview-badge-other';
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

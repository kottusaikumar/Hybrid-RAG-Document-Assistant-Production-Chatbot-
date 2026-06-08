const form    = document.getElementById('chat-form');
const input   = document.getElementById('question-input');
const history = document.getElementById('chat-history');
const sendBtn = document.getElementById('send-button');

const sessionId = crypto.randomUUID();

// ── Upload ────────────────────────────────────────────────────────────────────
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('pdf-upload');

uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const originalHTML  = uploadBtn.innerHTML;
    uploadBtn.innerHTML = '<i class="ph-bold ph-spinner ph-spin"></i> Uploading...';
    uploadBtn.disabled  = true;

    try {
        const res  = await fetch('/api/v1/reindex', { method: 'POST', body: formData });
        const data = await res.json();
        alert(res.ok ? data.message : `Error: ${data.detail}`);
    } catch {
        alert('Network error while uploading.');
    } finally {
        uploadBtn.innerHTML = originalHTML;
        uploadBtn.disabled  = false;
        fileInput.value     = '';
    }
});

// ── Textarea auto-resize ──────────────────────────────────────────────────────
input.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = this.value.trim() ? `${this.scrollHeight}px` : 'auto';
});

input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (this.value.trim()) form.dispatchEvent(new Event('submit'));
    }
});

// ── Message builders ──────────────────────────────────────────────────────────
function appendUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `
        <div class="avatar"><i class="ph-bold ph-user"></i></div>
        <div class="message-content"><p>${escapeHtml(text)}</p></div>`;
    history.appendChild(div);
    scrollBottom();
}

function createAIBubble() {
    const msgDiv  = document.createElement('div');
    msgDiv.className = 'message ai';

    const avatar  = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = '<i class="ph-bold ph-robot"></i>';

    const content = document.createElement('div');
    content.className = 'message-content';

    const statusEl = document.createElement('div');
    statusEl.className = 'status-msg';

    const textEl = document.createElement('p');
    textEl.className  = 'streaming-text';
    textEl.innerHTML  = '<span class="cursor">▍</span>';

    content.appendChild(statusEl);
    content.appendChild(textEl);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(content);
    history.appendChild(msgDiv);
    scrollBottom();

    return { msgDiv, statusEl, textEl, content };
}

// ── Metadata badges ───────────────────────────────────────────────────────────
function addMetadataBadges(content, sourcePdf, responseTime, scores) {
    const meta = document.createElement('div');
    meta.className = 'metadata';

    let confHtml = '';
    if (scores && scores.confidence !== null) {
        const level = scores.confidence < 50 ? 'low' : scores.confidence < 80 ? 'medium' : '';
        const title = `Faithfulness: ${(scores.faithfulness || 0).toFixed(2)}, Relevance: ${(scores.relevance || 0).toFixed(2)}`;
        confHtml = `<div class="badge confidence ${level}" title="${title}">
            <i class="ph-bold ph-shield-check"></i> Confidence: ${scores.confidence}%
        </div>`;
    }

    meta.innerHTML = `
        <div class="badge" title="Source Document">
            <i class="ph-bold ph-file-pdf"></i> ${escapeHtml(sourcePdf)}
        </div>
        <div class="badge" title="Generation Time">
            <i class="ph-bold ph-timer"></i> ${responseTime.toFixed(2)}s
        </div>
        ${confHtml}`;
    content.appendChild(meta);

    // Grounding warning — was defined in CSS but never rendered; fixed here
    if (scores && scores.grounding_warning) {
        const warn = document.createElement('div');
        warn.className = 'grounding-warning';
        warn.innerHTML = `<i class="ph-bold ph-warning-circle"></i>
            <span>Low faithfulness score — this answer may not be fully grounded in the document.</span>`;
        content.appendChild(warn);
    }
}

// ── Collapsible citations ─────────────────────────────────────────────────────
function addCitations(content, citations) {
    if (!citations || citations.length === 0) return;

    const wrapper   = document.createElement('div');
    wrapper.className = 'citations-wrapper';

    const toggle    = document.createElement('button');
    toggle.className = 'citations-toggle';
    toggle.innerHTML = `<i class="ph-bold ph-file-text"></i> Sources (${citations.length})
        <i class="ph-bold ph-caret-down citations-caret"></i>`;

    const list      = document.createElement('div');
    list.className  = 'citations-list';
    list.style.display = 'none';

    let open = false;
    toggle.addEventListener('click', () => {
        open = !open;
        list.style.display = open ? 'flex' : 'none';
        const caret = toggle.querySelector('.citations-caret');
        if (caret) caret.style.transform = open ? 'rotate(180deg)' : '';
    });

    citations.forEach((cite) => {
        const card = document.createElement('div');
        card.className = 'citation-card';
        card.innerHTML = `
            <div class="citation-header">
                <i class="ph-bold ph-file-pdf"></i>
                ${escapeHtml(cite.source)}
                <span class="citation-chunk">chunk #${cite.chunk_id}</span>
            </div>
            <p class="citation-excerpt">${escapeHtml(cite.excerpt)}${cite.excerpt.length >= 200 ? '…' : ''}</p>`;
        list.appendChild(card);
    });

    wrapper.appendChild(toggle);
    wrapper.appendChild(list);
    content.appendChild(wrapper);
}

// ── Feedback buttons ──────────────────────────────────────────────────────────
function addFeedbackButtons(content, questionText) {
    const wrapper  = document.createElement('div');
    wrapper.className = 'feedback-actions';

    const upBtn   = makeBtn('<i class="ph-bold ph-thumbs-up"></i>', 'Good response');
    const downBtn = makeBtn('<i class="ph-bold ph-thumbs-down"></i>', 'Bad response');

    let voted = false;
    const vote = async (type, active, other) => {
        if (voted) return;
        voted = true;
        active.classList.add('active', type === 'thumbs_up' ? 'up' : 'down');
        other.disabled = true;
        other.style.opacity = '0.5';
        try {
            await fetch('/api/v1/feedback', {
                method : 'POST',
                headers: { 'Content-Type': 'application/json' },
                body   : JSON.stringify({ session_id: sessionId, question: questionText, feedback_type: type }),
            });
        } catch { /* non-critical */ }
    };

    upBtn.onclick   = () => vote('thumbs_up',   upBtn, downBtn);
    downBtn.onclick = () => vote('thumbs_down', downBtn, upBtn);

    wrapper.appendChild(upBtn);
    wrapper.appendChild(downBtn);
    content.appendChild(wrapper);
}

function makeBtn(html, title) {
    const btn = document.createElement('button');
    btn.className = 'feedback-btn';
    btn.innerHTML = html;
    btn.title     = title;
    return btn;
}

// ── Markdown renderer ─────────────────────────────────────────────────────────
// Order: backtick spans FIRST (before HTML escaping so < > inside code is safe),
// then escape, then bold, then newlines.
function parseMarkdown(text) {
    // 1. Extract inline code spans before escaping (preserve their content)
    const codeSpans = [];
    text = text.replace(/`([^`]+)`/g, (_, code) => {
        codeSpans.push(code);
        return `\x00CODE${codeSpans.length - 1}\x00`;
    });

    // 2. Escape HTML
    text = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // 3. Bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // 4. Restore code spans (their content was captured before escaping)
    text = text.replace(/\x00CODE(\d+)\x00/g, (_, i) => {
        const escaped = codeSpans[+i]
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return `<code>${escaped}</code>`;
    });

    // 5. Newlines
    return text.replace(/\n/g, '<br>');
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function scrollBottom() {
    history.scrollTop = history.scrollHeight;
}

// ── Submit handler ────────────────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    input.value      = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    appendUserMessage(question);
    const { statusEl, textEl, content } = createAIBubble();

    let fullText = '';

    try {
        const res = await fetch('/api/v1/answer', {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify({ question, session_id: sessionId }),
        });

        if (!res.ok) {
            const err = await res.json();
            textEl.innerHTML = parseMarkdown(`Error: ${err.detail || 'Request failed'}`);
            return;
        }

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer    = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');

            for (let i = 0; i < lines.length - 1; i++) {
                const line = lines[i].trim();
                if (!line) continue;
                try {
                    const msg = JSON.parse(line);
                    handleMessage(msg, { statusEl, textEl, content, question });
                    if (msg.type === 'chunk') fullText += msg.content;
                } catch { /* incomplete JSON */ }
            }
            buffer = lines[lines.length - 1];
        }

        if (!fullText) {
            textEl.innerHTML = 'No response received.';
        } else if (textEl.innerHTML.includes('▍')) {
            textEl.innerHTML = parseMarkdown(fullText);
        }

    } catch {
        textEl.innerHTML = 'Network error: Cannot connect to the API. Make sure the server is running.';
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
});

function handleMessage(msg, { statusEl, textEl, content, question }) {
    switch (msg.type) {
        case 'status':
            statusEl.innerText = msg.message;
            scrollBottom();
            break;

        case 'early_citations':
            addCitations(content, msg.citations);
            statusEl.style.display = 'none';
            scrollBottom();
            break;

        case 'chunk':
            // fullText accumulated in caller
            textEl.innerHTML = parseMarkdown(
                // Recompute from running total — handled in submit loop
                (textEl.getAttribute('data-raw') || '') + msg.content
            ) + '<span class="cursor">▍</span>';
            textEl.setAttribute('data-raw', (textEl.getAttribute('data-raw') || '') + msg.content);
            scrollBottom();
            break;

        case 'metadata': {
            const raw = textEl.getAttribute('data-raw') || '';
            textEl.innerHTML = parseMarkdown(raw);
            textEl.removeAttribute('data-raw');
            addMetadataBadges(content, msg.source_pdf, msg.response_time_sec, msg.scores);
            if (msg.citations?.length && !content.querySelector('.citations-wrapper')) {
                addCitations(content, msg.citations);
            }
            addFeedbackButtons(content, question);
            scrollBottom();
            break;
        }
    }
}

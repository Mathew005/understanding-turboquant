/* ── Config ─────────────────────────────────────────────── */
const HOME = 'README.md';

const PAGES = {
  'README.md': 'Understanding TurboQuant',
  'benchmarks.md': 'Benchmarking pyturboquant',
};

/* ── Marked options ─────────────────────────────────────── */
marked.setOptions({ gfm: true, breaks: false });

/* ── Mermaid init ───────────────────────────────────────── */
mermaid.initialize({ startOnLoad: false, theme: 'default' });

/* ── Post-render: promote mermaid code blocks to diagrams ── */
async function renderMermaidBlocks() {
  const content = document.getElementById('content');
  const blocks = content.querySelectorAll('pre > code.language-mermaid');
  for (const code of blocks) {
    const pre = code.parentElement;
    const definition = code.textContent;
    const id = 'mermaid-' + Math.random().toString(36).slice(2);
    const div = document.createElement('div');
    div.className = 'mermaid-wrapper';
    try {
      const { svg } = await mermaid.render(id, definition);
      div.innerHTML = svg;
    } catch (e) {
      div.textContent = 'Diagram error: ' + e.message;
      div.style.color = 'red';
    }
    pre.replaceWith(div);
  }
}

/* ── KaTeX deferred render (called by KaTeX onload) ──────── */
function renderMathOnPage() {
  renderMathInElement(document.getElementById('content'), {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$',  right: '$',  display: false },
      { left: '\\(', right: '\\)', display: false },
      { left: '\\[', right: '\\]', display: true },
    ],
    throwOnError: false,
  });
}

/* ── Routing helpers ────────────────────────────────────── */
function hashToFile() {
  const raw = decodeURIComponent(location.hash.slice(1));
  return raw || HOME;
}

function navigateTo(file) {
  location.hash = encodeURIComponent(file);
}

/* ── Fetch and render a markdown file ───────────────────── */
async function loadPage(file) {
  const spinner = document.getElementById('spinner');
  const content = document.getElementById('content');
  const backBtn = document.getElementById('back-btn');
  const titleEl = document.getElementById('page-title');

  spinner.classList.add('active');
  content.style.opacity = '0';

  try {
    const res = await fetch(file);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const md = await res.text();

    // Rewrite internal .md hrefs to hash-based routes
    let html = marked.parse(md);
    html = html.replace(/href="([^"]+\.md)"/g, (_, href) => `href="#${href}"`);
    content.innerHTML = html;

    // Update UI chrome
    const isHome = file === HOME;
    backBtn.style.display = isHome ? 'none' : 'flex';
    document.title = PAGES[file] || file;
    titleEl.textContent = PAGES[file] || file;

    window.scrollTo({ top: 0, behavior: 'instant' });

    await renderMermaidBlocks();
    if (window.renderMathInElement) renderMathOnPage();

  } catch (err) {
    content.innerHTML = `<h2>Page not found</h2><p>Could not load <code>${file}</code>.</p><p>${err}</p>`;
  } finally {
    spinner.classList.remove('active');
    content.style.opacity = '1';
    content.style.animation = 'none';
    void content.offsetWidth;
    content.style.animation = '';
  }
}

/* ── Intercept clicks on rendered .md links ─────────────── */
document.getElementById('content').addEventListener('click', (e) => {
  const a = e.target.closest('a');
  if (!a) return;
  const href = a.getAttribute('href');
  if (href && href.startsWith('#') && href.endsWith('.md')) {
    e.preventDefault();
    location.hash = href.slice(1);
  }
});

/* ── Hash-change router ──────────────────────────────────── */
window.addEventListener('hashchange', () => loadPage(hashToFile()));

/* ── Initial load ───────────────────────────────────────── */
loadPage(hashToFile());

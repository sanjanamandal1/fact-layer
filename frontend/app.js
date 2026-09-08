/* ─────────────────────────────────────────────────────────────────────────
   FactLayer — frontend logic
   Simple, Minimalist & Cute UI Interaction
   ───────────────────────────────────────────────────────────────────────── */

const API = "http://localhost:8000";

// ── State ──────────────────────────────────────────────────────────────────

let allFacts = [];
let allRelationships = [];
let allDocuments = [];

// ── Tab Navigation ─────────────────────────────────────────────────────────

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".panel").forEach(p => p.classList.add("hidden"));

    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.remove("hidden");

    if (tab.dataset.tab === "facts")         renderFacts();
    if (tab.dataset.tab === "relationships") renderRelationships();
  });
});

// ── Upload ─────────────────────────────────────────────────────────────────

const uploadZone = document.getElementById("upload-zone");
const fileInput  = document.getElementById("file-input");

document.getElementById("browse-btn").addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("dragging");
});

uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragging"));

uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("dragging");
  [...e.dataTransfer.files].forEach(uploadFile);
});

fileInput.addEventListener("change", () => {
  [...fileInput.files].forEach(uploadFile);
  fileInput.value = "";
});

async function uploadFile(file) {
  if (!file.name.endsWith(".pdf")) {
    showResult("error", `"${file.name}" is not a PDF file. Please upload a PDF.`);
    return;
  }

  setProcessing(true, `Reading ${file.name}…`, "Extracting facts & checking relationships with Gemini");
  startProgressBar();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || "Upload failed");

    finishProgressBar();
    showResult("success",
      `✦ <strong>${escHtml(file.name)}</strong> successfully analyzed! ` +
      `Extracted <strong>${data.facts_extracted} facts</strong> across ${data.pages_processed} pages, ` +
      `and identified <strong>${data.relationships_found} connections</strong>.`
    );

    await refreshData();

  } catch (err) {
    finishProgressBar(true);
    showResult("error", `✕ ${err.message}`);
  } finally {
    setProcessing(false);
  }
}

// ── Progress Bar ──────────────────────────────────────────────────────────

let _progressTimer  = null;
let _elapsedTimer   = null;
let _elapsed        = 0;

const PROGRESS_STEPS = [
  { pct: 12, delay: 800,  label: "Reading pages…" },
  { pct: 28, delay: 4000, label: "Selecting key pages…" },
  { pct: 45, delay: 8000, label: "Extracting facts with Gemini…" },
  { pct: 62, delay: 18000, label: "Still extracting — large document…" },
  { pct: 76, delay: 28000, label: "Almost done…" },
  { pct: 88, delay: 38000, label: "Comparing across documents…" },
];

function startProgressBar() {
  const fill  = document.getElementById("progress-bar-fill");
  const timer = document.getElementById("processing-timer");
  if (!fill) return;

  fill.style.width = "0%";
  _elapsed = 0;
  if (timer) timer.textContent = "";

  // Tick elapsed time every second
  _elapsedTimer = setInterval(() => {
    _elapsed++;
    if (timer) timer.textContent = `${_elapsed}s elapsed — AI is thinking…`;
  }, 1000);

  // Step through realistic progress milestones
  _progressTimer = [];
  PROGRESS_STEPS.forEach(({ pct, delay, label }) => {
    const t = setTimeout(() => {
      fill.style.width = pct + "%";
      document.getElementById("processing-sub").textContent = label;
    }, delay);
    _progressTimer.push(t);
  });
}

function finishProgressBar(error = false) {
  const fill = document.getElementById("progress-bar-fill");
  if (fill) {
    fill.style.width = "100%";
    fill.style.background = error
      ? "var(--rose)"
      : "linear-gradient(90deg, var(--green), #34d399)";
  }
  if (_elapsedTimer) clearInterval(_elapsedTimer);
  if (_progressTimer) _progressTimer.forEach(clearTimeout);
  const timer = document.getElementById("processing-timer");
  if (timer) timer.textContent = error ? "Processing failed" : `Done in ${_elapsed}s`;
}


// ── Data Refresh ───────────────────────────────────────────────────────────

async function refreshData() {
  const [docs, rels] = await Promise.all([
    fetch(`${API}/documents`).then(r => r.json()).catch(() => []),
    fetch(`${API}/relationships`).then(r => r.json()).catch(() => []),
  ]);

  allDocuments     = docs;
  allRelationships = rels;

  // Aggregate all facts from document detail calls
  const factResults = await Promise.all(
    docs.map(d => fetch(`${API}/documents/${d.id}/facts`).then(r => r.json()).catch(() => ({ facts: [] })))
  );
  allFacts = factResults.flatMap(r => r.facts || []);

  renderDocuments();
  updateCounts();
  populateDocFilter();
}

// ── Document List ──────────────────────────────────────────────────────────

function renderDocuments() {
  const container = document.getElementById("documents-list");
  const empty     = document.getElementById("docs-empty");
  const clearBtn  = document.getElementById("clear-all-docs-btn");

  if (clearBtn) {
    clearBtn.onclick = clearAllDocuments;
  }

  if (!allDocuments.length) {
    if (container) container.innerHTML = "";
    if (empty) empty.classList.remove("hidden");
    if (clearBtn) clearBtn.classList.add("hidden");
    const recompBtn = document.getElementById("recompare-btn");
    if (recompBtn) recompBtn.classList.add("hidden");
    return;
  }

  if (empty) empty.classList.add("hidden");
  if (clearBtn) clearBtn.classList.remove("hidden");
  const recompBtn = document.getElementById("recompare-btn");
  if (recompBtn) {
    recompBtn.classList.toggle("hidden", allDocuments.length < 2);
    recompBtn.onclick = recompareAll;
  }

  container.innerHTML = allDocuments.map(doc => {
    const quality = Math.round((doc.quality_score || 0) * 100);
    const qColor  = quality >= 75 ? "var(--green)" : quality >= 50 ? "var(--amber)" : "var(--rose)";
    return `
      <div class="doc-card" id="doc-${doc.id}">
        <div class="doc-info">
          <div class="doc-name" title="${escHtml(doc.filename)}">📄 ${escHtml(doc.filename)}</div>
          <div class="doc-meta">
            <span class="doc-stat">${doc.page_count ?? "?"} pages</span>
            <span class="doc-stat">·</span>
            <span class="doc-stat">✦ ${doc.fact_count ?? 0} facts</span>
            <span class="doc-stat">·</span>
            <span class="doc-stat">
              Quality ${quality}%
              <span class="quality-bar">
                <span class="quality-fill" style="width:${quality}%; background:${qColor}"></span>
              </span>
            </span>
          </div>
        </div>
        <button class="delete-btn" onclick="deleteDocument('${doc.id}')" title="Remove document">Remove</button>
      </div>
    `;
  }).join("");
}

async function deleteDocument(docId) {
  if (!confirm("Remove this document and all its extracted facts?")) return;
  await fetch(`${API}/documents/${docId}`, { method: "DELETE" });
  await refreshData();
  renderFacts();
  renderRelationships();
}

async function clearAllDocuments() {
  if (!confirm("Are you sure you want to remove ALL documents and clear all facts?")) return;
  await fetch(`${API}/documents`, { method: "DELETE" });
  await refreshData();
  renderFacts();
  renderRelationships();
}

async function recompareAll() {
  const btn = document.getElementById("recompare-btn");
  if (btn) { btn.disabled = true; btn.textContent = "⟳ Running…"; }

  try {
    const res  = await fetch(`${API}/recompare`, { method: "POST" });
    const data = await res.json();
    showResult("success", `✦ Found <strong>${data.relationships_found} new connection${data.relationships_found !== 1 ? "s" : ""}</strong> across your documents.`);
    await refreshData();
    renderRelationships();
  } catch (err) {
    showResult("error", `✕ Recompare failed: ${err.message}`);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "⟳ Find Connections"; }
  }
}

// ── Facts View ─────────────────────────────────────────────────────────────

function renderFacts() {
  const container  = document.getElementById("facts-list");
  const docFilter  = document.getElementById("facts-doc-filter").value;
  const typeFilter = document.getElementById("facts-type-filter").value;
  const confFilter = document.getElementById("facts-conf-filter").value;

  let filtered = allFacts;
  if (docFilter)  filtered = filtered.filter(f => f.document_id === docFilter);
  if (typeFilter) filtered = filtered.filter(f => f.fact_type === typeFilter);
  if (confFilter) {
    if (confFilter === "high")   filtered = filtered.filter(f => f.confidence > 0.8);
    if (confFilter === "medium") filtered = filtered.filter(f => f.confidence >= 0.5 && f.confidence <= 0.8);
    if (confFilter === "low")    filtered = filtered.filter(f => f.confidence < 0.5);
  }

  if (!filtered.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">✦</div>
        <p class="empty-title">${allFacts.length ? "No facts match current filters" : "No facts extracted yet"}</p>
        <p class="empty-subtitle">${allFacts.length ? "Try resetting your filters above." : "Upload a PDF in the Upload tab to discover extracted claims."}</p>
      </div>
    `;
    return;
  }

  const docMap = Object.fromEntries(allDocuments.map(d => [d.id, d.filename]));

  container.innerHTML = filtered.map(f => {
    const confPct   = Math.round(f.confidence * 100);
    const confClass = f.confidence > 0.8 ? "high" : f.confidence >= 0.5 ? "medium" : "low";
    const scope     = [f.temporal_scope, f.entity_scope].filter(Boolean).join(" · ");

    return `
      <div class="fact-card">
        <div class="fact-card-top">
          <p class="fact-claim">${escHtml(f.claim)}</p>
          <span class="fact-confidence conf-${confClass}">${confPct}%</span>
        </div>
        <p class="fact-quote">"${escHtml(truncate(f.exact_quote, 160))}"</p>
        <div class="fact-meta">
          <span class="badge badge-${f.fact_type}">${f.fact_type}</span>
          ${scope ? `<span class="badge badge-scope">${escHtml(scope)}</span>` : ""}
        </div>
        ${f.uncertainty_reason ? `<p class="uncertainty-note">⚠ ${escHtml(f.uncertainty_reason)}</p>` : ""}
        <p class="fact-source">
          📄 ${escHtml(docMap[f.document_id] || "Unknown")} · Page ${f.page_number}
        </p>
      </div>
    `;
  }).join("");
}

["facts-doc-filter", "facts-type-filter", "facts-conf-filter"].forEach(id => {
  document.getElementById(id).addEventListener("change", renderFacts);
});

function populateDocFilter() {
  const select = document.getElementById("facts-doc-filter");
  const current = select.value;
  select.innerHTML = `<option value="">All documents</option>` +
    allDocuments.map(d => `<option value="${d.id}" ${d.id === current ? "selected" : ""}>${escHtml(d.filename)}</option>`).join("");
}

// ── Relationships View ─────────────────────────────────────────────────────

let activeRelFilter = "all";

document.querySelectorAll(".filter-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".filter-chip").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    activeRelFilter = chip.dataset.rel;
    renderRelationships();
  });
});

function renderRelationships() {
  const container = document.getElementById("relationships-list");

  let filtered = allRelationships;
  if (activeRelFilter !== "all") filtered = filtered.filter(r => r.relationship === activeRelFilter);

  if (!filtered.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🌱</div>
        <p class="empty-title">${allRelationships.length ? "No connections match this filter" : "No cross-document connections yet"}</p>
        <p class="empty-subtitle">${allRelationships.length ? "Try selecting 'All' above." : "Upload 2 or more reports (like FY23 and FY24) to see agreements and contradictions."}</p>
      </div>
    `;
    return;
  }

  container.innerHTML = filtered.map(r => {
    const typeLabel = {
      CORROBORATED: "✓ Corroborated",
      CONTRADICTED: "✕ Contradicted",
      RECONCILED:   "⟳ Reconciled",
    }[r.relationship] || r.relationship;

    return `
      <div class="rel-card">
        <div class="rel-header">
          <span class="rel-type-badge rel-${r.relationship}">${typeLabel}</span>
          <span class="rel-docs">${escHtml(r.filename_a)} ↔ ${escHtml(r.filename_b)}</span>
          <span class="rel-conf">${Math.round(r.confidence * 100)}% match</span>
        </div>

        <div class="rel-facts">
          <div class="rel-fact">
            <span class="rel-fact-label">From ${escHtml(r.filename_a)}</span>
            <p class="rel-fact-claim">${escHtml(r.claim_a)}</p>
            <p class="rel-fact-quote">"${escHtml(truncate(r.quote_a, 140))}"</p>
            <p class="rel-fact-source">Page ${r.page_a} · ${r.type_a}${r.temporal_a ? " · " + r.temporal_a : ""}</p>
          </div>
          <div class="rel-fact">
            <span class="rel-fact-label">From ${escHtml(r.filename_b)}</span>
            <p class="rel-fact-claim">${escHtml(r.claim_b)}</p>
            <p class="rel-fact-quote">"${escHtml(truncate(r.quote_b, 140))}"</p>
            <p class="rel-fact-source">Page ${r.page_b} · ${r.type_b}${r.temporal_b ? " · " + r.temporal_b : ""}</p>
          </div>
        </div>

        <div class="rel-reasoning">
          <span class="rel-reasoning-label">Reasoning & Alignment</span>
          <p class="rel-reasoning-text">${escHtml(r.reasoning)}</p>
          ${r.reconciliation_context
            ? `<p class="rel-reconcile-note">Note: ${escHtml(r.reconciliation_context)}</p>`
            : ""}
        </div>
      </div>
    `;
  }).join("");
}

// ── Helpers ────────────────────────────────────────────────────────────────

function updateCounts() {
  document.getElementById("facts-count").textContent = allFacts.length;
  document.getElementById("rels-count").textContent  = allRelationships.length;
}

function setProcessing(active, title = "", sub = "") {
  const card = document.getElementById("processing-card");
  if (active) {
    document.getElementById("processing-title").textContent = title;
    document.getElementById("processing-sub").textContent   = sub;
    card.classList.remove("hidden");
  } else {
    card.classList.add("hidden");
  }
}

function showResult(type, html) {
  const card = document.getElementById("result-card");
  card.className = `result-card ${type}`;
  card.innerHTML = html;
  setTimeout(() => card.classList.add("hidden"), 8000);
}

function escHtml(str) {
  if (!str) return "";
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function truncate(str, n) {
  if (!str || str.length <= n) return str || "";
  return str.slice(0, n) + "…";
}

// ── Init ───────────────────────────────────────────────────────────────────

refreshData();

# Fact Knowledge Layer

A system that reads financial and legal PDFs, extracts meaningful facts, grounds every fact in its source evidence, and discovers when facts across documents agree, conflict, or only *appear* to conflict.

---

## Setup and Run

**Requirements:** Python 3.10+, a free Gemini API key from [aistudio.google.com](https://aistudio.google.com)

```bash
# 1. Clone and navigate
git clone <your-repo-url>
cd fact-knowledge-layer

# 2. Install dependencies
cd backend
pip install -r requirements.txt

# 3. Set your Gemini API key
# Windows
set GEMINI_API_KEY=your_key_here

# macOS / Linux
export GEMINI_API_KEY=your_key_here

# 4. Start the server (serves both API and frontend)
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser. Upload any PDF — no config, no hardcoded rules.

---

## Demo Video

[▶ Watch the 3-minute demo](YOUR_VIDEO_LINK_HERE)

The video shows a full upload-to-insight flow and all four required cases below.

---

## The Four Required Cases

### 1. Corroboration — Same fact, different wording

Two documents may state the same registered address or revenue figure using different formats. The system embeds both claims semantically, flags them as candidates, and the LLM confirms they agree — surfacing both source quotes side by side.

> *"Both documents confirm the company's CIN as U72900KA2020PTC138452, though one writes it inline and the other in a header table."*

### 2. Genuine Contradiction

A director listed as active in one filing and as having resigned in a later document. Same person, same company, different status — a real conflict the LLM cannot explain away with context.

> *"Doc 1 (FY22 annual report): 'Mr. Rajiv Menon serves as Independent Director.' Doc 2 (FY24 filing): 'Mr. Rajiv Menon tendered his resignation effective June 2023.'"*

### 3. Apparent Contradiction — Explained by Context

Revenue figures that look mismatched because one is standalone and the other is consolidated, or because one covers six months while the other covers the full year. The system detects the scope and period tags on each fact and classifies the pair as RECONCILED with an explanation.

> *"₹95 Cr (standalone, H1 FY23) vs ₹210 Cr (consolidated, FY23). Different scope and period — not a conflict."*

### 4. Extraction Failure — Surfaced Honestly

Multi-row merged tables in pdfplumber lose their row-header alignment. Numbers get extracted without context. The system detects this by checking if page text quality falls below a threshold (character-density heuristic), lowers confidence on all facts from that page, and flags them in the "Uncertainty Log" in the UI.

**What I'd fix:** Use PyMuPDF's `get_text("dict")` or a bounding-box-aware parser to reconstruct table structure before passing to the LLM. For scanned documents, Tesseract OCR as a preprocessing step would recover text the current approach drops entirely.

---

## Approach

### The core problem I was solving

Financial documents are adversarial for simple extraction systems. The same fact can appear six times in one document and mean something different each time — audited vs unaudited, standalone vs consolidated, current year vs restated prior year. A system that extracts numbers without understanding this context will generate false contradictions and miss real ones. Most of the work here is in that middle layer.

### Architecture

```
PDF Upload
    │
    ▼
Page-by-page extraction (pdfplumber)
    + text quality assessment (character density heuristic)
    │
    ▼
Fact extraction per page (Gemini 1.5 Flash)
    → claim, type, temporal scope, entity scope, exact quote, confidence
    │
    ▼
Embedding (sentence-transformers / all-MiniLM-L6-v2, runs locally)
    │
    ▼
Cross-document candidate pairs (cosine similarity ≥ 0.55)
    │
    ▼
LLM comparison per candidate pair (Gemini 1.5 Flash)
    → CORROBORATED / CONTRADICTED / RECONCILED / UNRELATED + reasoning
    │
    ▼
SQLite storage → REST API → Vanilla JS UI
```

### Key decisions and why

**Why page-by-page extraction, not full-document?**
Fact extraction needs to return an exact page number for evidence grounding. Feeding the full document at once loses that granularity. It also lets us skip low-quality pages individually rather than degrading the whole document.

**Why a similarity filter before LLM comparison?**
Comparing every fact against every other fact across documents is O(n²) LLM calls — expensive and slow. Cosine similarity on local embeddings is cheap and runs in milliseconds. The LLM only sees candidate pairs that are already likely to be related.

**Why conservative classification?**
For a banking use-case, a false positive (calling something a contradiction when it isn't) erodes trust faster than a missed relationship. The comparison prompt is written to prefer RECONCILED over CONTRADICTED when any contextual explanation is plausible.

**Why SQLite, not a graph database?**
The assignment specifically notes that a graph database alone is not the solution. SQLite is transparent, portable, and sufficient for this scale. The interesting logic is in extraction and comparison — not the storage layer.

**Why Gemini 1.5 Flash for extraction?**
The 1M-token context window means we could theoretically send entire documents at once. More practically, it's free, fast, and capable enough that the bottleneck is prompt quality, not model capability.

**Why vanilla JS for the frontend?**
No build step, no framework overhead, runs directly from the static file server. The UI does three things — upload, show facts, show relationships — and vanilla JS handles that cleanly without hiding what's happening.

### AI tools used
- **Gemini 1.5 Flash** — fact extraction and cross-document comparison
- **sentence-transformers (all-MiniLM-L6-v2)** — local semantic embeddings (free, no API key)

---

## Limitations and Next Steps

**What doesn't work well yet:**

- **Scanned PDFs.** pdfplumber returns garbled or empty text for image-based PDFs. The quality heuristic catches this and lowers confidence, but the facts are still poor. Fix: Tesseract OCR or Google Vision as a preprocessing step.

- **Complex tables.** Merged cells and multi-header tables lose structure during text extraction. Numbers come through but row context is lost. Fix: bounding-box-aware extraction (PyMuPDF's dict mode) to reconstruct table structure before passing to the LLM.

- **Disambiguation at scale.** With many documents, "Mr. Sharma" across three companies is ambiguous. The system currently has no entity resolution step — it matches on semantic similarity, not identity. Fix: a separate entity-linking pass to canonicalize named entities before comparison.

- **Extraction latency.** Processing is synchronous. A 50-page PDF with dense content takes 30–60 seconds. Fix: a background job queue (Celery, or even a simple asyncio task) with polling for status.

**What I'd build next for Superjoin specifically:**

The natural next layer on top of this is a **fact audit checklist** for merchant bankers. Before a DRHP filing, a banker could upload the draft, prior annual reports, and director disclosures — and instead of reading 500 pages, get a structured list of: facts that are consistent across sources, facts that need reconciliation, and genuine conflicts that need resolution before filing. That's where this knowledge layer becomes genuinely valuable in the IPO readiness context.

---

## Additional Notes

The threshold for semantic similarity (0.55) is a tunable parameter. Lower it to catch more candidate pairs at the cost of more LLM calls and more noise. Raise it to only compare very similar claims. For financial documents, 0.55 worked well across the test set — it catches paraphrases and unit variations without generating too many spurious pairs.

The system handles new documents incrementally. When a new PDF is uploaded, it only computes relationships between the new document's facts and all existing facts — it does not recompute existing cross-document relationships. This keeps upload time proportional to the size of the new document, not the total knowledge base.

```text
███████╗ █████╗  ██████╗████████╗██╗      █████╗ ██╗   ██╗███████╗██████╗ 
██╔════╝██╔══██╗██╔════╝╚══██╔══╝██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗
█████╗  ███████║██║        ██║   ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝
██╔══╝  ██╔══██║██║        ██║   ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗
██║     ██║  ██║╚██████╗   ██║   ███████╗██║  ██║   ██║   ███████╗██║  ██║
╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
```

# Fact Knowledge Layer

Reads financial and legal PDFs, pulls out specific facts and numbers, grounds every claim with its original quote and page number, and figures out when facts across documents agree, conflict, or only seem to conflict. Built for merchant banking and IPO readiness checks.

---

## Setup and Run Instructions

### Prerequisites
- Python 3.10+
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### Installation & Run

1. Clone repository and go to the backend folder:
   ```bash
   git clone <git clone https://github.com/sanjanamandal1/fact-layer.git>
   cd fact-knowledge-layer/backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your Gemini API key:
   - **Windows (PowerShell):**
     ```powershell
     $env:GEMINI_API_KEY = "your_gemini_api_key_here"
     ```
   - **macOS / Linux:**
     ```bash
     export GEMINI_API_KEY="your_gemini_api_key_here"
     ```

4. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

5. Open your browser at:
   ```
   http://localhost:8000
   ```
   *The web UI is served directly by FastAPI — no Node.js, npm, or build step needed.*

---

## Video Demo

> 📹 **Video Demo Link:** [Watch Demo on Drive](https://drive.google.com/file/d/1sLnw-nexLpUijM1htwPvLS5JULkl343r/view?usp=drivesdk) *(a little more than 3 mins)*

---

## Approach

### How the Pipeline Works

```
PDF Upload
   │
   ▼
1. Page-by-Page Reading & Quality Check
   • Uses pypdf for speed, falls back to pdfplumber
   • Scores each page for readability (flags broken text or empty scans)
   │
   ▼
2. Page Ranking (Picking the Richest Pages)
   • Scores pages based on numbers, currency symbols, and financial terms
   • Selects the 20 most fact-rich pages (always keeps pages 1–3 for company info)
   │
   ▼
3. Fact Extraction via Gemini
   • Sends pages in batches to extract structured facts (JSON)
   • Captures: claim, type, period, scope, exact quote, confidence, and page number
   • Built-in model fallback automatically handles free-tier rate limits
   │
   ▼
4. Local Embeddings
   • scikit-learn HashingVectorizer (512-dim vectors)
   • Runs locally in milliseconds with zero API cost and no rate limits
   │
   ▼
5. Smart Pre-Filter (Similarity ≥ 0.30)
   • Filters thousands of possible pairs down to the ~10–20 that actually look related
   │
   ▼
6. Cross-Document Comparison (LLM)
   • Labels each pair: CORROBORATED / CONTRADICTED / RECONCILED / UNRELATED
   • Conservative approach: prefers RECONCILED if dates or scopes explain the difference
   │
   ▼
7. Storage & UI (SQLite + Clean Single-Page App)
   • Saves everything in a local SQLite database
   • Clean UI to search facts, check quotes, and inspect connections
```

### The Four Required Cases

#### 1. Corroboration — Same Fact, Different Wording
- **What happened:** Both the Delhivery annual report and the Q4 earnings presentation report the debt-to-equity ratio as `0.01x` as of March 31, 2024.
- **Evidence:** The presentation has it in a financial table as `"Debt/Equity (A/C)... 0.01x"`, while the annual report mentions `"March 31, 2024: 0.01"` in a summary section.
- **System Reasoning:** The system matches the dates and numbers, realizes both documents are saying the same thing despite different wording, and marks it `CORROBORATED` with 100% confidence.

#### 2. Genuine Contradiction
- **What happened:** In the Q4 earnings presentation (page 5), Delhivery states that Net Working Capital (NWC) days dropped from 38 to **31 days** in FY24. In the FY24 annual report (page 8), it says the NWC cycle dropped from 38 to **27 days**.
- **Evidence:** Both documents describe the exact same company, the same metric, and the same fiscal year (FY24), but give two different ending numbers (31 vs 27).
- **System Reasoning:** The system identifies that the scopes and dates match, but the numbers clash directly. It flags this as `CONTRADICTED` and shows both quotes side-by-side so an analyst can investigate.

#### 3. Apparent Contradiction Explained by Context (Reconciled)
- **What happened:** In the 2022 IPO Prospectus, Delhivery's Corporate Identity Number (CIN) is listed as `U63090DL2011PLC221234`. In the FY24 Annual Report, it is listed as `L63090DL2011PLC221234`.
- **Evidence:** Both documents report the official registration number, but with a different first letter ('U' vs 'L').
- **System Reasoning:** Rather than flagging this as an error, the system recognizes that 'U' stands for Unlisted and 'L' stands for Listed. It explains that the difference is due to Delhivery completing its IPO and becoming a publicly listed entity between the 2022 filing and the 2024 report.
- *(Also handles personnel updates over time, like the Company Secretary changing between 2022 and 2024, and standalone vs. consolidated financial differences).*

#### 4. Extraction Failure — Surfaced Honestly
- **What happened:** Tables with merged headers or complex columns lose their structure when turned into raw text. Numbers get extracted, but they can lose the row label that explains what the number means.
- **How it's handled:** The system checks the text quality of every page. If a page has strange character spacing or low text density, any facts pulled from that page get a lower confidence score (below 0.60) and an `uncertainty_reason` note. Analysts see these highlighted with caution flags rather than trusting them blindly.
- **How to improve it:** Use bounding-box table extraction (like PyMuPDF table rects) to reconstruct tables before sending text to the LLM, and add OCR preprocessing for scanned images.

### Key Engineering Decisions & Trade-Offs

- **Picking 20 Dense Pages vs. Dumping 400 Pages:** Feeding a 400-page prospectus into an LLM all at once loses accuracy, misses subtle details, and loses exact page citations. Scoring and picking the 20 richest financial pages captures the core numbers while keeping page numbers 100% accurate.
- **Local Embeddings vs. Embedding API Calls:** If 3 documents have 100 facts each, comparing every pair with an LLM would take almost 5,000 to 30,000 API calls. Using a local `HashingVectorizer` filters this down to 10–20 high-likelihood candidates in under a second for zero cost.
- **Conservative Comparison (Fewer False Alarms):** In IPO due diligence, crying wolf with false contradictions destroys trust immediately. The prompt is intentionally instructed to look for contextual explanations (like different periods or scopes) before declaring a contradiction.
- **SQLite vs. Graph Database:** A graph database adds heavy setup without solving the hard part. The real challenge is extracting and comparing facts. SQLite is lightweight, portable, fast, and easy for reviewers to inspect.
- **No-Build Vanilla Frontend:** Built with clean HTML, CSS, and plain JavaScript served right from FastAPI. No `npm install`, no webpack, and no broken node dependencies for the reviewer.

### AI Tools Used
- **Gemini Flash (`gemini-flash-latest`, `gemini-flash-lite-latest`):** Fast, low-cost model used for structured extraction and semantic comparison via Google GenAI SDK.
- **scikit-learn HashingVectorizer:** Fast local text vectorizer for candidate pair pre-filtering.

---

## Limitations and Next Steps

### Current Limitations
1. **Scanned PDFs:** Pure image scans return little to no text with standard parsers. While our quality filter flags these pages, adding an OCR step (like Tesseract) is needed for full scanned PDF coverage.
2. **Flattened Tables:** Complex multi-header tables lose their column-to-row alignments in plain text streams.
3. **Name Matching:** Names are compared as plain text. If one filing says "Sunil Bansal" and another says "S.K. Bansal", an entity resolution step is needed to recognize they are the same person.

### Next Steps & Product Vision
- **IPO Pre-Filing Audit Checklist:** An automated tool where merchant bankers upload draft prospectuses alongside prior annual reports to automatically flag discrepancies before filing with regulators.
- **Table Cell Reconstruction:** Using spatial bounding boxes to keep table rows and columns aligned before parsing.

---

## Additional Notes

- **Incremental Processing (Bonus Point):** You can upload new documents one at a time without re-processing older documents. New facts are saved to SQLite and automatically compared against previously stored facts. You can also click **⟳ Find Connections** at any time to re-run comparisons.
- **Rate-Limit Resilience:** The backend includes retry logic and fallback models so temporary rate limits on the free tier never crash the app.
- **Runs Fully Locally:** No external vector database or cloud storage needed; all data and embeddings stay inside `facts.db`.

import os
import json
import re
import time
import pdfplumber
from google import genai
from google.genai import types, errors as genai_errors
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=api_key)
    return _client

_ACTIVE_MODEL = None
CANDIDATE_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]

def generate_content_with_fallback(client, contents):
    global _ACTIVE_MODEL
    env_model = os.environ.get("GEMINI_MODEL")
    models_to_try = []
    if _ACTIVE_MODEL:
        models_to_try.append(_ACTIVE_MODEL)
    if env_model and env_model not in models_to_try and "1.5" not in env_model and "2.0" not in env_model:
        models_to_try.append(env_model)
    for m in CANDIDATE_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    last_error = None
    config = types.GenerateContentConfig(response_mime_type="application/json")
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(model=model_name, contents=contents, config=config)
            _ACTIVE_MODEL = model_name
            return response
        except genai_errors.ClientError as e:
            if "429" in str(e) or "resource_exhausted" in str(e).lower() or "quota" in str(e).lower():
                # Free-tier quota exceeded on this model — try pause or fall through to next model
                time.sleep(2.0)
                try:
                    response = client.models.generate_content(model=model_name, contents=contents, config=config)
                    _ACTIVE_MODEL = model_name
                    return response
                except Exception:
                    last_error = e
                    continue
            if "404" in str(e) or "not available" in str(e).lower() or "not found" in str(e).lower():
                last_error = e
                continue
            raise e
    if last_error:
        raise last_error

def parse_json_facts(raw: str) -> List[dict]:
    raw = (raw or "").strip()
    if not raw:
        return []

    # 1. Direct JSON parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict) and "claim" in x]
        if isinstance(data, dict):
            for k in ["facts", "claims", "items"]:
                if isinstance(data.get(k), list):
                    return [x for x in data[k] if isinstance(x, dict) and "claim" in x]
    except Exception:
        pass

    # 2. Strip code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict) and "claim" in x]
    except Exception:
        pass

    # 3. Balanced bracket extractor for [ ... ]
    start = raw.find("[")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            c = raw[i]
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == '[':
                    depth += 1
                elif c == ']':
                    depth -= 1
                    if depth == 0:
                        candidate = raw[start:i+1]
                        try:
                            data = json.loads(candidate)
                            if isinstance(data, list):
                                return [x for x in data if isinstance(x, dict) and "claim" in x]
                        except Exception:
                            pass
                        break

    return []

# ── PDF Text Extraction ────────────────────────────────────────────────────

RICH_KEYWORDS = [
    "revenue", "profit", "loss", "ebitda", "margin", "crore", "lakh", "million",
    "billion", "equity", "share", "dividend", "debt", "asset", "liability", "expense",
    "director", "auditor", "promoter", "subsidiary", "board", "incorporat", "registered",
    "cin", "sebi", "bse", "nse", "fy2", "fy1", "growth", "headcount", "customer",
    "delhivery", "limited", "restated", "consolidated", "standalone"
]

def _score_page_richness(text: str) -> float:
    """Score a page by how densely it contains metrics, currency, and corporate terms."""
    if not text:
        return 0.0
    lower = text.lower()
    kw_score = sum(3 for kw in RICH_KEYWORDS if kw in lower)
    digit_score = min(sum(1 for c in text if c.isdigit()), 40)
    symbol_score = min(sum(2 for c in text if c in "₹$%"), 20)
    return kw_score + digit_score + symbol_score


def extract_pages(pdf_path: str) -> Tuple[List[Tuple[int, str, float]], int, float]:
    """
    Extract text page-by-page and select the most fact-rich pages.

    Returns: (selected_pages, total_pdf_pages, overall_quality_score)
    """
    pages = []
    total_pages = 0

    # 1. Fast text extraction using pypdf, falling back to pdfplumber
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        total_pages = len(reader.pages)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            quality = _assess_text_quality(text)
            pages.append((i + 1, text, quality))
    except Exception:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                quality = _assess_text_quality(text)
                pages.append((i + 1, text, quality))

    if not pages:
        return [], 0, 0.0

    avg_quality = sum(q for _, _, q in pages) / len(pages)

    # Filter out empty or unreadable pages
    valid_pages = [p for p in pages if p[2] >= 0.35 and len(p[1].strip()) >= 50]

    # Target up to 36 most fact-dense pages
    MAX_PROCESSED_PAGES = 36
    if len(valid_pages) <= MAX_PROCESSED_PAGES:
        selected_pages = valid_pages
    else:
        # Keep early foundational pages (pages 1 to 3) for corporate identity/CIN
        early_pages = [p for p in valid_pages if p[0] <= 3]
        other_pages = [p for p in valid_pages if p[0] > 3]

        # Prioritize the most fact-rich pages (financials, tables, operations)
        scored = sorted(other_pages, key=lambda p: _score_page_richness(p[1]), reverse=True)
        needed = MAX_PROCESSED_PAGES - len(early_pages)
        selected_pages = early_pages + scored[:needed]
        # Re-sort by page number to keep natural document order
        selected_pages.sort(key=lambda p: p[0])

    return selected_pages, total_pages, round(avg_quality, 2)


def _assess_text_quality(text: str) -> float:
    """
    Heuristic: ratio of readable characters to total.
    """
    if not text or len(text.strip()) < 30:
        return 0.0
    readable = sum(
        1 for c in text
        if c.isalnum() or c.isspace() or c in ".,;:()-₹%/'\"$£€"
    )
    return round(readable / len(text), 2)


# ── Fact Extraction via Gemini ─────────────────────────────────────────────

DOCUMENT_EXTRACTION_PROMPT = """You are a senior financial and legal document analyst.
Your task is to extract a comprehensive, high-density collection of distinct, verifiable facts from the document pages below.
Do NOT give a superficial summary or skip pages. Aim for 3 to 6 distinct, granular facts per page.

Extract all verifiable facts across these categories:
- Numerical facts: specific revenue numbers, EBITDA, net profit/loss, margins, CAGR, total borrowings/debt, cash flow, total assets, equity valuation, issue size, share price/pricing bands, employee headcount, operational scale (parcels/orders handled, PIN codes serviced, automated hubs, vehicle fleet size).
- Entity facts: corporate entity names, founders, promoters, board directors, key managerial personnel (CEO, CFO, CS), statutory auditors, book running lead managers, syndicate members, key institutional shareholders.
- State facts: listed/unlisted status, stock exchange listing (NSE, BSE), incorporation status (public/private limited), board committee approvals, regulatory clearances (SEBI, RoC, RBI).
- Date facts: date of incorporation, conversion to public company, filing dates, reporting period dates (e.g. FY2021, FY2022, 9-months ended Dec 31, 2021), issue opening/closing dates.

For each fact return a JSON object:
{{
  "page_number": <integer page number from which this fact was extracted>,
  "claim": "clear, self-contained, and precise statement of the fact",
  "fact_type": "numerical" | "entity" | "state" | "date",
  "temporal_scope": "FY2022" | "as of Dec 31, 2021" | "Q3 FY22" | null,
  "entity_scope": "consolidated" | "standalone" | null,
  "exact_quote": "verbatim sentence or data line from that page supporting this fact",
  "confidence": 0.85 to 1.0,
  "uncertainty_reason": null
}}

Rules:
- Extract all verifiable facts — do NOT skip pages or omit granular metrics.
- exact_quote MUST be verbatim from the text of that specific page.
- Make sure "page_number" correctly matches the section header ('--- PAGE X ---').
- Return ONLY a valid JSON array of fact objects.

Document pages:
{pages_content}
"""


def extract_facts_from_document(pages: List[Tuple[int, str, float]], batch_size: int = 10) -> List[dict]:
    """
    Extract facts in small batches (e.g. 10 pages per call) using the high-density prompt.
    Produces 50-100+ granular facts while staying safely below the 15 RPM free-tier limit.
    """
    if not pages:
        return []

    # Split pages into batches of up to batch_size
    batches = [pages[i:i + batch_size] for i in range(0, len(pages), batch_size)]
    all_facts = []
    client = get_client()

    for idx, batch in enumerate(batches):
        sections = []
        for page_number, text, quality in batch:
            if text.strip() and quality >= 0.35:
                clean_text = text[:3200].strip()
                sections.append(f"--- PAGE {page_number} ---\n{clean_text}")

        if not sections:
            continue

        full_content = "\n\n".join(sections)
        prompt = DOCUMENT_EXTRACTION_PROMPT.format(pages_content=full_content)

        # Gentle pause between batches to stay safely within free-tier RPM
        if idx > 0:
            time.sleep(1.2)

        try:
            response = generate_content_with_fallback(client, prompt)
            raw = response.text.strip() if response and response.text else ""
            batch_facts = parse_json_facts(raw)

            valid_batch_pages = {p[0] for p in batch}
            fallback_page = batch[0][0]
            batch_qualities = {p[0]: p[2] for p in batch}

            for f in batch_facts:
                if not isinstance(f.get("page_number"), int) or f["page_number"] not in valid_batch_pages:
                    f["page_number"] = fallback_page

                q = batch_qualities.get(f["page_number"], 1.0)
                if q < 0.7:
                    f["confidence"] = round(f.get("confidence", 0.5) * q, 2)
                    if not f.get("uncertainty_reason"):
                        f["uncertainty_reason"] = f"Source page quality is low ({q:.0%})"

            all_facts.extend(batch_facts)
        except Exception as e:
            # Continue to next batch if one encounters an error
            continue

    return all_facts


def extract_facts_from_page(page_number: int, text: str, page_quality: float) -> List[dict]:
    """Backward compatibility helper for single-page extraction."""
    return extract_facts_from_document([(page_number, text, page_quality)])


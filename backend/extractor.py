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
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
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

# Free tier limit: 15 requests/min. A 0.5s pause keeps us under ~30 req/min
# on fast pages; for large documents we batch pages to stay well within limits.
_INTER_PAGE_DELAY = 1.0   # seconds between API calls
_MAX_PAGES = 40           # cap per upload — beyond this, batch pages together


# ── PDF Text Extraction ────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> Tuple[List[Tuple[int, str, float]], float]:
    """
    Extract text page-by-page and assess overall document quality.

    Why page-by-page?
    - Preserves exact page references for evidence grounding.
    - Lets us skip low-quality pages individually rather than rejecting the whole doc.

    Returns: (pages, overall_quality_score)
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            quality = _assess_text_quality(text)
            pages.append((i + 1, text, quality))

    if not pages:
        return [], 0.0

    avg_quality = sum(q for _, _, q in pages) / len(pages)

    # For large documents, sample evenly across the doc rather than just
    # taking the first N pages — gives better coverage for fact discovery.
    if len(pages) > _MAX_PAGES:
        step = len(pages) / _MAX_PAGES
        pages = [pages[int(i * step)] for i in range(_MAX_PAGES)]

    return pages, round(avg_quality, 2)


def _assess_text_quality(text: str) -> float:
    """
    Heuristic: ratio of readable characters to total.

    A scanned PDF after OCR often returns garbled text with low readable-char ratio.
    We flag these so downstream consumers can adjust confidence accordingly.
    """
    if not text or len(text) < 30:
        return 0.0
    readable = sum(
        1 for c in text
        if c.isalnum() or c.isspace() or c in ".,;:()-₹%/'\""
    )
    return round(readable / len(text), 2)


# ── Fact Extraction via Gemini ─────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a fact extractor for financial and legal documents.

Extract all meaningful facts from the page text below. Focus on:
- Numerical facts: revenue, profit, headcount, shares, percentages, valuations
- Entity facts: company names, director names, addresses, registration/CIN numbers
- State facts: a person's role or status (active/resigned), a company's status (listed/unlisted)
- Date facts: incorporation dates, filing dates, event dates

For each fact return a JSON object:
{{
  "claim": "clear, self-contained statement of the fact",
  "fact_type": "numerical" | "entity" | "state" | "date",
  "temporal_scope": "FY2023" | "Q1 2024" | "as of March 2024" | null,
  "entity_scope": "standalone" | "consolidated" | null,
  "exact_quote": "verbatim sentence(s) from the text supporting this fact",
  "confidence": 0.0 to 1.0,
  "uncertainty_reason": "brief reason if confidence < 0.75, else null"
}}

Rules:
- Only extract facts explicitly stated — never infer or assume.
- exact_quote must be verbatim from the source text.
- If a number appears without clear context, set confidence below 0.6.
- Skip boilerplate (page headers, table-of-contents entries, legal disclaimers).
- Return ONLY a valid JSON array, no other text. If nothing meaningful, return [].

Page {page_number} text:
{text}"""


def extract_facts_from_page(page_number: int, text: str, page_quality: float) -> List[dict]:
    """
    Ask Gemini to extract facts from a single page.

    We pass pages individually so the LLM can focus and so we always have
    an exact page number to attach to each fact as source evidence.
    A small inter-page delay respects the free-tier rate limit.
    """
    if not text.strip() or page_quality < 0.4:
        # Skip nearly empty or garbled pages — garbage in, garbage out.
        return []

    prompt = EXTRACTION_PROMPT.format(page_number=page_number, text=text[:4000])

    try:
        time.sleep(_INTER_PAGE_DELAY)   # respect free-tier rate limits
        client = get_client()
        response = generate_content_with_fallback(client, prompt)
        facts = parse_json_facts(raw)

        # Adjust confidence downward for low-quality pages
        if page_quality < 0.7:
            for f in facts:
                f["confidence"] = round(f.get("confidence", 0.5) * page_quality, 2)
                if not f.get("uncertainty_reason"):
                    f["uncertainty_reason"] = f"Source page quality is low ({page_quality:.0%})"

        return facts

    except genai_errors.ClientError as e:
        # API-level error (rate limit, auth, etc.) — surface clearly
        return [{
            "claim": f"API error on page {page_number}",
            "fact_type": "state",
            "temporal_scope": None,
            "entity_scope": None,
            "exact_quote": text[:200] if text else "",
            "confidence": 0.0,
            "uncertainty_reason": f"Gemini API error: {str(e)[:120]}",
        }]

    except (json.JSONDecodeError, Exception):
        # Extraction failed — we surface this as an explicit failure case
        return [{
            "claim": f"Extraction failed for page {page_number}",
            "fact_type": "state",
            "temporal_scope": None,
            "entity_scope": None,
            "exact_quote": text[:200] if text else "",
            "confidence": 0.0,
            "uncertainty_reason": "LLM returned unparseable output for this page.",
        }]

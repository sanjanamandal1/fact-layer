import os
import json
import re
import time
import pdfplumber
from google import genai
from google.genai import errors as genai_errors
from typing import List, Tuple

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

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
        response = _client.models.generate_content(model=_MODEL, contents=prompt)
        raw = response.text.strip()

        # Try stripping code fences first
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # Gemini sometimes wraps output in prose — find the JSON array directly
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)

        facts = json.loads(raw)
        if not isinstance(facts, list):
            return []

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

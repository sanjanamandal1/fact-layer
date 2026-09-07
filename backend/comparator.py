"""
Cross-document fact comparison.

Two-phase design:
  Phase 1 — Similarity filter: compute cosine similarity between all fact-pairs
            across documents and short-list candidates above a threshold.
            This avoids an O(n²) LLM calls problem.

  Phase 2 — LLM comparison: ask Gemini to reason about each candidate pair
            and return a structured relationship with explicit reasoning.

Why conservative defaults?
  False positives (calling something a contradiction when it isn't) are far
  worse than false negatives for a banking use-case. We bias toward RECONCILED
  over CONTRADICTED when temporal or scope context could explain a difference.
"""

import os
import json
import re
from google import genai
from google.genai import types, errors as genai_errors
from typing import List, Tuple

from embedder import cosine_similarity, embedding_from_list
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
    from google.genai import errors as genai_errors
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
                import time
                time.sleep(5.0)
                try:
                    response = client.models.generate_content(model=model_name, contents=contents, config=config)
                    _ACTIVE_MODEL = model_name
                    return response
                except Exception:
                    pass
            if "404" in str(e) or "not available" in str(e).lower() or "not found" in str(e).lower():
                last_error = e
                continue
            raise e
    if last_error:
        raise last_error

# Minimum semantic similarity to consider two facts worth comparing.
# Too low → noisy LLM calls; too high → miss paraphrases.
SIMILARITY_THRESHOLD = 0.55


COMPARISON_PROMPT = """You are a senior financial document analyst comparing two facts extracted from different source documents.

Fact A — from "{doc_a}":
  Claim: {claim_a}
  Period: {temporal_a}
  Scope: {scope_a}
  Source quote: "{quote_a}"

Fact B — from "{doc_b}":
  Claim: {claim_b}
  Period: {temporal_b}
  Scope: {scope_b}
  Source quote: "{quote_b}"

Determine the relationship. Choose exactly one:
- CORROBORATED: Both facts say the same thing (wording may differ, but meaning agrees)
- CONTRADICTED: Facts genuinely conflict — same topic, same period, same scope, but different values or claims
- RECONCILED: Appear to conflict, but the difference is explained by context (different fiscal periods, standalone vs consolidated, partial vs full year, different units, different entities)
- UNRELATED: These are about genuinely different topics and should not be compared

Return valid JSON only:
{{
  "relationship": "CORROBORATED" | "CONTRADICTED" | "RECONCILED" | "UNRELATED",
  "reasoning": "2–3 sentences explaining your conclusion, referencing the actual quotes",
  "reconciliation_context": "what context explains the difference (only if RECONCILED, else null)",
  "confidence": 0.0 to 1.0
}}

Be conservative: prefer RECONCILED over CONTRADICTED when temporal or scope differences could explain the gap."""


def find_candidate_pairs(new_facts: List[dict], existing_facts: List[dict]) -> List[Tuple[dict, dict, float]]:
    """
    Find (new_fact, existing_fact, similarity_score) pairs worth comparing.
    Only crosses document boundaries — we never compare a fact to itself.
    """
    candidates = []

    for nf in new_facts:
        if not nf.get("embedding"):
            continue
        emb_n = embedding_from_list(json.loads(nf["embedding"]))

        for ef in existing_facts:
            if ef["document_id"] == nf["document_id"]:
                continue  # skip same-document comparisons
            if not ef.get("embedding"):
                continue

            emb_e = embedding_from_list(json.loads(ef["embedding"]))
            sim = cosine_similarity(emb_n, emb_e)

            if sim >= SIMILARITY_THRESHOLD:
                candidates.append((nf, ef, sim))

    # Sort by similarity descending so we process most-likely pairs first
    return sorted(candidates, key=lambda x: -x[2])


def compare_facts(fact_a: dict, fact_b: dict, doc_a_name: str, doc_b_name: str) -> dict:
    """
    Ask Gemini to reason about the relationship between two facts.
    Returns a structured relationship dict.
    """
    prompt = COMPARISON_PROMPT.format(
        doc_a=doc_a_name,
        claim_a=fact_a["claim"],
        temporal_a=fact_a.get("temporal_scope") or "not specified",
        scope_a=fact_a.get("entity_scope") or "not specified",
        quote_a=fact_a["exact_quote"],
        doc_b=doc_b_name,
        claim_b=fact_b["claim"],
        temporal_b=fact_b.get("temporal_scope") or "not specified",
        scope_b=fact_b.get("entity_scope") or "not specified",
        quote_b=fact_b["exact_quote"],
    )

    try:
        client = get_client()
        response = generate_content_with_fallback(client, prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end+1]

        result = json.loads(raw)

        return {
            "relationship": result.get("relationship", "UNRELATED"),
            "reasoning": result.get("reasoning", ""),
            "reconciliation_context": result.get("reconciliation_context"),
            "confidence": float(result.get("confidence", 0.5)),
        }

    except Exception as e:
        return {
            "relationship": "UNRELATED",
            "reasoning": f"Comparison failed: {str(e)}",
            "reconciliation_context": None,
            "confidence": 0.0,
        }

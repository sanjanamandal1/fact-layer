import os
import uuid
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import db
from extractor import extract_pages, extract_facts_from_document
from embedder import embed, embedding_to_list
from comparator import find_candidate_pairs, compare_facts

# ── App Setup ──────────────────────────────────────────────────────────────

app = FastAPI(title="Fact Knowledge Layer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
db.init_db()

# Serve the frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Full pipeline for a new PDF:
      1. Save to disk
      2. Extract text page-by-page (pdfplumber)
      3. Extract facts per page (Gemini)
      4. Embed each fact (sentence-transformers)
      5. Compare new facts against all existing facts (similarity → Gemini)
      6. Persist everything

    Processing is synchronous — acceptable for a prototype where documents
    are uploaded one at a time. For production, this becomes an async job.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    doc_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}.pdf"

    # Save upload
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract text
    try:
        pages, total_pages, quality_score = extract_pages(str(save_path))
    except Exception as e:
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {str(e)}")

    if not pages:
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Could not extract any readable text from this PDF.")

    db.insert_document(doc_id, file.filename, total_pages, quality_score)

    # Extract + embed facts (single-call whole-document extraction)
    new_facts = []
    try:
        raw_facts = extract_facts_from_document(pages)
        for rf in raw_facts:
            fact_id = str(uuid.uuid4())
            embedding = embed(rf["claim"])
            fact = {
                "id": fact_id,
                "document_id": doc_id,
                "claim": rf["claim"],
                "fact_type": rf.get("fact_type", "entity"),
                "temporal_scope": rf.get("temporal_scope"),
                "entity_scope": rf.get("entity_scope"),
                "exact_quote": rf.get("exact_quote") or rf["claim"],
                "page_number": rf.get("page_number") or (pages[0][0] if pages else 1),
                "confidence": float(rf.get("confidence", 0.85)),
                "uncertainty_reason": rf.get("uncertainty_reason"),
                "embedding": json.dumps(embedding_to_list(embedding)),
            }
            db.insert_fact(fact)
            new_facts.append(fact)
    except RuntimeError as e:
        db.delete_document(doc_id)
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.delete_document(doc_id)
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error extracting facts: {str(e)}")

    db.update_document_fact_count(doc_id, len(new_facts))

    # Cross-document comparison — only if other facts exist
    existing_facts = db.get_all_facts_except_document(doc_id)
    relationships_created = 0

    if existing_facts:
        candidates = find_candidate_pairs(new_facts, existing_facts)[:20]

        # Resolve document names for readable comparison prompts
        doc_cache = {}
        for doc in db.list_documents():
            doc_cache[doc["id"]] = doc["filename"]

        for fact_a, fact_b, _sim in candidates:
            if db.relationship_exists(fact_a["id"], fact_b["id"]):
                continue

            result = compare_facts(
                fact_a, fact_b,
                doc_cache.get(fact_a["document_id"], "Unknown"),
                doc_cache.get(fact_b["document_id"], "Unknown"),
            )

            # Skip UNRELATED — not worth persisting noise
            if result["relationship"] == "UNRELATED":
                continue

            db.insert_relationship({
                "id": str(uuid.uuid4()),
                "fact_a_id": fact_a["id"],
                "fact_b_id": fact_b["id"],
                **result,
            })
            relationships_created += 1

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "pages_processed": total_pages,
        "pages_analyzed": len(pages),
        "quality_score": quality_score,
        "facts_extracted": len(new_facts),
        "relationships_found": relationships_created,
    }


@app.get("/documents")
def list_documents():
    return db.list_documents()


@app.get("/documents/{doc_id}/facts")
def get_document_facts(doc_id: str):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    facts = db.get_facts_for_document(doc_id)
    # Strip embedding from response — it's internal, not useful to display
    for f in facts:
        f.pop("embedding", None)
    return {"document": doc, "facts": facts}


@app.get("/relationships")
def get_relationships():
    rels = db.get_all_relationships()
    return rels


@app.delete("/documents")
def delete_all_documents():
    db.delete_all()
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.glob("*.pdf"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
    return {"status": "all deleted"}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    db.delete_document(doc_id)
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink(missing_ok=True)
    return {"deleted": doc_id}


@app.post("/recompare")
def recompare_all():
    """
    Re-run cross-document comparison across all existing facts.
    Useful after changing the similarity threshold or uploading new docs
    without triggering a comparison (e.g. first doc has no peers to compare).
    Only computes relationships that don't already exist.
    """
    all_docs = db.list_documents()
    if len(all_docs) < 2:
        return {"relationships_found": 0, "message": "Need at least 2 documents to compare."}

    doc_cache = {d["id"]: d["filename"] for d in all_docs}
    all_facts = db.get_all_facts()

    # Group facts by document
    facts_by_doc = {}
    for f in all_facts:
        facts_by_doc.setdefault(f["document_id"], []).append(f)

    doc_ids = list(facts_by_doc.keys())
    relationships_created = 0

    # Compare every pair of documents (both directions covered by find_candidate_pairs)
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            facts_a = facts_by_doc[doc_ids[i]]
            facts_b = facts_by_doc[doc_ids[j]]

            candidates = find_candidate_pairs(facts_a, facts_b)[:20]

            for fact_a, fact_b, _sim in candidates:
                if db.relationship_exists(fact_a["id"], fact_b["id"]):
                    continue

                result = compare_facts(
                    fact_a, fact_b,
                    doc_cache.get(fact_a["document_id"], "Unknown"),
                    doc_cache.get(fact_b["document_id"], "Unknown"),
                )

                if result["relationship"] == "UNRELATED":
                    continue

                db.insert_relationship({
                    "id": str(uuid.uuid4()),
                    "fact_a_id": fact_a["id"],
                    "fact_b_id": fact_b["id"],
                    **result,
                })
                relationships_created += 1

    return {"relationships_found": relationships_created}


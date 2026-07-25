# AI Resume Analyzer — Diagnostic Dashboard

A Flask web app that scores an uploaded CV against multiple job-role profiles and tells you exactly what to do next — not just a percentage, but *why* that percentage, what's missing, and how to close the gap.

Upload a resume (PDF / DOCX / TXT) and get:

- **Role-fit scores** across 10 job profiles (SDE, Data Scientist, Full Stack, Cloud/DevOps, ML Developer, HR, Operations, Business Analyst, QA, and more)
- **Direct job-description matching** — optionally paste or upload a real JD and get a match score against that exact posting instead of a generic profile
- **A role-fit radar chart** plotting every role at once
- **An ATS-style CV health check** — length, quantified impact, action verbs, contact info
- **A personalized "what to study next" roadmap**, prioritized and tied to concrete study tips
- **Plain-language narratives** explaining why a role fits (or doesn't) yet

## How scoring works

Every match score blends three signals:

| Signal | What it catches |
|---|---|
| **Lexical (keyword) matching** | Exact/near-exact keyword hits, weighted by importance per role |
| **TF-IDF similarity** | Statistical term-overlap between your CV and the role/JD text |
| **Semantic matching (Sentence-BERT, zero-shot)** | Meaning-level similarity — catches paraphrases and synonyms a keyword scan would miss (e.g. "led a cross-functional team" ≈ "people management") |

When Sentence-BERT (`all-MiniLM-L6-v2`) is available, the blend is **35% lexical + 20% TF-IDF + 45% semantic**. If the model can't load (missing dependency, no network access to download it, low-memory host), the app automatically falls back to **60% lexical + 40% TF-IDF** and shows a banner explaining that it's running in lexical-only mode. It never crashes because semantic matching is unavailable.

## Tech stack

- **Flask** — web app / routing
- **scikit-learn** — TF-IDF vectorization + cosine similarity
- **sentence-transformers** — zero-shot semantic similarity via Sentence-BERT
- **pypdf** / **python-docx** — text extraction from PDF and DOCX resumes
- No JS frameworks or chart libraries — all visuals (gauges, radar chart) are hand-built inline SVG

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000**.

> The first request that uses semantic matching downloads the `all-MiniLM-L6-v2` model (~90MB) from Hugging Face and caches it in memory for the life of the process — that first analysis will be slower than the rest.

## Deploying (e.g. Render)

- Start command: `gunicorn app:app`
- `sentence-transformers` pulls in PyTorch, which is a heavy dependency. On a memory-constrained free tier, the model may fail to download or load — if that happens, the app keeps working in lexical + TF-IDF mode automatically. If you'd rather not carry the extra weight at all, remove `sentence-transformers` from `requirements.txt`; no code changes needed.

## Project structure

Everything lives in a single `app.py` for easy deployment:

- `ROLE_PROFILES` — the 10 role definitions (keyword → weight)
- `STUDY_TIPS` — the advice shown for each missing skill
- Text extraction (`extract_text_from_*`) → cleaning (`clean_text`) → scoring (`keyword_match_score`, `tfidf_similarity_score`, `sbert_similarity`) → blending (`blend_scores`)
- `analyze_resume()` — scores the CV against every role profile
- `analyze_jd()` — scores the CV against a specific pasted/uploaded job description
- `cv_health_checks()` — the ATS-style health check
- `build_radar()` — computes the SVG radar chart geometry
- Two routes: `/` (upload form) and `/analyze` (POST, returns the report)

## Customizing

- **Add/edit roles or keyword weights** → edit `ROLE_PROFILES`
- **Change study advice** → edit `STUDY_TIPS` (falls back to a generic tip if a keyword isn't listed)
- **Adjust scoring weights** → edit `blend_scores()`
- **Swap the semantic model** → change `SBERT_MODEL_NAME` (any [sentence-transformers model](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) works; smaller/faster models are safer on constrained hosts)

## Privacy

Uploaded files are written to a temporary `uploads/` folder only for the duration of text extraction, then deleted immediately after. Nothing is persisted after the analysis completes.

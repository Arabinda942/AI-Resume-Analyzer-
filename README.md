# AI Resume Analyzer — Diagnostic Dashboard

A Flask web app that scores an uploaded CV against multiple job-role profiles and tells you exactly what to do next — not just a percentage, but *why* that percentage, what's missing, and how to close the gap.

> **Current deployment status:** Semantic (Sentence-BERT) scoring is **disabled** in `requirements.txt` for the live Render deployment. Render's logs confirmed that instance doesn't have enough memory to load PyTorch + sentence-transformers — every attempt was killed mid-load (OOM). The app runs fully and reliably on **lexical + TF-IDF scoring only**; see "Deploying" below for how to re-enable semantic scoring if you move to a higher-memory plan.

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
- **Importing `sentence-transformers` is deferred until the first request that needs it** (not done at boot). This keeps the process's startup memory low and avoids a common failure mode: the import itself pulling in torch at boot can be enough to get the worker OOM-killed on a 512MB free tier *before it ever serves a request*, which shows up as a permanent `502 Bad Gateway` with no useful error in the logs.
- **`DISABLE_SBERT` environment variable** — set this to `1` (or `true`) in Render's dashboard (Environment tab) to skip SBERT/torch entirely, with zero code changes. This is the fastest way to rule SBERT in or out as the cause of a boot crash or 502: set it, redeploy, and if the app comes up healthy, SBERT/memory was the problem.

### Troubleshooting a 502 Bad Gateway on Render

1. Open Render → your service → **Logs**, and look at the **most recent deploy**, not an old one.
2. If the log ends abruptly with no Python traceback (just stops, or shows something like the process exiting) — that's the signature of an **OOM kill**. The OS kills the process before Python gets a chance to log anything.
3. Quick fix to confirm: set `DISABLE_SBERT=1` under Environment, save, and let it redeploy. If the 502 goes away, the cause was memory pressure from loading `sentence-transformers`/torch — you can leave `DISABLE_SBERT` on permanently, or upgrade to a paid instance with more RAM to keep semantic scoring.
4. If you see `bash: line 1: gunicorn: command not found` in the build log, `gunicorn` is missing from `requirements.txt` (it's included here as `gunicorn==22.0.0`) or the start command is misconfigured — it must be exactly `gunicorn app:app`.
5. If the deploy log shows the build succeeding but the app still 502s after a minute or more, check whether a request is timing out — the first request that triggers an SBERT model download can take a while; gunicorn's default worker timeout (30s) can be too short for that. Add a longer timeout to the start command if needed, e.g. `gunicorn app:app --timeout 120`.
6. If the deploy log ends with **"Port scan timeout reached, no open ports detected"**, the container took too long to become ready. A common cause: `pip install` pulled the full GPU build of PyTorch, which drags in several GB of unused NVIDIA/CUDA packages (cuDNN, cuBLAS, NCCL, Triton, etc.) alongside it, since Render's instances are CPU-only and never touch a GPU. `requirements.txt` now pins the CPU-only build via `--extra-index-url https://download.pytorch.org/whl/cpu`, which is a small fraction of the size and skips all the CUDA packages — this should resolve a port-scan timeout on its own. If it still times out after that, fall back to `DISABLE_SBERT=1` (step 3) while you investigate further.
7. If the **build/boot succeeds but clicking "Analyze" shows a generic "Internal Server Error"** page (rather than a 502), that means the app itself hit an unhandled exception while scoring or rendering the report. The app now catches this: it logs the **full Python traceback** and shows the visitor a friendly error message instead of the raw error page. To find the actual cause, go to Render → your service → **Logs** (the live/runtime logs, not the build log), click "Analyze" again, and look for a line starting with `[analyze] Unhandled error...` — the traceback right below it is the real error. Paste that here and it can be fixed directly.
8. **A note on adding/changing environment variables on Render:** saving a new environment variable automatically triggers a fresh deploy — it re-runs the *entire* build (reinstalling everything from `requirements.txt`) rather than just restarting the app. So if the deploy fails right after adding `DISABLE_SBERT`, it is very likely hitting the *same* underlying build issue described in steps 6–7 (e.g. an old commit still using the GPU torch build), not a problem with the environment variable itself. Check that the repo actually has the current `requirements.txt` (CPU-only torch) and `app.py` pushed *before* adding the variable, then check the resulting build log the same way as any other deploy.

### Verifying SBERT status from the logs (no analysis needed)

The app now logs its SBERT status explicitly, so you can check whether semantic scoring is active straight from the Render (or any host's) logs, without uploading a CV:

- **At startup / import time** — one line, immediately:
  - `[SBERT] Import OK — 'sentence_transformers' is installed. Model will be lazy-loaded on first analysis request.`
  - or `[SBERT] Import FAILED — 'sentence_transformers' is not installed or could not be imported (...)`. This means the dependency itself never made it into the deployed environment — check the build log for whether `sentence-transformers`/`torch` actually installed.
- **On first `/analyze` request that needs it** — the model loads lazily, so you'll see:
  - `[SBERT] Loading model 'all-MiniLM-L6-v2' for the first time in this worker ...`
  - then either `[SBERT] Model 'all-MiniLM-L6-v2' loaded successfully in X.Xs — semantic scoring is ACTIVE for this worker.`
  - or `[SBERT] Model load FAILED after X.Xs — falling back to LEXICAL + TF-IDF scoring only ...` with a full traceback (common causes: OOM kill on a low-memory host, no network access to Hugging Face, disk space limits).
- **Every `/analyze` request** also logs one line confirming which mode actually scored that request: `SEMANTIC (lexical + TF-IDF + SBERT)` or `FALLBACK (lexical + TF-IDF only)` — so you can correlate a specific upload with what ran, without needing to inspect the report's banner/pills in the browser.

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

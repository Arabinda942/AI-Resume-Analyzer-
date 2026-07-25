"""
AI Resume Analyzer — Diagnostic Dashboard v3
------------------------------------------
Upload a CV (PDF / DOCX / TXT). The app scores it against multiple job-role
profiles and returns:
  - a % match per role (keyword coverage + TF-IDF similarity blend)
  - a plain-language "fit" verdict per role
  - which skills from your CV already matched a role
  - which skills are missing, with a short study tip for each
  - a combined "what to study next" roadmap built from your top-fit roles
  - a short narrative explaining *why* your top role is a good target
  - a radar chart plotting your fit across every role at a glance
  - an ATS-style CV health check (length, metrics, action verbs, contact info)

Optionally, paste or upload a specific Job Description alongside your CV to get:
  - a direct match % against that exact posting
  - the JD's own key terms, split into what you already show and what's missing
  - a tailored narrative on why (or why not yet) you're a fit for that job

Run:
    pip install -r requirements.txt
    python app.py
Then open: http://127.0.0.1:5000

Edit ROLE_PROFILES to add/remove roles or tune keyword weights for your org.
Edit STUDY_TIPS to change what advice is shown for a given missing skill.
"""

import os
import re
import math
from collections import defaultdict

from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

import pypdf
import docx

# Sentence-BERT is optional: it's a heavy dependency (pulls in torch), so the
# app must keep working with lexical-only scoring if it isn't installed or
# fails to load (e.g. on a memory-constrained host). Everything below checks
# SBERT_IMPORT_OK / get_sbert_model() before using it.
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SBERT_IMPORT_OK = True
except ImportError:
    SBERT_IMPORT_OK = False

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB (CV + optional JD file)

# Gauge geometry (SVG circles) — precomputed once
CARD_R = 46
CARD_CIRC = round(2 * math.pi * CARD_R, 2)
HERO_R = 90
HERO_CIRC = round(2 * math.pi * HERO_R, 2)

# Sentence-BERT — used as a zero-shot semantic matcher (no fine-tuning, no
# labeled training data: just embedding cosine similarity). Small model so it
# stays usable on a CPU-only, low-RAM host. Loaded lazily on first use.
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
_sbert_model = None
_sbert_load_failed = False
_role_embedding_cache = {}

# Radar chart geometry
RADAR_SIZE = 380
RADAR_CX = RADAR_SIZE / 2
RADAR_CY = RADAR_SIZE / 2
RADAR_R = 118
RADAR_AXES = 8  # how many roles to plot

SHORT_ROLE_NAMES = {
    "HR Management": "HR Mgmt",
    "Operations": "Operations",
    "DSA / SDE": "DSA / SDE",
    "ML Developer": "ML Dev",
    "Software Development (SD)": "SW Dev",
    "Data Scientist": "Data Sci",
    "Full Stack Developer": "Full Stack",
    "Cloud / DevOps Engineer": "Cloud/DevOps",
    "Business Analyst": "Biz Analyst",
    "QA / Testing": "QA/Testing",
}

ACTION_VERBS = {
    "led", "managed", "built", "developed", "designed", "implemented", "created",
    "improved", "increased", "reduced", "launched", "optimized", "automated",
    "architected", "delivered", "spearheaded", "coordinated", "analyzed",
    "achieved", "streamlined", "drove", "established", "mentored", "owned",
    "scaled", "shipped", "resolved", "negotiated", "presented", "trained",
}

# ----------------------------------------------------------------------
# ROLE PROFILES
# Each role = dict of {keyword: weight}. Higher weight = more important.
# ----------------------------------------------------------------------
ROLE_PROFILES = {
    "HR Management": {
        "recruitment": 3, "talent acquisition": 3, "onboarding": 2,
        "employee relations": 3, "performance management": 2,
        "hr policies": 2, "payroll": 2, "compliance": 2,
        "hris": 2, "exit interview": 1, "training and development": 2,
        "conflict resolution": 2, "labour law": 2, "workforce planning": 2,
    },
    "Operations": {
        "process improvement": 3, "operations management": 3, "sla": 2,
        "kpi": 2, "vendor management": 2, "supply chain": 2,
        "six sigma": 2, "quality control": 2, "stakeholder management": 2,
        "workflow optimization": 3, "resource planning": 2, "logistics": 1,
        "cost reduction": 2,
    },
    "DSA / SDE": {
        "data structures": 3, "algorithms": 3, "competitive programming": 2,
        "time complexity": 2, "dynamic programming": 2, "graph": 2,
        "tree": 2, "sorting": 1, "searching": 1, "recursion": 1,
        "c++": 2, "java": 2, "python": 2, "leetcode": 1, "problem solving": 2,
    },
    "ML Developer": {
        "machine learning": 3, "deep learning": 3, "tensorflow": 2,
        "pytorch": 2, "scikit-learn": 2, "nlp": 2, "computer vision": 2,
        "feature engineering": 2, "neural networks": 2, "model deployment": 2,
        "mlops": 2, "python": 2, "keras": 1, "opencv": 1,
    },
    "Software Development (SD)": {
        "software development": 3, "java": 2, "python": 2, "c++": 2,
        "git": 2, "agile": 2, "rest api": 2, "microservices": 2,
        "oop": 2, "design patterns": 2, "ci/cd": 2, "unit testing": 2,
        "debugging": 1, "docker": 1, "kubernetes": 1,
    },
    "Data Scientist": {
        "data analysis": 3, "statistics": 3, "python": 2, "r": 1,
        "sql": 2, "machine learning": 2, "data visualization": 2,
        "pandas": 2, "numpy": 1, "tableau": 1, "power bi": 1,
        "hypothesis testing": 2, "big data": 2, "a/b testing": 1,
        "regression": 2, "clustering": 1,
    },
    "Full Stack Developer": {
        "html": 2, "css": 2, "javascript": 3, "react": 3, "node.js": 3,
        "express": 2, "mongodb": 2, "sql": 2, "rest api": 2, "git": 2,
        "frontend": 2, "backend": 2, "api integration": 2, "typescript": 1,
        "next.js": 1,
    },
    "Cloud / DevOps Engineer": {
        "aws": 3, "azure": 2, "gcp": 2, "docker": 3, "kubernetes": 3,
        "ci/cd": 3, "terraform": 2, "jenkins": 2, "linux": 2,
        "cloud infrastructure": 2, "monitoring": 1, "ansible": 1,
        "networking": 1,
    },
    "Business Analyst": {
        "requirement gathering": 3, "business analysis": 3, "sql": 2,
        "stakeholder management": 2, "process mapping": 2, "excel": 2,
        "user stories": 2, "agile": 2, "documentation": 1,
        "data analysis": 2, "gap analysis": 2, "power bi": 1,
    },
    "QA / Testing": {
        "manual testing": 2, "automation testing": 3, "selenium": 3,
        "test cases": 2, "bug tracking": 2, "qa": 2, "regression testing": 2,
        "api testing": 2, "jira": 2, "test plan": 2, "sql": 1,
    },
}

# ----------------------------------------------------------------------
# STUDY TIPS — short, generic, actionable advice per skill.
# Falls back to a generic tip if a keyword isn't listed here.
# ----------------------------------------------------------------------
STUDY_TIPS = {
    "recruitment": "Learn the end-to-end hiring funnel: sourcing, screening, interviewing, offer rollout.",
    "talent acquisition": "Study sourcing channels and employer-branding basics; try mock sourcing exercises.",
    "onboarding": "Design a sample 30/60/90-day onboarding plan for a fictional new hire.",
    "employee relations": "Read up on grievance handling and conflict-resolution frameworks (e.g. interest-based negotiation).",
    "performance management": "Practice writing OKRs/KPIs and a sample performance review cycle.",
    "hr policies": "Review a public company handbook to learn how policies are structured.",
    "payroll": "Learn payroll cycle basics: deductions, compliance, and common payroll software concepts.",
    "compliance": "Study local labour law basics and common statutory compliance checklists.",
    "hris": "Get hands-on with a free/demo HRIS tool (e.g. Zoho People, BambooHR trial) to learn the workflows.",
    "training and development": "Design a sample training needs-analysis and a short learning module.",
    "conflict resolution": "Practice structured conflict-resolution scripts and mediation basics.",
    "labour law": "Read a summary of key labour law provisions relevant to your country.",
    "workforce planning": "Learn headcount forecasting basics using a simple spreadsheet model.",
    "process improvement": "Learn a lightweight framework like PDCA or Kaizen and apply it to a real process.",
    "operations management": "Study core ops metrics: throughput, utilization, and cycle time.",
    "sla": "Learn how SLAs are defined, tracked, and reported against.",
    "kpi": "Practice building a KPI dashboard for a sample process in Excel/Sheets.",
    "vendor management": "Learn vendor scorecards and basic contract/SLA negotiation concepts.",
    "supply chain": "Study the core supply-chain stages: procurement, inventory, logistics, delivery.",
    "six sigma": "Take a free intro course on Six Sigma/Lean basics (DMAIC framework).",
    "quality control": "Learn basic QC tools: checklists, control charts, root-cause analysis.",
    "stakeholder management": "Practice mapping stakeholders by influence/interest for a sample project.",
    "workflow optimization": "Map a real process end-to-end and identify 2-3 bottlenecks to remove.",
    "resource planning": "Practice capacity planning using a simple resource-allocation spreadsheet.",
    "logistics": "Learn the basics of warehousing, transportation, and last-mile delivery.",
    "cost reduction": "Study common cost-reduction levers: process automation, vendor renegotiation, waste elimination.",
    "data structures": "Practice arrays, linked lists, stacks, queues, trees, and graphs on a coding platform.",
    "algorithms": "Practice sorting, searching, and greedy/DP problems regularly (aim for daily reps).",
    "competitive programming": "Join contests on platforms like Codeforces or LeetCode weekly.",
    "time complexity": "Practice Big-O analysis on your own solved problems.",
    "dynamic programming": "Solve classic DP problems (knapsack, LIS, edit distance) to build pattern recognition.",
    "graph": "Practice BFS/DFS, shortest path, and topological sort problems.",
    "tree": "Practice tree traversals, BSTs, and balancing concepts.",
    "sorting": "Re-implement common sorts by hand to understand trade-offs.",
    "searching": "Practice binary search variants until they're second nature.",
    "recursion": "Practice recursive problems and always trace the call stack by hand at least once.",
    "c++": "Build a small project in C++ covering OOP, STL containers, and memory management.",
    "java": "Build a small Java project using OOP principles and collections.",
    "python": "Build a small end-to-end Python project (script, tests, and a README).",
    "leetcode": "Solve problems consistently, tracking patterns, not just problem count.",
    "problem solving": "Practice breaking problems into smaller sub-problems before coding.",
    "machine learning": "Complete one full ML project: data cleaning, model training, evaluation, write-up.",
    "deep learning": "Build a small neural network from scratch, then repeat it in TensorFlow/PyTorch.",
    "tensorflow": "Follow the official TensorFlow quickstart and rebuild it without copy-pasting.",
    "pytorch": "Follow the official PyTorch tutorials and reproduce one end-to-end.",
    "scikit-learn": "Practice the full scikit-learn pipeline: preprocessing, model, cross-validation.",
    "nlp": "Build a small text-classification or sentiment-analysis project.",
    "computer vision": "Build a simple image-classification project using a pretrained model.",
    "feature engineering": "Practice creating and evaluating new features on a public dataset.",
    "neural networks": "Learn forward/backward propagation by implementing a tiny network manually.",
    "model deployment": "Deploy one trained model behind a simple API (e.g. Flask/FastAPI).",
    "mlops": "Learn the basics of experiment tracking and model versioning (e.g. MLflow).",
    "keras": "Rebuild a Keras example model and modify its architecture yourself.",
    "opencv": "Build a small image-processing script using OpenCV (edge detection, filters).",
    "software development": "Build and ship one complete small application end-to-end.",
    "git": "Practice branching, merging, and resolving conflicts on a personal repo.",
    "agile": "Learn Scrum basics: sprints, standups, backlog grooming, retrospectives.",
    "rest api": "Build a simple REST API and consume it from a small client app.",
    "microservices": "Read about microservices vs monoliths and sketch a simple service-split for a sample app.",
    "oop": "Revisit OOP fundamentals: encapsulation, inheritance, polymorphism, abstraction — with code examples.",
    "design patterns": "Learn 3-4 common patterns (Singleton, Factory, Observer) and apply one in a project.",
    "ci/cd": "Set up a basic CI/CD pipeline (e.g. GitHub Actions) for a personal repo.",
    "unit testing": "Add unit tests to an existing personal project using a standard testing framework.",
    "debugging": "Practice using a debugger (breakpoints, stepping) instead of print statements.",
    "docker": "Containerize a small app with Docker and run it locally.",
    "kubernetes": "Learn core Kubernetes concepts (pods, services, deployments) via a local cluster (minikube).",
    "data analysis": "Practice cleaning and analyzing a public dataset end-to-end in Python/Excel.",
    "statistics": "Review core stats: distributions, hypothesis testing, confidence intervals.",
    "r": "Complete a small analysis project in R using tidyverse.",
    "sql": "Practice joins, subqueries, and window functions on a public SQL practice dataset.",
    "data visualization": "Build 3-4 different chart types for the same dataset and note which best fit the story.",
    "pandas": "Practice groupby, merge, and pivot operations on a real dataset.",
    "numpy": "Practice vectorized operations and broadcasting instead of loops.",
    "tableau": "Recreate one public dashboard in Tableau to learn its core building blocks.",
    "power bi": "Build a small interactive report in Power BI using a public dataset.",
    "hypothesis testing": "Practice designing and interpreting a t-test or chi-square test on sample data.",
    "big data": "Learn the basics of distributed processing (e.g. Spark) with a small local example.",
    "a/b testing": "Design a mock A/B test end-to-end: hypothesis, metric, sample size, result interpretation.",
    "regression": "Practice linear/logistic regression on a public dataset and interpret the coefficients.",
    "clustering": "Practice k-means/hierarchical clustering and evaluate with silhouette score.",
    "html": "Build a small semantic HTML page from scratch without a framework.",
    "css": "Practice flexbox and grid layouts by rebuilding a real website's layout.",
    "javascript": "Build a small interactive page using vanilla JS (DOM manipulation, events, fetch).",
    "react": "Build a small multi-component app in React using hooks and props.",
    "node.js": "Build a small backend service in Node.js with routing and basic middleware.",
    "express": "Build a REST API using Express with proper route structure and error handling.",
    "mongodb": "Practice CRUD operations and basic aggregation pipelines in MongoDB.",
    "frontend": "Build and deploy one small, fully responsive front-end project.",
    "backend": "Build a small backend with auth, database, and at least 3 endpoints.",
    "api integration": "Integrate a public third-party API into a small personal project.",
    "typescript": "Convert an existing small JS project to TypeScript to learn type safety.",
    "next.js": "Build a small Next.js app using routing and server-side rendering.",
    "aws": "Get hands-on with EC2, S3, and IAM basics using the AWS free tier.",
    "azure": "Get hands-on with Azure VMs and storage using the free tier.",
    "gcp": "Get hands-on with GCP Compute Engine and Cloud Storage using the free tier.",
    "terraform": "Write a small Terraform script to provision one resource end-to-end.",
    "jenkins": "Set up a simple Jenkins pipeline for a personal project.",
    "linux": "Practice core shell commands, permissions, and process management.",
    "cloud infrastructure": "Design a simple architecture diagram for a small app and explain each component's role.",
    "monitoring": "Set up basic monitoring/alerting (e.g. with a free tier tool) for a personal project.",
    "ansible": "Write a simple Ansible playbook to automate one repetitive setup task.",
    "networking": "Review core networking concepts: DNS, TCP/IP, load balancing basics.",
    "requirement gathering": "Practice writing a requirements document from a sample stakeholder brief.",
    "business analysis": "Practice a SWOT or gap analysis on a real (or sample) business case.",
    "process mapping": "Map an existing process using a simple flowchart tool.",
    "excel": "Practice pivot tables, lookups, and basic macros on a sample dataset.",
    "user stories": "Practice writing user stories with clear acceptance criteria for a sample feature.",
    "documentation": "Write clear documentation for a personal project as if handing it to a new teammate.",
    "gap analysis": "Practice comparing a current-state vs desired-state process and list the gaps.",
    "manual testing": "Write a full manual test-case suite for a small sample application.",
    "automation testing": "Automate a few manual test cases using a standard automation tool.",
    "selenium": "Build a small automated test suite using Selenium WebDriver.",
    "test cases": "Practice writing clear, reproducible test cases with expected results.",
    "bug tracking": "Practice logging bugs with clear repro steps in a tool like Jira.",
    "qa": "Learn the QA lifecycle: planning, case design, execution, defect tracking, sign-off.",
    "regression testing": "Build a small regression test suite and practice running it after each change.",
    "api testing": "Practice testing REST APIs using a tool like Postman, including edge cases.",
    "jira": "Practice creating and managing a small backlog and sprint board in Jira.",
    "test plan": "Write a full test plan for a small sample feature, including scope and risks.",
}
GENERIC_TIP = "Build a small hands-on project using {kw} and add it to your portfolio or GitHub."


# ----------------------------------------------------------------------
# TEXT EXTRACTION
# ----------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_txt(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()


def extract_text(path, filename):
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(path)
    elif ext == "docx":
        return extract_text_from_docx(path)
    elif ext == "txt":
        return extract_text_from_txt(path)
    return ""


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+./\s#-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------------------------------------------------------
# SCORING + ANALYSIS
# ----------------------------------------------------------------------
def keyword_match_score(cv_text, keywords_dict):
    total_weight = sum(keywords_dict.values())
    matched_weight = 0
    matched, missing = [], []
    for kw, weight in keywords_dict.items():
        if re.search(re.escape(kw), cv_text):
            matched_weight += weight
            matched.append((kw, weight))
        else:
            missing.append((kw, weight))
    score = (matched_weight / total_weight) * 100 if total_weight else 0
    matched.sort(key=lambda x: -x[1])
    missing.sort(key=lambda x: -x[1])
    return round(score, 2), matched, missing


def tfidf_similarity_score(cv_text, role_text):
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([cv_text, role_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(sim * 100, 2)
    except ValueError:
        return 0.0


def get_sbert_model():
    """Lazy-load the SBERT model once per worker process. Returns None (and
    stops retrying) if it can't be loaded, so the rest of the app falls back
    to lexical-only scoring instead of crashing."""
    global _sbert_model, _sbert_load_failed
    if not SBERT_IMPORT_OK or _sbert_load_failed:
        return None
    if _sbert_model is None:
        try:
            _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        except Exception:
            _sbert_load_failed = True
            return None
    return _sbert_model


def sbert_embed(text):
    """Encode a single piece of text to a normalized embedding, or None if
    the model isn't available. Text is capped since MiniLM only attends to
    ~256 tokens anyway — encoding more just costs time for no extra signal."""
    model = get_sbert_model()
    if model is None or not text.strip():
        return None
    try:
        return model.encode(text[:4000], normalize_embeddings=True)
    except Exception:
        return None


def sbert_role_embedding(role_name, role_text):
    """Cached embedding for a role profile's keyword text — static per run,
    so we only encode each role once instead of once per uploaded CV."""
    if role_name not in _role_embedding_cache:
        _role_embedding_cache[role_name] = sbert_embed(role_text)
    return _role_embedding_cache[role_name]


def sbert_similarity(embedding_a, embedding_b):
    """Cosine similarity between two pre-normalized embeddings, as a 0-100
    score. This is the 'zero-shot' half of the match: no keywords, no
    training on resumes — just semantic closeness of meaning."""
    if embedding_a is None or embedding_b is None:
        return None
    sim = float(np.dot(embedding_a, embedding_b))
    return round(max(sim, 0.0) * 100, 2)


def blend_scores(kw_score, tfidf_score, sbert_score):
    """Combine lexical keyword coverage, TF-IDF statistical overlap, and
    SBERT semantic similarity into one final score. SBERT gets the largest
    weight when available since it catches paraphrased/synonym matches that
    pure keyword matching misses (e.g. 'led a team' vs 'people management')."""
    if sbert_score is not None:
        return round(0.35 * kw_score + 0.20 * tfidf_score + 0.45 * sbert_score, 2)
    return round(0.6 * kw_score + 0.4 * tfidf_score, 2)


def fit_label(score):
    if score >= 70:
        return "Strong Fit", "You're well aligned with this role."
    elif score >= 45:
        return "Good Fit", "You meet a solid portion of what's expected."
    elif score >= 25:
        return "Developing Fit", "You have some relevant ground, but notable gaps remain."
    else:
        return "Early Stage", "This role would need significant upskilling before applying."


def build_narrative(role, score, matched, missing):
    label, subtext = fit_label(score)
    top_matched = [m[0] for m in matched[:3]]
    top_missing = [m[0] for m in missing[:3]]

    if top_matched:
        strength_part = ("Your CV already shows strength in " + ", ".join(top_matched) +
                          f" — these map directly to day-to-day work in {role}.")
    else:
        strength_part = f"Your CV doesn't yet show clear evidence of the core skills {role} looks for."

    if top_missing:
        gap_part = f" To move past {score}%, focus next on: {', '.join(top_missing)}."
    else:
        gap_part = " You're covering nearly everything this role profile looks for."

    return f"{subtext} {strength_part}{gap_part}"


def build_roadmap(results, jd_analysis=None, top_n_roles=3, max_items=8):
    """Combine missing skills across the top N roles (and the target JD, if given)
    into one prioritized study list. JD-specific gaps are weighted higher since
    they're tied to a real posting, not a generic profile."""
    weight_acc = defaultdict(float)
    for r in results[:top_n_roles]:
        for kw, weight in r["missing"]:
            weight_acc[kw] += weight

    if jd_analysis:
        for kw, weight in jd_analysis["missing"]:
            weight_acc[kw] += weight * 2.5  # prioritize gaps tied to the real job posting

    ranked = sorted(weight_acc.items(), key=lambda x: -x[1])[:max_items]
    roadmap = []
    for kw, _ in ranked:
        tip = STUDY_TIPS.get(kw, GENERIC_TIP.format(kw=kw))
        roadmap.append({"skill": kw, "tip": tip, "from_jd": bool(jd_analysis and kw in dict(jd_analysis["missing"]))})
    return roadmap


def extract_jd_keywords(jd_text_clean, top_n=18):
    """Pull the most frequent meaningful terms (1-2 word phrases) out of a
    pasted/uploaded job description, so we have something concrete to check
    the CV against even though the JD wasn't written as a keyword list."""
    if not jd_text_clean.strip():
        return {}
    try:
        vectorizer = CountVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=150, min_df=1
        )
        matrix = vectorizer.fit_transform([jd_text_clean])
    except ValueError:
        return {}

    freqs = matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()
    pairs = sorted(zip(terms, freqs), key=lambda x: -x[1])

    keywords = {}
    for term, freq in pairs:
        if freq < 1:
            continue
        if term.isdigit() or len(term) < 3:
            continue
        if all(w in ENGLISH_STOP_WORDS for w in term.split()):
            continue
        keywords[term] = int(freq)
        if len(keywords) >= top_n:
            break
    return keywords


def analyze_jd(cv_text_clean, jd_raw_text, cv_embedding=None):
    """Compare the CV directly against one specific job description, blending
    lexical keyword coverage with SBERT zero-shot semantic similarity between
    the full CV and the full JD text."""
    jd_text_clean = clean_text(jd_raw_text)
    jd_keywords = extract_jd_keywords(jd_text_clean)

    if jd_keywords:
        kw_score, matched, missing = keyword_match_score(cv_text_clean, jd_keywords)
    else:
        kw_score, matched, missing = 0.0, [], []

    tfidf_score = tfidf_similarity_score(cv_text_clean, jd_text_clean)

    sbert_score = None
    if cv_embedding is not None:
        jd_embedding = sbert_embed(jd_text_clean)
        sbert_score = sbert_similarity(cv_embedding, jd_embedding)

    final_score = min(blend_scores(kw_score, tfidf_score, sbert_score), 100.0)
    label, subtext = fit_label(final_score)
    narrative = build_narrative("this specific role", final_score, matched, missing)

    missing_with_tips = [
        {"skill": kw, "tip": STUDY_TIPS.get(kw, GENERIC_TIP.format(kw=kw))}
        for kw, _ in missing[:8]
    ]

    return {
        "final_score": final_score,
        "keyword_score": kw_score,
        "tfidf_score": tfidf_score,
        "sbert_score": sbert_score,
        "fit_label": label,
        "fit_subtext": subtext,
        "narrative": narrative,
        "matched_names": [m[0] for m in matched],
        "missing": missing,
        "missing_top": missing_with_tips,
        "term_count": len(jd_keywords),
        "card_offset": round(CARD_CIRC * (1 - final_score / 100), 2),
    }


def cv_health_checks(raw_text, cv_text_clean):
    """Lightweight, explainable ATS-style checks — not a black box score,
    just the concrete things a recruiter or parser would notice first."""
    word_count = len(raw_text.split())
    quantified = re.findall(r"\d+(?:\.\d+)?\s?%|[$₹]\s?\d[\d,]*|\b\d{2,}\+?\b", raw_text)
    words = re.findall(r"[a-z]+", cv_text_clean)
    verb_hits = sum(1 for w in words if w in ACTION_VERBS)
    has_email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", raw_text))
    has_phone = bool(re.search(r"(\+?\d[\d\-\s()]{8,}\d)", raw_text))
    est_pages = max(1, round(word_count / 500))

    checks = []

    if word_count < 200:
        checks.append({"label": "Length", "status": "warn",
                        "detail": f"Only {word_count} words — likely too thin for most recruiters. Add more detail on scope and impact."})
    elif word_count > 1100:
        checks.append({"label": "Length", "status": "warn",
                        "detail": f"{word_count} words (~{est_pages} pages) — trim toward 1-2 pages for most roles below senior level."})
    else:
        checks.append({"label": "Length", "status": "ok",
                        "detail": f"{word_count} words (~{est_pages} page{'s' if est_pages != 1 else ''}) — a reasonable length."})

    if len(quantified) < 3:
        checks.append({"label": "Quantified impact", "status": "warn",
                        "detail": f"Only {len(quantified)} numeric result(s) found. Add metrics (%, counts, time saved) to back up your bullet points."})
    else:
        checks.append({"label": "Quantified impact", "status": "ok",
                        "detail": f"{len(quantified)} quantified results found — good use of concrete numbers."})

    if verb_hits < 5:
        checks.append({"label": "Action verbs", "status": "warn",
                        "detail": f"Only {verb_hits} strong action verbs detected. Start bullets with verbs like 'led', 'built', 'launched' instead of 'responsible for'."})
    else:
        checks.append({"label": "Action verbs", "status": "ok",
                        "detail": f"{verb_hits} strong action verbs detected — bullets read as active, not passive."})

    missing_contact = []
    if not has_email:
        missing_contact.append("email")
    if not has_phone:
        missing_contact.append("phone number")
    if missing_contact:
        checks.append({"label": "Contact info", "status": "warn",
                        "detail": f"Couldn't detect a {' or '.join(missing_contact)} — make sure it's plain text, not an image, so ATS systems can read it."})
    else:
        checks.append({"label": "Contact info", "status": "ok",
                        "detail": "Email and phone number both detected in plain text."})

    return checks


def build_radar(results, n=RADAR_AXES):
    """Plot final_score for the top N roles as a radar/spider chart so the
    whole fit picture is visible at a glance, not just one gauge at a time."""
    top = results[:n]
    count = len(top)
    if count < 3:
        return None

    angle_step = 2 * math.pi / count
    axis_lines = []
    roles = []

    for i, r in enumerate(top):
        angle = -math.pi / 2 + i * angle_step
        radius = RADAR_R * (r["final_score"] / 100)
        x = round(RADAR_CX + radius * math.cos(angle), 1)
        y = round(RADAR_CY + radius * math.sin(angle), 1)

        ax = round(RADAR_CX + RADAR_R * math.cos(angle), 1)
        ay = round(RADAR_CY + RADAR_R * math.sin(angle), 1)
        axis_lines.append((RADAR_CX, RADAR_CY, ax, ay))

        lx = round(RADAR_CX + (RADAR_R + 34) * math.cos(angle), 1)
        ly = round(RADAR_CY + (RADAR_R + 34) * math.sin(angle), 1)
        cos_a = math.cos(angle)
        anchor = "middle"
        if cos_a > 0.35:
            anchor = "start"
        elif cos_a < -0.35:
            anchor = "end"

        roles.append({
            "role": r["role"],
            "short": SHORT_ROLE_NAMES.get(r["role"], r["role"]),
            "final_score": r["final_score"],
            "point": f"{x},{y}",
            "label_x": lx,
            "label_y": ly,
            "anchor": anchor,
        })

    rings = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        pts = []
        for i in range(count):
            angle = -math.pi / 2 + i * angle_step
            x = round(RADAR_CX + RADAR_R * frac * math.cos(angle), 1)
            y = round(RADAR_CY + RADAR_R * frac * math.sin(angle), 1)
            pts.append(f"{x},{y}")
        rings.append(" ".join(pts))

    polygon = " ".join(r["point"] for r in roles)

    return {
        "size": RADAR_SIZE,
        "cx": RADAR_CX,
        "cy": RADAR_CY,
        "polygon": polygon,
        "rings": rings,
        "axis_lines": axis_lines,
        "roles": roles,
    }


def analyze_resume(cv_text):
    cv_text_clean = clean_text(cv_text)
    cv_embedding = sbert_embed(cv_text_clean)  # None if SBERT unavailable — everything below handles that
    results = []

    for role, keywords in ROLE_PROFILES.items():
        role_text = clean_text(" ".join(keywords.keys()))
        kw_score, matched, missing = keyword_match_score(cv_text_clean, keywords)
        tfidf_score = tfidf_similarity_score(cv_text_clean, role_text)

        sbert_score = None
        if cv_embedding is not None:
            role_embedding = sbert_role_embedding(role, role_text)
            sbert_score = sbert_similarity(cv_embedding, role_embedding)

        final_score = min(blend_scores(kw_score, tfidf_score, sbert_score), 100.0)
        label, subtext = fit_label(final_score)

        card_offset = round(CARD_CIRC * (1 - final_score / 100), 2)
        hero_offset = round(HERO_CIRC * (1 - final_score / 100), 2)

        top_missing_with_tips = [
            {"skill": kw, "tip": STUDY_TIPS.get(kw, GENERIC_TIP.format(kw=kw))}
            for kw, _ in missing[:5]
        ]

        results.append({
            "role": role,
            "final_score": final_score,
            "keyword_score": kw_score,
            "tfidf_score": tfidf_score,
            "sbert_score": sbert_score,
            "matched": matched,
            "missing": missing,
            "matched_names": [m[0] for m in matched],
            "missing_top": top_missing_with_tips,
            "total_keywords": len(keywords),
            "fit_label": label,
            "fit_subtext": subtext,
            "card_offset": card_offset,
            "hero_offset": hero_offset,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    for r in results:
        r["narrative"] = build_narrative(r["role"], r["final_score"], r["matched"], r["missing"])

    health = cv_health_checks(cv_text, cv_text_clean)
    return results, health, cv_embedding


# ----------------------------------------------------------------------
# TEMPLATES
# ----------------------------------------------------------------------
BASE_STYLE = """
<style>
  :root{
    --bg:#050912; --panel:#0E1526; --panel-2:#0A1020;
    --text:#F2F5FB; --muted:#8CA0C2;
    --gold:#D9B44A; --gold-soft:rgba(217,180,74,.14);
    --blue:#3E7BFA; --blue-soft:rgba(62,123,250,.14);
    --danger:#E0556F; --danger-soft:rgba(224,85,111,.12);
    --border:#1B2740;
    --radius:14px;
    --font-display:'Space Grotesk',sans-serif;
    --font-body:'IBM Plex Sans',sans-serif;
    --font-mono:'IBM Plex Mono',monospace;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; background:
      radial-gradient(1100px 520px at 12% -8%, rgba(62,123,250,.10), transparent 60%),
      radial-gradient(900px 460px at 110% 10%, rgba(217,180,74,.08), transparent 55%),
      var(--bg);
    color:var(--text);
    font-family:var(--font-body); line-height:1.5;
  }
  a{color:var(--blue);}
  .wrap{max-width:1040px; margin:0 auto; padding:48px 24px 80px;}
  .eyebrow{
    font-family:var(--font-mono); letter-spacing:.14em; text-transform:uppercase;
    font-size:12px; color:var(--gold); margin:0 0 10px;
  }
  h1{
    font-family:var(--font-display); font-size:clamp(28px,4vw,42px);
    margin:0 0 8px; letter-spacing:-.01em;
  }
  .lede{color:var(--muted); font-size:16px; margin:0 0 32px; max-width:62ch;}
  .panel{
    background:var(--panel); border:1px solid var(--border);
    border-radius:var(--radius); padding:28px;
  }
  :focus-visible{outline:2px solid var(--blue); outline-offset:2px;}
  @media (prefers-reduced-motion:reduce){ *{animation:none !important; transition:none !important;} }
</style>
"""

UPLOAD_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Resume Analyzer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
""" + BASE_STYLE + """
<style>
  .scanzone{
    position:relative; border:1px dashed var(--border); border-radius:var(--radius);
    padding:48px 24px; text-align:center; overflow:hidden; cursor:pointer;
    background:var(--panel-2); transition:border-color .2s;
  }
  .scanzone.drag{ border-color:var(--blue); }
  .scanzone::before{
    content:""; position:absolute; left:0; right:0; top:-40%; height:40%;
    background:linear-gradient(180deg, rgba(62,123,250,0) 0%, rgba(62,123,250,.32) 50%, rgba(62,123,250,0) 100%);
    animation:sweep 3.2s linear infinite;
  }
  .scanzone.jd::before{
    background:linear-gradient(180deg, rgba(217,180,74,0) 0%, rgba(217,180,74,.30) 50%, rgba(217,180,74,0) 100%);
  }
  .scanzone.jd.drag{ border-color:var(--gold); }
  @keyframes sweep{ 0%{top:-40%;} 100%{top:100%;} }
  .scan-icon{font-family:var(--font-mono); font-size:12px; color:var(--muted); letter-spacing:.08em;}
  .scan-title{font-family:var(--font-display); font-size:19px; margin:12px 0 4px;}
  .scan-sub{color:var(--muted); font-size:13.5px;}
  .filename{ font-family:var(--font-mono); color:var(--gold); font-size:13px; margin-top:12px; min-height:18px; }
  input[type=file]{ display:none; }
  button{
    font-family:var(--font-body); font-weight:600; background:var(--blue); color:#04101F;
    border:none; padding:14px 26px; border-radius:10px; cursor:pointer; font-size:15px;
    margin-top:22px; width:100%;
  }
  button:disabled{ background:#1c2c47; color:#5f7a97; cursor:not-allowed; }
  button:not(:disabled):hover{ background:#5b93ff; }
  .roles{ display:flex; flex-wrap:wrap; gap:8px; margin-top:28px; }
  .chip{
    font-family:var(--font-mono); font-size:12px; color:var(--muted);
    border:1px solid var(--border); border-radius:999px; padding:6px 12px;
  }
  .error{ color:var(--danger); font-weight:600; margin-bottom:16px; }

  .section-label{ font-family:var(--font-display); font-size:17px; margin:0 0 6px; }
  .section-sub{ color:var(--muted); font-size:13.5px; margin:0 0 16px; }
  .divider{
    display:flex; align-items:center; gap:14px; margin:40px 0 24px;
    font-family:var(--font-mono); font-size:11px; letter-spacing:.1em; color:var(--muted); text-transform:uppercase;
  }
  .divider::before, .divider::after{ content:""; flex:1; height:1px; background:var(--border); }
  .badge-optional{ font-family:var(--font-mono); font-size:10.5px; color:var(--gold); border:1px solid var(--gold-soft);
    background:var(--gold-soft); padding:2px 8px; border-radius:999px; margin-left:8px; text-transform:uppercase; letter-spacing:.06em; }

  .jd-toggle{ display:flex; gap:8px; margin-bottom:14px; }
  .toggle-btn{
    font-family:var(--font-mono); font-size:12px; letter-spacing:.04em; text-transform:uppercase;
    background:transparent; color:var(--muted); border:1px solid var(--border); border-radius:8px;
    padding:8px 14px; cursor:pointer; width:auto; margin-top:0; flex:1;
  }
  .toggle-btn.active{ color:var(--gold); border-color:var(--gold-soft); background:var(--gold-soft); }
  .toggle-btn:hover{ background:rgba(255,255,255,.03); }

  textarea{
    width:100%; min-height:140px; resize:vertical; background:var(--panel-2); color:var(--text);
    border:1px solid var(--border); border-radius:var(--radius); padding:16px; font-family:var(--font-body);
    font-size:14px; line-height:1.6;
  }
  textarea::placeholder{ color:var(--muted); }
  textarea:focus{ outline:none; border-color:var(--gold); }
</style>
</head>
<body>
  <div class="wrap">
    <p class="eyebrow">Diagnostic Scan · v3</p>
    <h1>Run your CV through the analyzer.</h1>
    <p class="lede">Drop a resume and get a role-fit % across every open profile, a skill radar, an ATS health check, and what to study next to close the gap.</p>

    <div class="panel">
      {% if error %}<p class="error">{{ error }}</p>{% endif %}
      <form method="POST" action="/analyze" enctype="multipart/form-data" id="uploadForm">

        <p class="section-label">Your CV</p>
        <p class="section-sub">Required — we scan text only, nothing is stored after analysis.</p>
        <label class="scanzone" id="dropzone" for="fileInput" tabindex="0">
          <div class="scan-icon">PDF · DOCX · TXT</div>
          <div class="scan-title">Drop your CV, or click to browse</div>
          <div class="scan-sub">Used against every role profile below.</div>
          <div class="filename" id="filename"></div>
        </label>
        <input type="file" name="resume" id="fileInput" accept=".pdf,.docx,.txt" required>

        <div class="divider">Target a specific job <span class="badge-optional">Optional</span></div>
        <p class="section-sub">Add a real job description to get a direct match % against that exact posting — not just a generic role profile.</p>

        <div class="jd-toggle">
          <button type="button" class="toggle-btn active" id="jdModeFile">Upload JD file</button>
          <button type="button" class="toggle-btn" id="jdModeText">Paste JD text</button>
        </div>

        <div id="jdFileWrap">
          <label class="scanzone jd" id="jdDropzone" for="jdFileInput" tabindex="0">
            <div class="scan-icon">PDF · DOCX · TXT</div>
            <div class="scan-title">Drop a job description, or click to browse</div>
            <div class="scan-sub">We'll pull its key terms and check them against your CV.</div>
            <div class="filename" id="jdFilename"></div>
          </label>
          <input type="file" name="jd_file" id="jdFileInput" accept=".pdf,.docx,.txt">
        </div>

        <div id="jdTextWrap" style="display:none;">
          <textarea name="jd_text" id="jdTextArea" placeholder="Paste the job description here..."></textarea>
        </div>

        <button type="submit" id="submitBtn" disabled>Run Diagnostic</button>
      </form>
    </div>

    <div class="roles">
      {% for role in roles %}<span class="chip">{{ role }}</span>{% endfor %}
    </div>
  </div>

<script>
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const filenameEl = document.getElementById('filename');
  const submitBtn = document.getElementById('submitBtn');

  function setFile(el, file){
    if(!file) return;
    el.textContent = "Selected: " + file.name;
  }
  fileInput.addEventListener('change', e => { setFile(filenameEl, e.target.files[0]); submitBtn.disabled = false; });
  ['dragenter','dragover'].forEach(evt=>{
    dropzone.addEventListener(evt, e=>{ e.preventDefault(); dropzone.classList.add('drag'); });
  });
  ['dragleave','drop'].forEach(evt=>{
    dropzone.addEventListener(evt, e=>{ e.preventDefault(); dropzone.classList.remove('drag'); });
  });
  dropzone.addEventListener('drop', e=>{
    const file = e.dataTransfer.files[0];
    if(file){ fileInput.files = e.dataTransfer.files; setFile(filenameEl, file); submitBtn.disabled = false; }
  });

  // JD file dropzone
  const jdDropzone = document.getElementById('jdDropzone');
  const jdFileInput = document.getElementById('jdFileInput');
  const jdFilenameEl = document.getElementById('jdFilename');
  jdFileInput.addEventListener('change', e => setFile(jdFilenameEl, e.target.files[0]));
  ['dragenter','dragover'].forEach(evt=>{
    jdDropzone.addEventListener(evt, e=>{ e.preventDefault(); jdDropzone.classList.add('drag'); });
  });
  ['dragleave','drop'].forEach(evt=>{
    jdDropzone.addEventListener(evt, e=>{ e.preventDefault(); jdDropzone.classList.remove('drag'); });
  });
  jdDropzone.addEventListener('drop', e=>{
    const file = e.dataTransfer.files[0];
    if(file){ jdFileInput.files = e.dataTransfer.files; setFile(jdFilenameEl, file); }
  });

  // JD mode toggle: file upload vs paste text
  const jdModeFile = document.getElementById('jdModeFile');
  const jdModeText = document.getElementById('jdModeText');
  const jdFileWrap = document.getElementById('jdFileWrap');
  const jdTextWrap = document.getElementById('jdTextWrap');
  const jdTextArea = document.getElementById('jdTextArea');

  jdModeFile.addEventListener('click', () => {
    jdModeFile.classList.add('active'); jdModeText.classList.remove('active');
    jdFileWrap.style.display = ''; jdTextWrap.style.display = 'none';
    jdTextArea.value = '';
  });
  jdModeText.addEventListener('click', () => {
    jdModeText.classList.add('active'); jdModeFile.classList.remove('active');
    jdTextWrap.style.display = ''; jdFileWrap.style.display = 'none';
    jdFileInput.value = ''; jdFilenameEl.textContent = '';
  });
</script>
</body></html>
"""

RESULT_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diagnostic Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
""" + BASE_STYLE + """
<style>
  .hero{ display:flex; gap:32px; align-items:center; flex-wrap:wrap; margin-bottom:36px; }
  .gauge-wrap{ position:relative; width:200px; height:200px; flex:0 0 auto; }
  .gauge-wrap svg{ transform:rotate(-90deg); }
  .gauge-bg{ fill:none; stroke:var(--border); stroke-width:12; }
  .gauge-fg{ fill:none; stroke:var(--gold); stroke-width:12; stroke-linecap:round;
             transition:stroke-dashoffset 1s ease-out; }
  .gauge-label{ position:absolute; inset:0; display:flex; flex-direction:column;
                align-items:center; justify-content:center; }
  .gauge-score{ font-family:var(--font-display); font-size:38px; }
  .gauge-tag{ font-family:var(--font-mono); font-size:11px; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; }
  .hero-text{ flex:1; min-width:240px; }
  .hero-role{ font-family:var(--font-display); font-size:26px; margin:0 0 6px; }
  .badge{
    display:inline-block; font-family:var(--font-mono); font-size:11px; letter-spacing:.06em;
    text-transform:uppercase; padding:4px 10px; border-radius:999px; margin-bottom:10px;
  }
  .badge.strong{ background:var(--blue-soft); color:var(--blue); }
  .badge.good{ background:var(--gold-soft); color:var(--gold); }
  .badge.dev{ background:var(--danger-soft); color:#f0899a; }
  .badge.early{ background:rgba(224,85,111,.2); color:var(--danger); }
  .narrative{ color:var(--muted); font-size:14.5px; max-width:60ch; }

  .score-breakdown{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  .score-pill{
    font-family:var(--font-mono); font-size:11px; padding:4px 10px; border-radius:999px;
    border:1px solid var(--border); color:var(--muted);
  }
  .score-pill b{ color:var(--text); }
  .score-pill.semantic{ border-color:var(--gold-soft); color:var(--gold); }
  .score-pill.semantic b{ color:var(--gold); }

  .sbert-note{
    display:flex; align-items:center; gap:10px; font-size:13px; color:var(--muted);
    background:var(--panel); border:1px solid var(--border); border-radius:10px;
    padding:12px 16px; margin-bottom:32px;
  }
  .sbert-note .dot{ width:8px; height:8px; border-radius:50%; flex:0 0 auto; }
  .sbert-note.on .dot{ background:var(--gold); }
  .sbert-note.off .dot{ background:var(--muted); }

  h2.section{ font-family:var(--font-display); font-size:19px; margin:44px 0 4px; }
  p.section-desc{ color:var(--muted); font-size:13.5px; margin:0 0 16px; max-width:64ch; }

  /* Job match panel */
  .jd-panel{
    border-radius:var(--radius); padding:28px; margin-bottom:8px;
    background:linear-gradient(135deg, var(--blue-soft), var(--gold-soft)), var(--panel);
    border:1px solid var(--border); position:relative; overflow:hidden;
  }
  .jd-panel::before{
    content:""; position:absolute; inset:0; border-radius:var(--radius); padding:1px;
    background:linear-gradient(135deg, rgba(62,123,250,.5), rgba(217,180,74,.5));
    -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite:xor; mask-composite:exclude; pointer-events:none;
  }
  .jd-top{ display:flex; gap:26px; align-items:center; flex-wrap:wrap; }
  .jd-score{ font-family:var(--font-display); font-size:44px; flex:0 0 auto; }
  .jd-score-tag{ font-family:var(--font-mono); font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }
  .jd-narrative{ color:var(--text); font-size:14.5px; max-width:62ch; margin-top:4px; }
  .jd-chips{ display:flex; flex-wrap:wrap; gap:6px; margin-top:18px; }

  .chips{ display:flex; flex-wrap:wrap; gap:6px; margin-top:14px; }
  .chip-mini{ font-family:var(--font-mono); font-size:11px; padding:4px 9px; border-radius:999px; }
  .chip-mini.match{ background:var(--blue-soft); color:var(--blue); }
  .chip-mini.miss{ background:var(--danger-soft); color:#f0899a; }

  /* Radar */
  .radar-wrap{
    background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
    padding:32px 60px; display:flex; justify-content:center;
  }
  .radar-svg{ overflow:visible; max-width:100%; height:auto; }
  .radar-ring{ fill:none; stroke:var(--border); stroke-width:1; }
  .radar-axis{ stroke:var(--border); stroke-width:1; }
  .radar-fill{
    fill:url(#radarFillGrad); stroke:var(--gold); stroke-width:2; stroke-linejoin:round;
    transform-origin:center; transform-box:fill-box; animation:radarGrow .9s cubic-bezier(.16,1,.3,1) both;
  }
  @keyframes radarGrow{ from{ transform:scale(0); opacity:0; } to{ transform:scale(1); opacity:1; } }
  .radar-dot{ fill:var(--gold); }
  .radar-label{ font-family:var(--font-mono); font-size:11px; fill:var(--text); }
  .radar-label-score{ font-family:var(--font-mono); font-size:10px; fill:var(--muted); }

  /* Health checks */
  .health-grid{ display:grid; gap:12px; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); }
  .check-card{
    background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px 18px;
    border-left:3px solid var(--border);
  }
  .check-card.ok{ border-left-color:var(--blue); }
  .check-card.warn{ border-left-color:var(--gold); }
  .check-top{ display:flex; align-items:center; gap:8px; margin-bottom:6px; }
  .check-icon{
    width:20px; height:20px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-family:var(--font-mono); font-size:12px; flex:0 0 auto;
  }
  .check-card.ok .check-icon{ background:var(--blue-soft); color:var(--blue); }
  .check-card.warn .check-icon{ background:var(--gold-soft); color:var(--gold); }
  .check-label{ font-weight:600; font-size:14px; }
  .check-detail{ font-size:13px; color:var(--muted); }

  .roadmap{ display:grid; gap:10px; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); }
  .road-item{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; position:relative; }
  .road-item.from-jd{ border-color:var(--gold-soft); }
  .road-skill{ font-family:var(--font-mono); color:var(--gold); font-size:13px; margin-bottom:4px; text-transform:uppercase; letter-spacing:.04em; display:flex; align-items:center; gap:6px;}
  .jd-tag{ font-family:var(--font-mono); font-size:9px; background:var(--blue-soft); color:var(--blue); padding:2px 6px; border-radius:999px; text-transform:uppercase; letter-spacing:.05em; }
  .road-tip{ font-size:13.5px; color:var(--muted); }

  .grid{ display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); }
  details.role-card{
    background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:20px;
  }
  details.role-card summary{ cursor:pointer; list-style:none; }
  details.role-card summary::-webkit-details-marker{ display:none; }
  .card-top{ display:flex; align-items:center; gap:14px; }
  .mini-gauge{ width:70px; height:70px; flex:0 0 auto; position:relative; }
  .mini-gauge svg{ transform:rotate(-90deg); }
  .mini-score{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
               font-family:var(--font-mono); font-size:14px; }
  .card-role{ font-weight:600; font-size:15px; }
  .card-fit{ font-size:12px; color:var(--muted); }
  .card-narrative{ font-size:13px; color:var(--muted); margin-top:14px; }

  a.back{ display:inline-block; margin-top:40px; color:var(--blue); text-decoration:none; font-weight:600; font-family:var(--font-mono); font-size:14px;}
</style>
</head>
<body>
<div class="wrap">
  <p class="eyebrow">Diagnostic Report</p>

  <div class="sbert-note {{ 'on' if sbert_active else 'off' }}">
    <span class="dot"></span>
    {% if sbert_active %}
      Scores below blend lexical keyword matching with Sentence-BERT zero-shot semantic similarity (<code>{{ sbert_model }}</code>) — so paraphrased or synonym skills count too, not just exact keyword hits.
    {% else %}
      Semantic model (Sentence-BERT) unavailable on this run — falling back to lexical + TF-IDF scoring only. Results are still accurate, just keyword-driven.
    {% endif %}
  </div>

  {% set top = results[0] %}
  <div class="hero">
    <div class="gauge-wrap">
      <svg width="200" height="200" viewBox="0 0 200 200">
        <circle class="gauge-bg" cx="100" cy="100" r="90"></circle>
        <circle class="gauge-fg" cx="100" cy="100" r="90"
                stroke-dasharray="{{ hero_circ }}"
                stroke-dashoffset="{{ top.hero_offset }}"></circle>
      </svg>
      <div class="gauge-label">
        <div class="gauge-score">{{ top.final_score | round(0) | int }}%</div>
        <div class="gauge-tag">match</div>
      </div>
    </div>
    <div class="hero-text">
      <span class="badge {{ 'strong' if top.fit_label=='Strong Fit' else 'good' if top.fit_label=='Good Fit' else 'dev' if top.fit_label=='Developing Fit' else 'early' }}">{{ top.fit_label }}</span>
      <h1 class="hero-role">Best fit: {{ top.role }}</h1>
      <p class="narrative">{{ top.narrative }}</p>
      <div class="score-breakdown">
        <span class="score-pill">Lexical <b>{{ top.keyword_score | round(0) | int }}%</b></span>
        <span class="score-pill">TF-IDF <b>{{ top.tfidf_score | round(0) | int }}%</b></span>
        {% if top.sbert_score is not none %}<span class="score-pill semantic">Semantic (SBERT) <b>{{ top.sbert_score | round(0) | int }}%</b></span>{% endif %}
      </div>
    </div>
  </div>

  {% if jd %}
  <h2 class="section">Match against the job you provided</h2>
  <p class="section-desc">Scored against the actual terms pulled from that posting ({{ jd.term_count }} key terms detected), not a generic role profile.</p>
  <div class="jd-panel">
    <div class="jd-top">
      <div>
        <div class="jd-score">{{ jd.final_score | round(0) | int }}%</div>
        <div class="jd-score-tag">direct match</div>
      </div>
      <div style="flex:1; min-width:220px;">
        <span class="badge {{ 'strong' if jd.fit_label=='Strong Fit' else 'good' if jd.fit_label=='Good Fit' else 'dev' if jd.fit_label=='Developing Fit' else 'early' }}">{{ jd.fit_label }}</span>
        <p class="jd-narrative">{{ jd.narrative }}</p>
        <div class="score-breakdown">
          <span class="score-pill">Lexical <b>{{ jd.keyword_score | round(0) | int }}%</b></span>
          <span class="score-pill">TF-IDF <b>{{ jd.tfidf_score | round(0) | int }}%</b></span>
          {% if jd.sbert_score is not none %}<span class="score-pill semantic">Semantic (SBERT) <b>{{ jd.sbert_score | round(0) | int }}%</b></span>{% endif %}
        </div>
      </div>
    </div>
    <div class="jd-chips">
      {% for m in jd.matched_names[:10] %}<span class="chip-mini match">{{ m }}</span>{% endfor %}
      {% for m in jd.missing_top[:8] %}<span class="chip-mini miss">{{ m.skill }}</span>{% endfor %}
    </div>
  </div>
  {% endif %}

  {% if radar %}
  <h2 class="section">Role-fit radar</h2>
  <p class="section-desc">Every role profile plotted at once — the further a point sits from center, the stronger that fit.</p>
  <div class="radar-wrap">
    <svg class="radar-svg" width="{{ radar.size }}" height="{{ radar.size }}" viewBox="0 0 {{ radar.size }} {{ radar.size }}">
      <defs>
        <linearGradient id="radarFillGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="rgba(62,123,250,.32)"/>
          <stop offset="100%" stop-color="rgba(217,180,74,.28)"/>
        </linearGradient>
      </defs>
      {% for ring in radar.rings %}<polygon class="radar-ring" points="{{ ring }}"></polygon>{% endfor %}
      {% for x1,y1,x2,y2 in radar.axis_lines %}<line class="radar-axis" x1="{{x1}}" y1="{{y1}}" x2="{{x2}}" y2="{{y2}}"></line>{% endfor %}
      <polygon class="radar-fill" points="{{ radar.polygon }}"></polygon>
      {% for r in radar.roles %}
        {% set coords = r.point.split(',') %}
        <circle class="radar-dot" cx="{{ coords[0] }}" cy="{{ coords[1] }}" r="3.5"></circle>
        <text class="radar-label" x="{{ r.label_x }}" y="{{ r.label_y }}" text-anchor="{{ r.anchor }}">{{ r.short }}</text>
        <text class="radar-label-score" x="{{ r.label_x }}" y="{{ r.label_y + 13 }}" text-anchor="{{ r.anchor }}">{{ r.final_score | round(0) | int }}%</text>
      {% endfor %}
    </svg>
  </div>
  {% endif %}

  <h2 class="section">CV health check</h2>
  <p class="section-desc">Quick, explainable checks — the kind of things a recruiter or an ATS parser notices in the first few seconds.</p>
  <div class="health-grid">
    {% for c in health %}
    <div class="check-card {{ c.status }}">
      <div class="check-top">
        <span class="check-icon">{{ '✓' if c.status == 'ok' else '!' }}</span>
        <span class="check-label">{{ c.label }}</span>
      </div>
      <div class="check-detail">{{ c.detail }}</div>
    </div>
    {% endfor %}
  </div>

  <h2 class="section">What to study next</h2>
  <p class="section-desc">{% if jd %}Prioritized using both your top-fit roles and the gaps against the job you provided.{% else %}Combined from the missing skills across your top-fit roles.{% endif %}</p>
  <div class="roadmap">
    {% for item in roadmap %}
    <div class="road-item {{ 'from-jd' if item.from_jd else '' }}">
      <div class="road-skill">{{ item.skill }}{% if item.from_jd %}<span class="jd-tag">from JD</span>{% endif %}</div>
      <div class="road-tip">{{ item.tip }}</div>
    </div>
    {% endfor %}
  </div>

  <h2 class="section">All roles — full breakdown</h2>
  <div class="grid">
    {% for r in results %}
    <details class="role-card" {% if loop.first %}open{% endif %}>
      <summary>
        <div class="card-top">
          <div class="mini-gauge">
            <svg width="70" height="70" viewBox="0 0 70 70">
              <circle class="gauge-bg" cx="35" cy="35" r="28" stroke-width="7"></circle>
              <circle class="gauge-fg" cx="35" cy="35" r="28" stroke-width="7"
                      stroke-dasharray="{{ card_circ }}"
                      stroke-dashoffset="{{ r.card_offset }}"></circle>
            </svg>
            <div class="mini-score">{{ r.final_score | round(0) | int }}%</div>
          </div>
          <div>
            <div class="card-role">{{ r.role }}</div>
            <div class="card-fit">{{ r.fit_label }}</div>
          </div>
        </div>
      </summary>
      <div class="score-breakdown">
        <span class="score-pill">Lexical <b>{{ r.keyword_score | round(0) | int }}%</b></span>
        <span class="score-pill">TF-IDF <b>{{ r.tfidf_score | round(0) | int }}%</b></span>
        {% if r.sbert_score is not none %}<span class="score-pill semantic">Semantic <b>{{ r.sbert_score | round(0) | int }}%</b></span>{% endif %}
      </div>
      <div class="chips">
        {% for m in r.matched_names[:6] %}<span class="chip-mini match">{{ m }}</span>{% endfor %}
        {% for m in r.missing_top[:5] %}<span class="chip-mini miss">{{ m.skill }}</span>{% endfor %}
      </div>
      <p class="card-narrative">{{ r.narrative }}</p>
    </details>
    {% endfor %}
  </div>

  <a class="back" href="/">&larr; Analyze another CV</a>
</div>
</body></html>
"""


# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(UPLOAD_PAGE, error=None, roles=list(ROLE_PROFILES.keys()))


def _save_and_extract(file_storage, prefix=""):
    """Save an uploaded FileStorage to disk, extract its text, then clean up."""
    filename = secure_filename(file_storage.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{prefix}{filename}")
    file_storage.save(filepath)
    try:
        return extract_text(filepath, filename)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/analyze", methods=["POST"])
def analyze():
    roles = list(ROLE_PROFILES.keys())

    if "resume" not in request.files or request.files["resume"].filename == "":
        return render_template_string(UPLOAD_PAGE, error="No CV uploaded.", roles=roles)

    resume_file = request.files["resume"]
    if not allowed_file(resume_file.filename):
        return render_template_string(UPLOAD_PAGE, error="Your CV must be a PDF, DOCX, or TXT file.", roles=roles)

    try:
        cv_text = _save_and_extract(resume_file)
    except Exception:
        return render_template_string(UPLOAD_PAGE, error="Couldn't read that CV file. Try another file.", roles=roles)

    if not cv_text.strip():
        return render_template_string(UPLOAD_PAGE, error="Could not extract any text from the CV. Try another file.", roles=roles)

    # Optional job description: pasted text takes priority over an uploaded file.
    jd_raw_text = (request.form.get("jd_text") or "").strip()
    if not jd_raw_text:
        jd_file = request.files.get("jd_file")
        if jd_file and jd_file.filename:
            if allowed_file(jd_file.filename):
                try:
                    jd_raw_text = _save_and_extract(jd_file, prefix="jd_").strip()
                except Exception:
                    jd_raw_text = ""

    results, health, cv_embedding = analyze_resume(cv_text)
    cv_text_clean = clean_text(cv_text)
    sbert_active = cv_embedding is not None

    jd_analysis = analyze_jd(cv_text_clean, jd_raw_text, cv_embedding) if jd_raw_text else None
    roadmap = build_roadmap(results, jd_analysis)
    radar = build_radar(results)

    return render_template_string(
        RESULT_PAGE,
        results=results,
        roadmap=roadmap,
        health=health,
        jd=jd_analysis,
        radar=radar,
        sbert_active=sbert_active,
        sbert_model=SBERT_MODEL_NAME,
        hero_circ=HERO_CIRC,
        card_circ=CARD_CIRC,
    )


if __name__ == "__main__":
    # use_reloader=False avoids a "signal only works in main thread" crash
    # that occurs in some environments (certain IDEs, Windows setups, threads).
    app.run(debug=True, port=5000, use_reloader=False)

"""LLM Component — Knowledge-Base Ingestion (LLM_PLAN.md Phase 2).

Turns the guideline source documents into a clean, scoped, chunked corpus:

    knowledge_base/sources.csv  +  knowledge_base/raw/*  ->  processed/chunks.jsonl

Pipeline per source (driven by sources.csv):
  1. Skip sources with include_in_retrieval == false (background/planned).
  2. Read the local file:
       * .pdf       -> extract text from the configured page_ranges (PyMuPDF).
       * .md / .txt -> split into passages on '## Heading {topic=...}' markers.
  3. Clean text (collapse whitespace, drop page-number lines).
  4. Chunk to ~target words with overlap.
  5. THREE-TIER SAFETY CLASSIFICATION (the plan's "manual scoping pass", automated):
       * exclude_from_retrieval -> DROPPED: medication/drug, dialysis/KRT,
         transplant, drug dosing, direct numeric BP targets.
       * restricted_clinician_referral -> KEPT + flagged: KDOQI nutrition source,
         or any chunk with a numeric nutrition target (protein/sodium/etc.).
       * safe_public -> KEPT: general lifestyle/prevention education.
  6. (lifestyle scope only) also drop off-topic + reference chunks; infer a topic.
  7. Tag every chunk with metadata + safety_label and write JSONL + summary.

Why three tiers: the real KDIGO/ADA/KDOQI PDFs are large clinical documents
containing drug therapy, dialysis, and transplant content that is unsafe for a
non-clinical kidney-EDUCATION assistant to surface. We DROP that content entirely
(not merely flag it), keep numeric-nutrition content only as dietitian-referral,
and keep general lifestyle education as safe.

Run:
    python -m ckd_risk_app.llm.rag.ingest
    python -m ckd_risk_app.llm.rag.ingest --report-only
Then audit:
    python -m ckd_risk_app.llm.rag.audit
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LLM_DIR = Path(__file__).resolve().parents[1]            # ckd_risk_app/llm
KB_DIR = LLM_DIR / "knowledge_base"
RAW_DIR = KB_DIR / "raw"
PROCESSED_DIR = KB_DIR / "processed"
SOURCES_CSV = KB_DIR / "sources.csv"
DEFAULT_OUT = PROCESSED_DIR / "chunks.jsonl"

# ---------------------------------------------------------------------------
# Chunking config (word-based ≈ 0.75 words/token).
# ---------------------------------------------------------------------------
# Smaller chunks keep safety exclusion surgical: a single clinical sentence
# (e.g. one "dialysis" mention) drops only its own small chunk, not a whole
# page of otherwise-safe patient-education content.
TARGET_WORDS = 150
OVERLAP_WORDS = 25
MIN_CHUNK_WORDS = 20
WORDS_PER_TOKEN = 0.75

_PASSAGE_HEADER = re.compile(r"(?m)^\s*##\s+(.*?)\s*$")
_TOPIC_TAG = re.compile(r"\{topic\s*=\s*([a-zA-Z0-9_\-]+)\s*\}")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_PAGE_NUMBER_LINE = re.compile(r"(?m)^\s*\d{1,4}\s*$")

# Running header/footer boilerplate that PDF text extraction concatenates into
# the body (e.g. the free "Save Your Kidneys" patient book repeats a promo
# footer + chapter running header on every page). Patterns are specific enough
# not to touch the clinical guideline sources.
_BOILERPLATE_LINE = [
    re.compile(r"(?im)^.*(?:kidneyeducation\.com|free kidney book in 35).*$"),
    re.compile(r"(?im)^\s*\d{0,4}\.?\s*save your kidneys\s*$"),
    re.compile(r"(?im)^\s*chp\.\s*\d+\..*$"),
]
_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

# ---------------------------------------------------------------------------
# Topic vocabulary (KEEP lifestyle chunks + infer a topic).
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "bp": ["blood pressure", "hypertension", "systolic", "diastolic", "sodium", "salt"],
    "activity": ["physical activity", "exercise", "walking", "sedentary", "aerobic",
                 "active", "fitness", "workout"],
    "weight": ["weight", "obesity", "body mass", "bmi", "overweight"],
    "diabetes": ["glucose", "glycemic", "glycaemic", "hba1c", "diabetes", "diabetic",
                 "blood sugar"],
    "nutrition": ["diet", "dietary", "nutrition", "protein intake", "potassium",
                  "phosphorus", "food", "eating", "fruit", "vegetable", "fiber",
                  "whole grain"],
    "smoking": ["smoking", "tobacco", "cigarette", "smoker"],
    "sleep": ["sleep", "insomnia", "sleep quality"],
    "hydration": ["fluid intake", "hydration", "water intake"],
    "followup": ["self-management", "self-care", "monitoring", "follow-up", "screening",
                 "referral", "lifestyle modification", "behavioral", "behaviour",
                 "counseling", "education", "healthy lifestyle", "get tested", "risk factor"],
}
LIFESTYLE_KEYWORDS: List[str] = sorted(
    {kw for kws in TOPIC_KEYWORDS.values() for kw in kws}
    | {"lifestyle", "healthy living", "prevention", "wellbeing", "well-being", "weight loss"}
)

# ---------------------------------------------------------------------------
# Safety classification vocabulary (THREE TIERS)
# ---------------------------------------------------------------------------
# Tier 1: EXCLUDE entirely (clinical content unsafe for a non-clinical assistant).
EXCLUDE_PATTERNS: Dict[str, re.Pattern] = {
    "medication": re.compile(
        r"\b(ace[- ]?inhibitor|acei|arb|angiotensin|renin[- ]angiotensin|rasi?|"
        r"sglt-?2|sglt|flozin|glp-?1|glutide|gliptin|dpp-?4|finerenone|"
        r"mineralocorticoid|spironolactone|eplerenone|statin|metformin|sulfonylurea|"
        r"insulin|thiazolidinedione|pioglitazone|diuretic|thiazide|furosemide|"
        r"patiromer|sevelamer|phosphate binder|potassium binder|anticoagulant|warfarin|"
        r"apixaban|dabigatran|rivaroxaban|edoxaban|doac|noac|allopurinol|febuxostat|"
        r"urate[- ]lowering|erythropoietin|epoetin|darbepoetin|nsaid|glucocorticoid|"
        r"prednisone|corticosteroid|immunosuppress|pharmacolog|pharmacotherap|"
        r"drug therapy|antihypertensive (?:drug|agent|medication))\b", re.I),
    "dialysis": re.compile(
        r"\b(dialys\w*|haemodialys\w*|hemodialys\w*|peritoneal|"
        r"kidney replacement therapy|krt|dialyz\w*|dialyser\w*|ultrafiltration)\b", re.I),
    "transplant": re.compile(
        r"\b(transplant\w*|allograft|post-?transplant|donor kidney|graft (?:survival|function|loss))\b",
        re.I),
    # Real medication dosing only. NOTE: a bare "\d+ mg" is intentionally NOT
    # matched — it false-positives on lab thresholds like "30 mg/g" (UACR) and
    # "mg/dL". We require a per-day/frequency context or explicit dosing words.
    "dosing": re.compile(
        r"(\b\d+(\.\d+)?\s?mg\s?(/d\b|/day|per day|daily)|mg/d\b|mg/day|mg per day|"
        r"\btitrat\w*|\bonce daily\b|\btwice daily\b|\bb\.?i\.?d\.?\b|\bdosing\b)", re.I),
    "bp_target": re.compile(
        r"(<\s?1[0-9]0\s?mm\s?hg|≤\s?1[0-9]0\s?mm\s?hg|"
        r"systolic[^.]{0,40}<\s?1[0-9]0|target\s+(?:blood pressure|systolic)|"
        r"(?:blood pressure|bp)\s+target|<\s?1[0-9]0/\d{2})", re.I),
}

# Tier 2: RESTRICT (keep, but surface only as clinician/dietitian referral).
_NUTRIENT = re.compile(r"\b(protein|sodium|potassium|phosphorus|phosphate|salt)\b", re.I)
_NUTRIENT_DOSE = re.compile(
    r"\d+(\.\d+)?\s*(g/kg|mg/day|mg/d\b|g/day|g/d\b|mmol|grams? per day|g per day)", re.I)

SAFE_PUBLIC = "safe_public"
RESTRICTED = "restricted_clinician_referral"
EXCLUDED = "exclude_from_retrieval"


def has_numeric_nutrition_target(text: str) -> bool:
    return bool(_NUTRIENT.search(text) and _NUTRIENT_DOSE.search(text))


def classify_chunk(text: str, source_restricted: bool) -> Tuple[str, Optional[str]]:
    """Return (safety_label, reason).

    Order matters: exclusion wins over restriction wins over safe.
    """
    for category, pat in EXCLUDE_PATTERNS.items():
        if pat.search(text):
            return EXCLUDED, category
    if source_restricted:
        return RESTRICTED, "source_restricted"
    if has_numeric_nutrition_target(text):
        return RESTRICTED, "numeric_nutrition_target"
    return SAFE_PUBLIC, None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class SourceRecord:
    source_id: str
    title: str
    organization: str
    year: str
    priority: str
    restricted: bool
    include_in_retrieval: bool
    scope: str
    page_ranges: str
    topics: List[str]
    local_file: str
    url: str
    access_date: str
    license_note: str
    status: str


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_id: str
    title: str
    organization: str
    year: str
    priority: str
    safety_label: str
    restricted: bool
    topic: str
    passage_title: str
    n_words: int
    est_tokens: int
    status: str
    extra: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def load_sources(sources_csv: Path = SOURCES_CSV) -> List[SourceRecord]:
    if not sources_csv.exists():
        raise FileNotFoundError(f"sources.csv not found at {sources_csv}")
    records: List[SourceRecord] = []
    with sources_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            records.append(SourceRecord(
                source_id=row["source_id"].strip(),
                title=row.get("title", "").strip(),
                organization=row.get("organization", "").strip(),
                year=row.get("year", "").strip(),
                priority=row.get("priority", "").strip(),
                restricted=_as_bool(row.get("restricted", "false")),
                include_in_retrieval=_as_bool(row.get("include_in_retrieval", "true")),
                scope=(row.get("scope", "all").strip() or "all"),
                page_ranges=(row.get("page_ranges", "all").strip() or "all"),
                topics=[t.strip() for t in row.get("topics", "").split(";") if t.strip()],
                local_file=row.get("local_file", "").strip(),
                url=row.get("url", "").strip(),
                access_date=row.get("access_date", "").strip(),
                license_note=row.get("license_note", "").strip(),
                status=row.get("status", "").strip(),
            ))
    return records


# ---------------------------------------------------------------------------
# Cleaning / scoping helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = _HTML_COMMENT.sub("", text)
    for pat in _BOILERPLATE_LINE:
        text = pat.sub("", text)
    text = _PAGE_NUMBER_LINE.sub("", text)
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _matches_lifestyle(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in LIFESTYLE_KEYWORDS)


def _looks_like_references(text: str) -> bool:
    t = text.lower()
    if "doi:" in t or t.count("http") >= 2:
        return True
    if t.count(" et al") >= 2:
        return True
    if len(_YEAR.findall(t)) >= 5:
        return True
    digits = sum(c.isdigit() for c in text)
    if text and digits / len(text) > 0.18:
        return True
    return False


def infer_topic(text: str, default_topic: str) -> str:
    t = text.lower()
    best_topic, best_hits = default_topic, 0
    for topic, kws in TOPIC_KEYWORDS.items():
        hits = sum(t.count(kw) for kw in kws)
        if hits > best_hits:
            best_topic, best_hits = topic, hits
    return best_topic if best_hits > 0 else (default_topic or "general")


def parse_page_ranges(spec: str, n_pages: int) -> List[int]:
    spec = (spec or "all").strip().lower()
    if spec in {"", "all"}:
        return list(range(n_pages))
    pages: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(max(int(a) - 1, 0), min(int(b), n_pages)))
        else:
            idx = int(part) - 1
            if 0 <= idx < n_pages:
                pages.append(idx)
    return sorted(set(pages))


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def read_markdown_passages(path: Path, default_topic: str) -> List[Tuple[str, str, str]]:
    raw = _HTML_COMMENT.sub("", path.read_text(encoding="utf-8"))
    passages: List[Tuple[str, str, str]] = []
    matches = list(_PASSAGE_HEADER.finditer(raw))
    if not matches:
        body = clean_text(raw)
        return [("", default_topic, body)] if body else []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        topic_match = _TOPIC_TAG.search(heading)
        topic = topic_match.group(1) if topic_match else default_topic
        title = _TOPIC_TAG.sub("", heading).strip()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = clean_text(raw[m.end():body_end])
        if body:
            passages.append((title, topic, body))
    return passages


def read_pdf_text(path: Path, page_ranges: str) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = parse_page_ranges(page_ranges, doc.page_count)
        return "\n\n".join(doc[i].get_text("text") for i in pages)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] PyMuPDF failed on {path.name}: {exc}")
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = parse_page_ranges(page_ranges, len(pdf.pages))
            return "\n\n".join((pdf.pages[i].extract_text() or "") for i in pages)
    except ImportError:
        print(f"  [warn] No PDF library; skipping {path.name}. Run: pip install pymupdf")
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] pdfplumber failed on {path.name}: {exc}")
    return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, target_words: int = TARGET_WORDS,
               overlap_words: int = OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= target_words:
        return [text]
    chunks: List[str] = []
    step = max(target_words - overlap_words, 1)
    start = 0
    while start < len(words):
        window = words[start:start + target_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + target_words >= len(words):
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_chunks(sources: List[SourceRecord], raw_dir: Path = RAW_DIR
                 ) -> Tuple[List[Chunk], Counter]:
    chunks: List[Chunk] = []
    excluded_by_cat: Counter = Counter()

    for src in sources:
        if not src.include_in_retrieval:
            print(f"[skip] {src.source_id}: include_in_retrieval=false ({src.status})")
            continue
        if not src.local_file:
            print(f"[skip] {src.source_id}: no local_file (status={src.status})")
            continue
        path = raw_dir / src.local_file
        if not path.exists():
            print(f"[miss] {src.source_id}: file not found -> {src.local_file}")
            continue

        default_topic = src.topics[0] if src.topics else "general"
        lifestyle_scope = src.scope.lower() == "lifestyle"

        # Read candidate pieces (PDF page-range or md passages).
        suffix = path.suffix.lower()
        pieces: List[Tuple[str, Optional[str], str]] = []
        if suffix in {".md", ".txt"}:
            for title, topic, body in read_markdown_passages(path, default_topic):
                for piece in chunk_text(body):
                    pieces.append((title, topic, piece))
        elif suffix == ".pdf":
            text = clean_text(read_pdf_text(path, src.page_ranges))
            for piece in chunk_text(text):
                pieces.append(("", None, piece))
        else:
            print(f"[skip] {src.source_id}: unsupported file type {suffix}")
            continue

        per_topic_seq: Counter = Counter()
        kept = excluded = off_topic = refs = 0
        for passage_title, declared_topic, text in pieces:
            if len(text.split()) < MIN_CHUNK_WORDS:
                continue

            if lifestyle_scope:
                if _looks_like_references(text):
                    refs += 1
                    continue
                if not _matches_lifestyle(text):
                    off_topic += 1
                    continue

            # THREE-TIER safety classification (applies to ALL sources).
            label, reason = classify_chunk(text, src.restricted)
            if label == EXCLUDED:
                excluded += 1
                excluded_by_cat[reason] += 1
                continue

            topic = infer_topic(text, default_topic) if lifestyle_scope else (declared_topic or default_topic)
            extra: Dict[str, object] = {}
            if src.restricted:
                extra["license_note"] = src.license_note
            if reason == "numeric_nutrition_target":
                extra["restricted_reason"] = "contains numeric nutrition target"

            per_topic_seq[topic] += 1
            n_words = len(text.split())
            chunks.append(Chunk(
                chunk_id=f"{src.source_id}_{topic}_{per_topic_seq[topic]:03d}",
                text=text, source_id=src.source_id, title=src.title,
                organization=src.organization, year=src.year, priority=src.priority,
                safety_label=label, restricted=(label == RESTRICTED),
                topic=topic, passage_title=passage_title, n_words=n_words,
                est_tokens=round(n_words / WORDS_PER_TOKEN), status=src.status, extra=extra,
            ))
            kept += 1

        detail = f"excluded {excluded} clinical"
        if lifestyle_scope:
            detail += f", {off_topic} off-topic, {refs} reference"
        print(f"[ok]   {src.source_id}: kept {kept} ({detail}) from {path.name}")

    return chunks, excluded_by_cat


def write_jsonl(chunks: List[Chunk], out_path: Path = DEFAULT_OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(chunks: List[Chunk], excluded_by_cat: Counter) -> None:
    if not chunks:
        print("\n[summary] 0 chunks produced.")
        return
    by_source = Counter(c.source_id for c in chunks)
    by_label = Counter(c.safety_label for c in chunks)
    by_topic = Counter(c.topic for c in chunks)
    sizes = [c.est_tokens for c in chunks]

    print("\n" + "=" * 60)
    print(f"KNOWLEDGE BASE SUMMARY — {len(chunks)} chunks")
    print("=" * 60)
    print("By source: " + ", ".join(f"{k}={v}" for k, v in by_source.most_common()))
    print("By safety: " + ", ".join(f"{k}={v}" for k, v in by_label.most_common()))
    print("By topic:  " + ", ".join(f"{k}={v}" for k, v in by_topic.most_common()))
    print(f"Est. tokens/chunk: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.0f}")
    if excluded_by_cat:
        print("Excluded (dropped) by category: "
              + ", ".join(f"{k}={v}" for k, v in excluded_by_cat.most_common()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(sources_csv: Path = SOURCES_CSV, raw_dir: Path = RAW_DIR,
        out_path: Path = DEFAULT_OUT, report_only: bool = False) -> List[Chunk]:
    print(f"Sources: {sources_csv}\nRaw dir: {raw_dir}\n")
    sources = load_sources(sources_csv)
    chunks, excluded_by_cat = build_chunks(sources, raw_dir)
    if not report_only:
        write_jsonl(chunks, out_path)
        print(f"\nWrote {len(chunks)} chunks -> {out_path}")
    summarize(chunks, excluded_by_cat)
    return chunks


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Ingest guideline sources into chunks.jsonl")
    p.add_argument("--sources", type=Path, default=SOURCES_CSV)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)
    run(args.sources, args.raw_dir, args.out, args.report_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

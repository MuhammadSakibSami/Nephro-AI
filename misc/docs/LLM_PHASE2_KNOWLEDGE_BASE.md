# LLM Phase 2 — Knowledge Base Construction

**Status:** ✅ Done — 9 real sources, 3-tier safety model, **audit gate passing (0 leakage)**
**Plan reference:** `LLM_PLAN.md` → Phase 2
**Date implemented:** June 2026 (v3: placeholders → real PDFs → 3-tier safety model after KB audit)

---

## 1. What this phase is (in plain language)

The LLM bases its advice on **trusted guidelines**, so first we turn those guideline
documents into small, searchable, well-labeled pieces of text called **chunks**.

The hard part is **safety**. The real guidelines are huge clinical PDFs that are
mostly about things our assistant must **never** talk about — medications, drug
dosing, dialysis, transplant. We only want the **lifestyle / prevention / diet /
activity / screening** parts. So this phase does an automated **scoping + safety
pass** and ends with an **audit gate** that proves no unsafe content leaked
through before we freeze the knowledge base.

The result is `chunks.jsonl` — clean, labeled "index cards" that Phase 3 makes
searchable and Phase 5's LLM reads before answering.

---

## 2. The three-tier safety model (the key idea)

An earlier version used only two tiers (safe vs restricted) and **kept**
medication/dialysis/transplant chunks as "unrestricted." A KB audit found that was
unsafe: those chunks could be retrieved and lead the LLM to give clinical advice.

The fix is **three tiers**, applied to every chunk:

| Tier | What it covers | What happens |
|---|---|---|
| `exclude_from_retrieval` | medication/drug names & classes, drug dosing, dialysis/KRT, transplant, direct numeric BP targets | **DROPPED** — never enters the KB |
| `restricted_clinician_referral` | KDOQI nutrition source, or any chunk with a numeric nutrition target (protein/sodium/potassium/phosphorus) | **KEPT + flagged** — surfaced only as "discuss with a clinician/renal dietitian" |
| `safe_public` | general lifestyle / prevention / activity / screening education | **KEPT** — normal use |

So medication/dialysis/transplant content is **removed entirely**, not merely
flagged. Each chunk carries a `safety_label` field recording its tier.

---

## 3. The 9 sources (final set)

| source_id | Document | Tier role | Scope |
|---|---|---|---|
| `KDIGO2024` | KDIGO 2024 CKD guideline | primary | lifestyle §3.2–3.3 (pp 41–42, 90–97, 144–148) |
| `KDIGO2021BP` | KDIGO 2021 BP in CKD | primary | Chapter 2 lifestyle interventions (pp 37–41) |
| `ADAKDIGO2022` | ADA/KDIGO 2022 Diabetes-in-CKD consensus | primary | lifestyle/self-management (pp 4–6) |
| `ADA2026` | **ADA Standards of Care 2026, Ch.11 CKD & Risk Mgmt** (`dc26s011.pdf`) | primary | screening/lifestyle/nutrition (pp 2,5,7,13) |
| `KDOQI2020NUTR` | NKF KDOQI 2020 Nutrition in CKD | **restricted** | Summary statements (pp 18–23) |
| `NKFPED` | NKF "Exercise and CKD" — **cleaned markdown** | patient education | general exercise only |
| `CDC_CKD_BASICS` | CDC CKD Basics | patient education | all (3 pp) |
| `CDC_CKD_PREVENT` | CDC Preventing CKD | patient education | all (2 pp) |
| `CDC_CKD_RISK` | CDC Risk Factors for CKD | patient education | all (2 pp) |

Plus two non-retrieved rows kept for documentation: `CDCKIDNEY` (legacy stub,
superseded) and `KDIGO2012` (background — justifies the target label, not a
recommendation source).

**Notes:**
- The diabetes consensus (`ADAKDIGO2022`) is the **ADA/KDIGO 2022 consensus**, a
  different document from the new **ADA 2026** Standards chapter — both are now
  included.
- `NKFPED` was switched from the original PDF (which contained dialysis/transplant
  content) to a hand-cleaned markdown (`nkf_exercise_ckd_clean.md`) holding only
  general exercise guidance. The original PDF remains in `raw/` but is unused.
- CDC sources are **US-government public domain** (easiest licensing); they add
  plain-language, prevention-focused, readable content.

---

## 4. How scoping + safety work together (3 mechanisms)

1. **Page ranges** (`page_ranges` in `sources.csv`) restrict each big PDF to its
   lifestyle/screening sections (located by searching the PDF for section
   headings). This removes most out-of-scope material up front.
2. **Lifestyle keyword filter + reference drop** (`scope=lifestyle`) keeps only
   prevention/lifestyle chunks and drops citation lists; infers a `topic`.
3. **Three-tier safety classifier** (`classify_chunk`, applied to **all** sources)
   drops `exclude` chunks, flags `restricted` chunks, keeps `safe_public`.

**Chunk size matters for safety.** Chunks are ~150 words (~190 tokens). Smaller
chunks make exclusion *surgical*: one "dialysis" sentence drops only its own small
chunk, not a whole page of otherwise-safe CDC prevention text. (An earlier 320-word
setting wiped out entire CDC/ADA pages over a single clinical mention.)

**One tuning detail worth remembering:** the dosing pattern deliberately does
**not** match a bare `"\d+ mg"`, because that false-positives on lab thresholds
like **"30 mg/g" (UACR albuminuria)** and "mg/dL" — which are screening content we
want to keep, not drug doses.

---

## 5. Current corpus (after audit)

- **118 chunks** kept; **75 dropped** as clinical (dialysis 31, medication 29,
  bp_target 6, dosing 5, transplant 4).
- By safety: **safe_public = 86**, **restricted_clinician_referral = 32**.
- By source: KDIGO2024=45, KDIGO2021BP=29, KDOQI2020NUTR=15, ADAKDIGO2022=7,
  ADA2026=7, NKFPED=6, CDC_CKD_BASICS=3, CDC_CKD_PREVENT=3, CDC_CKD_RISK=3.
- By topic: nutrition=47, activity=22, bp=18, followup=10, general=9, diabetes=5,
  weight=4, smoking=2, sleep=1.
- Est. tokens/chunk: min 35, max 200, mean 187.

**Audit gate: PASSED** — 0 safe_public chunks contain medication / dialysis /
transplant / dosing / BP-target / numeric-nutrition content; all restricted chunks
correctly labeled; no excluded chunks present.

---

## 6. What was built (files)

```
ckd_risk_app/llm/
  rag/
    ingest.py        # pipeline: page-range + lifestyle filter + 3-tier safety
    audit.py         # independent safety gate over the final chunks.jsonl
  knowledge_base/
    sources.csv      # registry: 9 collected + 2 documentation-only rows
    raw/             # 5 guideline PDFs + 3 CDC PDFs + 1 ADA PDF + cleaned NKF .md
    processed/chunks.jsonl   # GENERATED: 118 chunks (each has safety_label)
    index/           # (Phase 3 FAISS index)
```

Each chunk record: `chunk_id`, `text`, `source_id`, `title`, `organization`,
`year`, `priority`, **`safety_label`**, `restricted`, `topic`, `passage_title`,
`n_words`, `est_tokens`, `status`, `extra` (license note / restricted reason).

---

## 7. How to test / run / experiment

### 7.1 Build the knowledge base
```bash
pip install pymupdf            # one-time, for PDF extraction
python -m ckd_risk_app.llm.rag.ingest
```
Prints per-source kept / excluded counts and a summary.

### 7.2 Run the safety audit (the freeze gate)
```bash
python -m ckd_risk_app.llm.rag.audit
```
Exit code 0 = pass. Must pass before freezing the KB or starting Phase 3/7.

### 7.3 Inspect
```bash
python -c "import json,collections; rows=[json.loads(l) for l in open('ckd_risk_app/llm/knowledge_base/processed/chunks.jsonl',encoding='utf-8')]; print(collections.Counter(r['safety_label'] for r in rows))"
```
> Windows: prefix `PYTHONIOENCODING=utf-8` if printing chunk text raises
> `UnicodeEncodeError` (PDF ligatures).

### 7.4 Experiment / tune
- **Re-scope a source:** edit its `page_ranges` in `sources.csv`, re-run, re-audit.
- **Tighten/loosen safety:** edit `EXCLUDE_PATTERNS` in `ingest.py`. If `audit.py`
  ever FAILs, it prints offending chunks → expand the patterns → re-run.
- **Chunk size:** `TARGET_WORDS` / `OVERLAP_WORDS` in `ingest.py`.
- **Add a source:** drop a file in `raw/`, add a `sources.csv` row, re-run + audit.

---

## 8. Design decisions worth remembering

- **Exclude, don't flag, clinical content.** Medication/dialysis/transplant chunks
  are removed from the KB entirely — the safest design for a non-clinical assistant.
- **Numeric nutrition targets are restricted everywhere**, not just in KDOQI.
- **Small chunks = surgical safety.** ~150-word chunks let us keep safe
  patient-education content next to a single clinical sentence.
- **Audit is an independent gate.** `audit.py` re-scans the final file; it is the
  go/no-go before freezing. Keep it green.
- **Registry-driven + reproducible.** `sources.csv` doubles as the paper's citation
  table (title, org, year, url, access date, license note, scope).
- **CDC licensing.** Public domain; the paper should add CDC's
  attribution + non-endorsement disclaimer.

---

## 9. Known open items / before publication

1. **Licensing.** KDIGO/ADA/KDOQI PDFs are copyrighted — raw stored locally only;
   release only permitted excerpts. CDC is public domain (with attribution).
2. **Minor web boilerplate** in a few CDC chunks (e.g. "Download the fact sheet")
   — harmless noise; can be cleaned with a small stop-line filter if desired.
3. **Topic inference is heuristic** — a few chunks get an approximate topic; fine
   for retrieval, hand-correct high-value ones if needed.
4. **Orphaned original NKF PDF** remains in `raw/` (unused); delete if you want a
   tidy raw folder.
5. **Retrieval-relevance gate** (≥80% on 10 test queries) is part of the audit
   philosophy but is implemented in **Phase 3** (needs the retriever).

---

## 10. Next steps

**Phase 3 — RAG Retrieval Pipeline:**
- `embed_index.py` — embed the 118 chunks with **BGE-M3**, build a **FAISS** index
  into `knowledge_base/index/`.
- `retriever.py` — `retrieve(query, k)` from an ML risk output (risk class + top
  factors → human terms), returning top-k chunks and respecting `safety_label`
  (restricted chunks surface only as dietitian-referral).
- Retrieval-relevance check on 10 test queries (≥80%) to complete the audit gate.

Documented in `docs/LLM_PHASE3_RAG_RETRIEVAL.md` when complete.

---

## 11. File map (quick reference)

```
ckd_risk_app/llm/rag/ingest.py                          # python -m ckd_risk_app.llm.rag.ingest
ckd_risk_app/llm/rag/audit.py                           # python -m ckd_risk_app.llm.rag.audit
ckd_risk_app/llm/knowledge_base/sources.csv             # registry + scoping + licensing
ckd_risk_app/llm/knowledge_base/raw/                    # source PDFs + cleaned NKF .md
ckd_risk_app/llm/knowledge_base/processed/chunks.jsonl  # 118 chunks (safety-labeled)
docs/LLM_PHASE2_KNOWLEDGE_BASE.md                       # this file
```

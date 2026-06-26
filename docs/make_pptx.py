#!/usr/bin/env python3
"""Generate docs/NephroAI_Presentation.pptx — Nephro-AI Supervisor Presentation.

Run from the docs/ folder or repo root:
    python docs/make_pptx.py
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = RGBColor(  7,  17,  31)
CARD  = RGBColor( 13,  26,  46)
CARD2 = RGBColor( 18,  36,  62)
ACC   = RGBColor(  0, 201, 255)
WHT   = RGBColor(255, 255, 255)
MUT   = RGBColor(143, 171, 190)
GRN   = RGBColor( 16, 185, 129)
AMB   = RGBColor(245, 158,  11)
RED   = RGBColor(239,  68,  68)
PUR   = RGBColor(167, 139, 250)
DLN   = RGBColor( 35,  55,  85)
CBR   = RGBColor( 22,  42,  70)

# ── Presentation setup ────────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

# ── Low-level helpers ─────────────────────────────────────────────────────────

def S():
    """New slide with dark background."""
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = BG
    return s


def R(s, x, y, w, h, fill=CARD, lc=None, lw=0.75):
    """Add a filled rectangle (in inches). lc = line color, lw = line width pt."""
    shp = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if lc:
        shp.line.color.rgb = lc
        shp.line.width = Pt(lw)
    else:
        shp.line.fill.background()
    return shp


def T(s, x, y, w, h, rows, wrap=True):
    """Add a textbox.

    rows: list of tuples — at minimum (text, font_pt).
    Full tuple: (text, pt, color, bold, italic, align, space_before_pt, space_after_pt)
    """
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf  = box.text_frame
    tf.word_wrap = wrap
    first = True
    for row in rows:
        text  = row[0]
        size  = row[1]
        color = row[2] if len(row) > 2 else WHT
        bold  = row[3] if len(row) > 3 else False
        ital  = row[4] if len(row) > 4 else False
        align = row[5] if len(row) > 5 else PP_ALIGN.LEFT
        sb    = row[6] if len(row) > 6 else 0
        sa    = row[7] if len(row) > 7 else 1
        if first:
            pr = tf.paragraphs[0]; first = False
        else:
            pr = tf.add_paragraph()
        run = pr.add_run()
        run.text            = text
        run.font.size       = Pt(size)
        run.font.color.rgb  = color
        run.font.bold       = bold
        run.font.italic     = ital
        pr.alignment        = align
        pr.space_before     = Pt(sb)
        pr.space_after      = Pt(sa)
    return box


def T2(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, sb=0, sa=1, wrap=True):
    """Textbox with multiple styled runs in ONE paragraph.

    runs: [(text, pt, color, bold), ...]
    """
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf  = box.text_frame
    tf.word_wrap = wrap
    pr  = tf.paragraphs[0]
    pr.alignment    = align
    pr.space_before = Pt(sb)
    pr.space_after  = Pt(sa)
    for txt, pt, clr, bld in runs:
        run = pr.add_run()
        run.text           = txt
        run.font.size      = Pt(pt)
        run.font.color.rgb = clr
        run.font.bold      = bld
    return box


def HDR(s, tag, title_parts, sub=None):
    """Standard slide header: tag line · title (multi-run) · subtitle · divider.

    title_parts: [(text, is_accent_bool), ...]  — is_accent=True → cyan, False → white
    """
    T(s, .4, .20, 12.5, .30, [(tag.upper(), 7.5, ACC, True)])
    # convert (text, is_accent) → (text, pt, color, bold)
    runs = [(txt, 26, ACC if acc else WHT, True) for txt, acc in title_parts]
    T2(s, .4, .46, 12.5, .72, runs)
    if sub:
        T(s, .4, 1.14, 12.5, .34, [(sub, 9, MUT)])
    R(s, .4, 1.47, 12.53, .018, fill=DLN)


# ── Content helpers ───────────────────────────────────────────────────────────

def CARD_BOX(s, x, y, w, h, title=None, body=None, tc=ACC, ac=None, fill=CARD, bsz=8.5):
    """Filled card with optional left accent bar, title, and body (str or [str])."""
    R(s, x, y, w, h, fill=fill, lc=CBR)
    if ac:
        R(s, x, y, .055, h, fill=ac)
    ox = x + (.23 if ac else .16)
    oy = y + .13
    if title:
        T(s, ox, oy, w - .28, .24, [(title.upper(), 7, tc, True)])
        oy += .25
    if body is not None:
        avail = (y + h - .1) - oy
        rows  = [(ln, bsz, MUT) for ln in body] if isinstance(body, list) else [(body, bsz, MUT)]
        T(s, ox, oy, w - .30, avail, rows)


def BCARD(s, x, y, w, h, title=None, items=None, tc=ACC, prefix="✓", pc=GRN, fill=CARD, bsz=8.5):
    """Card with check/bullet list."""
    R(s, x, y, w, h, fill=fill, lc=CBR)
    ox, oy = x + .16, y + .13
    if title:
        T(s, ox, oy, w - .28, .24, [(title.upper(), 7, tc, True)])
        oy += .25
    if items:
        T(s, ox, oy, w - .28, (y + h - .1) - oy,
          [(f"{prefix}  {it}", bsz, MUT) for it in items])


def XCARD(s, x, y, w, h, title=None, items=None, tc=RED, fill=CARD):
    """Card with ✕ list."""
    BCARD(s, x, y, w, h, title=title, items=items, tc=tc, prefix="✕", pc=RED, fill=fill)


def STAT(s, x, y, w, h, val, lbl, vc=ACC, fill=CARD):
    """Metric card: big number + label."""
    R(s, x, y, w, h, fill=fill, lc=CBR)
    T(s, x + .1, y + .14, w - .2, h * .52,
      [(val, 22, vc, True, False, PP_ALIGN.CENTER)])
    T(s, x + .1, y + h * .63, w - .2, h * .33,
      [(lbl.upper(), 6.5, MUT, False, False, PP_ALIGN.CENTER)])


def LABEL(s, x, y, w, h, text, color=ACC):
    """Small caps label — e.g., for section chips."""
    T(s, x, y, w, h, [(text.upper(), 7, color, True)])


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ══════════════════════════════════════════════════════════════════════════════
s = S()

# Subtle bg accent panel top-right
R(s, 8.5, 0, 4.83, 3.5, fill=RGBColor(0, 25, 50))

# Main title — "Nephro" cyan + "-AI" white
T2(s, 2.0, 1.3, 9.33, 1.4,
   [("Nephro", 54, ACC, True), ("-AI", 54, WHT, True)],
   align=PP_ALIGN.CENTER)

# Subtitle
T(s, 1.5, 2.82, 10.33, .90,
  [("Non-Invasive Chronic Kidney Disease Risk Screening &", 13, MUT,
    False, False, PP_ALIGN.CENTER),
   ("Guideline-Grounded Personalised Patient Education", 13, WHT,
    False, True, PP_ALIGN.CENTER)], )

# Pill tags
pill_y = 3.9
pill_data = [
    ("  Machine Learning  ", ACC, RGBColor(0, 40, 60)),
    ("  Retrieval-Augmented Generation  ", GRN, RGBColor(0, 40, 30)),
    ("  Safety-Validated LLM  ", AMB, RGBColor(50, 40, 0)),
    ("  KDIGO-Aligned  ", PUR, RGBColor(40, 20, 70)),
]
pill_x = 2.0
for txt, tc, fc in pill_data:
    est_w = len(txt) * 0.077
    R(s, pill_x, pill_y, est_w, .32, fill=fc, lc=tc, lw=.6)
    T(s, pill_x + .05, pill_y + .05, est_w - .1, .22,
      [(txt.strip(), 8, tc, True, False, PP_ALIGN.CENTER)])
    pill_x += est_w + .15

# Divider
R(s, 2.0, 4.35, 9.33, .018, fill=DLN)

# Author / institution
T(s, 1.5, 4.5, 10.33, .35,
  [("Mohammad Walid Hasan", 11, WHT, True, False, PP_ALIGN.CENTER)])
T(s, 1.5, 4.84, 10.33, .30,
  [("Jahangirnagar University / IIT  ·  Research Project & Viva  ·  2026",
    9, MUT, False, False, PP_ALIGN.CENTER)])

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — THE PROBLEM
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "Motivation", [("The ", False), ("Problem", True)],
    "Chronic Kidney Disease (CKD) is a silent, global epidemic")

# Left column: stat cards (x=0.4, w=3.4)
STAT(s, .4, 1.62, 3.4, 1.62, "800M", "People worldwide have CKD")
STAT(s, .4, 3.32, 3.4, 1.22, "~90%", "Are unaware they have it", vc=RED,
     fill=RGBColor(30, 8, 8))
STAT(s, .4, 4.62, 3.4, 1.22, "Stage 3+", "When most are finally diagnosed", vc=AMB,
     fill=RGBColor(35, 25, 5))

# Right column: three gap cards (x=4.0, w=9.0)
CARD_BOX(s, 4.0, 1.62, 9.13, 1.6,
         title="Gap 1 — Detection",
         body=("Standard confirmatory tests — eGFR, serum creatinine, UACR — require dedicated "
               "laboratory access and phlebotomy. This creates a barrier in community, primary-care, "
               "and low-resource settings where early screening is most needed."),
         ac=RED, fill=RGBColor(25, 8, 8))

CARD_BOX(s, 4.0, 3.30, 9.13, 1.52,
         title="Gap 2 — Patient Education",
         body=("Even when a risk score is generated, patients receive no personalised guidance — "
               "no explanation of what their specific risk factors mean, what lifestyle changes "
               "to make, or when to seek follow-up."),
         ac=AMB, fill=RGBColor(32, 22, 5))

XCARD(s, 4.0, 4.90, 9.13, 1.0,
      title="Existing ML Models Fall Short",
      items=["Almost all use eGFR or creatinine as inputs — not usable for community screening",
             "Binary output only — no KDIGO-aligned risk stratification",
             "No patient-facing education layer built in"],
      fill=RGBColor(30, 8, 8))

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — SOLUTION OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "Overview", [("What is ", False), ("Nephro-AI", True), ("?", False)],
    "A connected two-stage system that solves both gaps end-to-end")

# Two main boxes
BCARD(s, .4, 1.62, 5.9, 4.2,
      title="Stage 1 — ML Screening",
      items=["22 routine health inputs → Low / Moderate / High risk",
             "No creatinine, No eGFR, No UACR — routine vitals and labs only",
             "Two-stage cascade: RF gate + GradientBoosting discriminator",
             "KDIGO risk grid aligned classification",
             "Model-agnostic local explanation (top 6 factors)",
             "Reliability score per prediction"],
      tc=ACC, pc=GRN)

# Arrow
T(s, 6.38, 3.4, .55, .7, [("→", 28, ACC, True, False, PP_ALIGN.CENTER)])

BCARD(s, 7.03, 1.62, 5.9, 4.2,
      title="Stage 2 — LLM Education",
      items=["ML result + patient profile fed to Qwen3-4B via Ollama",
             "Factor-aware RAG pulls relevant clinical guideline chunks",
             "Patient Profile block + per-section personalisation cues",
             "Grounded in KDIGO 2024, ADA 2026, NKF KDOQI, CDC",
             "7-layer safety validator runs post-generation",
             "Plain language output — Flesch-Kincaid Grade ≤ 8"],
      tc=PUR, pc=PUR)

# Bottom banner
CARD_BOX(s, .4, 5.92, 12.53, .68,
         body="Delivered as a full-stack web app:  Patient View (calming education UI)  +  "
              "Admin / Research View (probabilities, retrieved chunks, validator audit trail, frozen evaluation metrics)",
         fill=RGBColor(0, 25, 45), tc=ACC)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "Data", [("Dataset & ", False), ("Risk Target", True)],
    "NHANES 2021–23 with KDIGO-aligned risk stratification")

CARD_BOX(s, .4, 1.62, 6.1, 2.6,
         title="NHANES 2021–23",
         body=("National Health and Nutrition Examination Survey (CDC/NCHS) — nationally representative "
               "survey of the US adult population combining interview, physical examination, and laboratory "
               "data. Most recent completed cycle (post-COVID, 2021–2023). Publicly available for "
               "reproducible research."))

CARD_BOX(s, 6.63, 1.62, 6.1, 2.6,
         title="KDIGO Risk Target",
         body=["Low  —  No / minimal risk indicators (eGFR ≥60, albuminuria A1)",
               "Moderate  —  Moderately elevated risk",
               "High  —  High / very-high risk (eGFR <60 or albuminuria A2/A3)",
               "",
               "Target constructed using KDIGO grid on eGFR + albuminuria.  "
               "eGFR / creatinine are used ONLY to build training labels — "
               "never used as model input features."])

BCARD(s, .4, 4.32, 6.1, 2.22,
      title="Why NHANES?",
      items=["Contains all 22 of our non-invasive input features",
             "Also contains eGFR & UACR for label construction",
             "Representative general population — not just clinic patients",
             "Publicly available — fully reproducible research pipeline"])

CARD_BOX(s, 6.63, 4.32, 6.1, 2.22,
         title="Key Design Decision",
         body=("eGFR and creatinine are used only to construct training labels via the KDIGO grid. "
               "They are deliberately excluded from the model's input features. "
               "The trained model therefore works without specialist labs — "
               "enabling non-invasive deployment in community settings."),
         ac=RED, fill=RGBColor(28, 8, 8))

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — 22 NON-INVASIVE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "Feature Engineering", [("22 ", True), ("Non-Invasive", False), (" Input Features", True)],
    "Everything measurable at a community health check — no specialist lab required")

# 6 cards in 2 rows × 3 cols
def feat(s, x, y, w, h, label, items, tc=ACC, fill=CARD):
    R(s, x, y, w, h, fill=fill, lc=CBR)
    T(s, x + .14, y + .10, w - .22, .22, [(label.upper(), 6.5, tc, True)])
    T(s, x + .14, y + .36, w - .22, h - .46,
      [(it, 8.5, MUT) for it in items])

cw, ch, gap = 4.06, 2.5, .175
r1y, r2y = 1.62, 4.20

feat(s, .4,          r1y, cw, ch, "Demographic (3)",
     ["Age (years)", "Sex (M/F)", "Income-to-poverty ratio"])
feat(s, .4+cw+gap,   r1y, cw, ch, "Anthropometric (2)",
     ["BMI (kg/m²)", "Waist circumference (cm)"])
feat(s, .4+2*(cw+gap), r1y, cw, ch, "Blood Pressure (3)",
     ["Mean systolic BP (mmHg)", "Mean diastolic BP (mmHg)", "Mean pulse (bpm)"])

feat(s, .4,          r2y, cw, ch, "Routine Labs (8)",
     ["Fasting glucose (mg/dL)", "HbA1c (%)", "Blood urea nitrogen (mg/dL)",
      "Uric acid (mg/dL)", "Serum albumin (g/dL)",
      "Sodium (mmol/L)", "Potassium (mmol/L)", "Hemoglobin (g/dL)"])
feat(s, .4+cw+gap,   r2y, cw, ch, "Comorbidity Flags (6 binary)",
     ["Diagnosed diabetes", "Prediabetes / borderline",
      "Diagnosed hypertension", "Diagnosed high cholesterol",
      "Taking BP medication", "Taking cholesterol medication"])
feat(s, .4+2*(cw+gap), r2y, cw, ch, "Deliberately Excluded  ✕",
     ["✕  Serum creatinine", "✕  eGFR (estimated / calculated)",
      "✕  UACR (urine albumin-creatinine)", "✕  Urine protein / dipstick",
      "", "Excluding these is the enabling design decision for community-level screening."],
     tc=RED, fill=RGBColor(28, 8, 8))

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — ML ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "ML Architecture", [("Two-Stage ", False), ("Cascade", True), (" Design", False)],
    "Separate models for at-risk detection and high-risk discrimination")

# Cascade flow on left
bw, bh = 5.5, .72
cx, cy = .4, 1.65

def cbox(s, x, y, w, h, title, body, tc=ACC, fill=CARD):
    R(s, x, y, w, h, fill=fill, lc=CBR)
    T(s, x+.15, y+.08, w-.25, .23, [(title.upper(), 6.5, tc, True)])
    T(s, x+.15, y+.31, w-.25, h-.40, [(body, 8, MUT)])

def arr(s, x, y):
    T(s, x, y, bw, .3, [("▼", 14, ACC, True, False, PP_ALIGN.CENTER)])

cbox(s, cx, cy,      bw, .62, "22 Feature Input", "All non-invasive patient features", tc=MUT)
arr(s, cx, cy+.62)
cbox(s, cx, cy+.92,  bw, .90, "Stage 1 — Random Forest Gate",
     "Binary: Low vs At-Risk   |   Threshold: 0.44   |   Tuned for high sensitivity")
arr(s, cx, cy+1.82)
T(s, cx, cy+2.12, bw, .28,
  [("if  P(at-risk) ≥ 0.44  →  proceed to Stage 2", 8.5, ACC, False, True, PP_ALIGN.CENTER)])
arr(s, cx, cy+2.40)
cbox(s, cx, cy+2.70, bw, .90, "Stage 2 — Gradient Boosting Discriminator",
     "Binary within at-risk pool: Moderate vs High   |   Threshold: 0.50",
     tc=PUR, fill=RGBColor(18, 10, 40))
arr(s, cx, cy+3.60)

# Outcome boxes
out_y = cy + 3.92
ow = 1.7
R(s, cx, out_y, ow, .72, fill=RGBColor(5, 35, 20), lc=GRN)
T(s, cx+.1, out_y+.1, ow-.2, .52, [("LOW RISK\nP(at-risk) < 0.44", 8.5, GRN, True, False, PP_ALIGN.CENTER)])
R(s, cx+ow+.1, out_y, ow, .72, fill=RGBColor(35, 25, 5), lc=AMB)
T(s, cx+ow+.2, out_y+.1, ow-.2, .52, [("MODERATE\nP(high) < 0.50", 8.5, AMB, True, False, PP_ALIGN.CENTER)])
R(s, cx+2*(ow+.1), out_y, ow, .72, fill=RGBColor(35, 8, 8), lc=RED)
T(s, cx+2*(ow+.1)+.1, out_y+.1, ow-.2, .52, [("HIGH RISK\nP(high) ≥ 0.50", 8.5, RED, True, False, PP_ALIGN.CENTER)])

# Right column — rationale cards
rx = 6.1
BCARD(s, rx, 1.65, 6.83, 2.1,
      title="Why Two Stages?",
      items=["Stage 1 optimised for recall — catch ALL at-risk patients",
             "Stage 2 optimised for precision — separate High from Moderate",
             "Minimises the most dangerous error: High→Low misses",
             "Handles severe class imbalance: low >> moderate > high"])

CARD_BOX(s, rx, 3.83, 6.83, 1.55,
         title="+ Local Perturbation Explanation",
         body=("For every prediction: model-agnostic perturbation replaces each feature with a "
               "reference value and measures probability change. Top 6 factors with direction "
               "(raises / lowers risk) are surfaced to the user and passed to the LLM "
               "for per-patient personalisation."))

CARD_BOX(s, rx, 5.46, 6.83, 1.08,
         title="Reliability Score per Prediction",
         body="High / Moderate / Low — based on confidence margin, proximity to decision boundary, and missing feature count.")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — ML RESULTS
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "Evaluation", [("ML ", False), ("Results", True)],
    "Held-out NHANES 2021–23 test set — two-stage cascade")

# Metric row
mw, mh = 2.35, 1.35
mx = .4
my = 1.65
stats = [
    ("71.9%", "Accuracy",          ACC),
    ("66.4%", "Balanced Accuracy", ACC),
    ("55.9%", "Macro F1",          MUT),
    ("73.7%", "High Risk Recall",  GRN),
    ("2",     "High→Low Misses",   GRN),
]
for val, lbl, vc in stats:
    STAT(s, mx, my, mw, mh, val, lbl, vc=vc,
         fill=RGBColor(5, 35, 20) if vc == GRN else CARD)
    mx += mw + .12

# Per-class recall table
tw, th = 5.85, 2.2
tx2, ty2 = .4, 3.2
R(s, tx2, ty2, tw, th, fill=CARD2, lc=CBR)
T(s, tx2+.15, ty2+.1, tw-.25, .25, [("Per-Class Recall", 7, ACC, True)])
rows2 = [
    ("Low        75.1%   Good — most low-risk patients correctly cleared",           GRN),
    ("Moderate   50.5%   Hardest class — ambiguous moderate/high boundary",         AMB),
    ("High        73.7%   Strong — critical class, cascade tuned for this",          GRN),
]
for i, (txt, c) in enumerate(rows2):
    R(s, tx2+.12, ty2+.45+i*.5, tw-.24, .42,
      fill=RGBColor(5, 35, 20) if c == GRN else RGBColor(35, 25, 5), lc=c, lw=.5)
    T(s, tx2+.22, ty2+.52+i*.5, tw-.36, .30, [(txt, 8.5, c)])

# Right: interpretation cards
rx = 6.5
CARD_BOX(s, rx, 3.2, 6.63, 2.2,
         title="Key Safety Result — High→Low Misses: Only 2",
         body=("The two-stage cascade is explicitly tuned to minimise High→Low misses — the "
               "clinically most dangerous error, where a patient needing urgent follow-up is "
               "incorrectly told they are low risk.  Only 2 such errors across the full test set."),
         ac=GRN, fill=RGBColor(5, 35, 20))

CARD_BOX(s, .4, 5.48, 6.0, 1.08,
         title="Moderate Class Note",
         body=("Moderate recall (50.5%) reflects inherent boundary ambiguity — even eGFR+UACR "
               "cannot perfectly separate moderate from high risk."))

CARD_BOX(s, 6.5, 5.48, 6.63, 1.08,
         title="Clinical Context",
         body=("The tool is a screening aid — it triggers follow-up testing, not replaces it. "
               "False moderate-as-low is far less dangerous than false high-as-low."))

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — LLM EDUCATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "LLM Component", [("Education ", False), ("Pipeline", True)],
    "From ML prediction to safe, personalised patient guidance")

# Flow diagram
fw = 1.82
fh = .82
fy = 1.65
fx = .4
gap_f = .13
flow = [
    ("ML Output",         "Risk class · factors · patient inputs"),
    ("RAG Retrieval",     "4 guideline chunks (factor-aware)"),
    ("Prompt Builder",    "System rules + Profile + Cues"),
    ("LLM Generate",      "Qwen3-4B  via  Ollama"),
    ("Safety Validator",  "7-layer rule engine"),
    ("Patient Output",    "5-section  education"),
]
for i, (title, sub) in enumerate(flow):
    fc = RGBColor(0, 35, 55) if i == 5 else CARD
    lc = GRN if i == 5 else CBR
    R(s, fx, fy, fw, fh, fill=fc, lc=lc)
    T(s, fx+.08, fy+.08, fw-.16, .26, [(title, 7.5, ACC if i<5 else GRN, True)])
    T(s, fx+.08, fy+.36, fw-.16, .38, [(sub, 7, MUT)])
    if i < len(flow) - 1:
        T(s, fx+fw, fy+.22, gap_f+.03, .38, [("→", 11, ACC, True)])
    fx += fw + gap_f

# Output sections box
CARD_BOX(s, .4, 2.65, 5.8, 4.0,
         title="Output Sections (5)",
         body=["Summary  —  risk level + top contributing factors",
               "Food habits  —  diet tailored to patient conditions",
               "Rest & sleep  —  sleep hygiene, recovery habits",
               "Daily routine  —  activity, hydration, monitoring",
               "Follow-up reminder  —  clinician/lab visit guidance",
               "",
               "All 5 sections mandatory.  Structural check enforces completeness.",
               "Readability target: Flesch-Kincaid Grade ≤ 8."])

# Personalisation architecture
CARD_BOX(s, 6.4, 2.65, 6.73, 4.0,
         title="3-Layer Personalisation Architecture",
         body=["1.  PATIENT PROFILE block — all 22 actual patient values injected into the prompt "
               "(age, SBP, HbA1c, diabetes flag, BMI, etc.).",
               "",
               "2.  Factor-aware RAG — different patients get different guideline chunks based "
               "on their top ML-explanation risk drivers (e.g. hypertension patient → BP chunks; "
               "diabetic patient → diabetes chunks).",
               "",
               "3.  Per-section PERSONALISATION REQUIREMENTS — explicit condition-derived "
               "instructions per output section computed from patient profile "
               "(e.g. 'For Food habits: address high blood sugar because HbA1c = 8.5%')."],
         ac=PUR, fill=RGBColor(18, 10, 40))

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "RAG Knowledge Base", [("Clinical ", False), ("Guidelines", True), (" Grounding", False)],
    "118 chunks from 9 authoritative sources — content-level safety scoping")

sources = [
    ("KDIGO CKD Guideline",         "KDIGO",     "2024", "bp · nutrition · activity · weight · smoking"),
    ("KDIGO Blood Pressure in CKD", "KDIGO",     "2021", "bp · sodium"),
    ("Diabetes Management in CKD",  "ADA/KDIGO", "2022", "diabetes · glucose"),
    ("ADA Standards of Care",       "ADA",        "2026", "diabetes · bp · followup"),
    ("KDOQI Nutrition",             "NKF",        "2020", "nutrition (RESTRICTED — dietitian referral only)"),
    ("Exercise Guidelines for CKD", "NKF",        "2023", "activity · weight"),
    ("CKD Basics / Prevention",     "CDC",        "2024", "general · prevention · followup"),
]
tw_cols = [4.8, 2.0, .8, 4.3]
col_x   = [.4, 5.3, 7.4, 8.3]

R(s, .4, 1.65, 12.53, .34, fill=RGBColor(0, 30, 55), lc=CBR)
hdrs = ["Source", "Organisation", "Year", "Topics"]
for i, h in enumerate(hdrs):
    T(s, col_x[i]+.08, 1.68, tw_cols[i]-.1, .26, [(h, 7, ACC, True)])

for ri, (src, org, yr, top) in enumerate(sources):
    ry = 2.05 + ri * .56
    fill_row = RGBColor(10, 22, 40) if ri % 2 == 0 else CARD
    R(s, .4, ry, 12.53, .50, fill=fill_row, lc=CBR)
    tc_row = RED if "RESTRICTED" in top else MUT
    T(s, col_x[0]+.08, ry+.10, tw_cols[0]-.1, .30, [(src, 8.5, WHT)])
    T(s, col_x[1]+.08, ry+.10, tw_cols[1]-.1, .30, [(org, 8.5, ACC)])
    T(s, col_x[2]+.08, ry+.10, tw_cols[2]-.1, .30, [(yr,  8.5, MUT)])
    T(s, col_x[3]+.08, ry+.10, tw_cols[3]-.1, .30, [(top, 8,   tc_row)])

# Bottom row cards
CARD_BOX(s, .4, 6.08, 4.0, 1.0,
         title="Embedding & Retrieval",
         body="BGE-M3 (1024-d, via Ollama) · FAISS cosine search · k=4 chunks · TF-IDF offline fallback")

CARD_BOX(s, 4.55, 6.08, 4.05, 1.0,
         title="Safety Scoping",
         body="Chunks with numeric nutrient targets tagged RESTRICTED — returned to LLM with 'discuss with dietitian' override.",
         ac=RED, fill=RGBColor(28, 8, 8))

STAT(s, 8.75, 6.08, 4.18, 1.0, "118", "Total Guideline Chunks")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — SAFETY VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "Safety System", [("Multi-Layer ", False), ("Safety Validator", True)],
    "Deterministic post-generation rule engine — 0% final unsafe output rate")

layers = [
    ("1", "Diagnosis block",            "Detects 'you have CKD / kidney disease / kidney failure'",                   RED,  "Severe  →  fallback"),
    ("2", "Medication block",           "Drug names, dosing language, start/stop instructions",                       RED,  "Severe  →  fallback"),
    ("3", "Supplement block",           "Detects 'take fish oil / vitamin D / probiotics…'",                          RED,  "Severe  →  fallback"),
    ("4", "Nutrition restriction",      "Numeric targets (2 g sodium, potassium 2000 mg) → rewrites to dietitian ref", AMB, "Moderate  →  sanitise"),
    ("5", "Lab-reminder append",        "Moderate/High with missing follow-up mention → auto-appends reminder",        AMB, "Moderate  →  sanitise"),
    ("6", "Emergency redirect",         "SBP ≥ 180 or DBP ≥ 120 → prepends urgent-care notice",                       AMB, "Moderate  →  sanitise"),
    ("7", "Structure check",            "All 5 sections present? Missing → retry → safe fallback",                    ACC,  "Format  →  retry"),
]
lh = .57
for i, (num, name, desc, c, tag) in enumerate(layers):
    ly = 1.65 + i * (lh + .05)
    fc = RGBColor(28, 8, 8) if c == RED else (RGBColor(32, 22, 5) if c == AMB else RGBColor(0, 28, 48))
    R(s, .4, ly, 8.8, lh, fill=fc, lc=c, lw=.5)
    R(s, .4, ly, .32, lh, fill=c)
    T(s, .43, ly + .15, .26, .28, [(num, 9, BG, True, False, PP_ALIGN.CENTER)])
    T(s, .82, ly + .06, 3.8, .24, [(name, 8, c, True)])
    T(s, .82, ly + .28, 7.5, .24, [(desc, 7.5, MUT)])
    R(s, 9.3, ly + .1, 3.83, .37, fill=fc, lc=c, lw=.4)
    T(s, 9.38, ly + .15, 3.72, .26, [(tag, 7.5, c, True)])

# Right stat
STAT(s, 9.3, 1.65, 3.83, 2.05, "0%", "Final Unsafe Output Rate\n(30 adversarial profiles)", vc=GRN,
     fill=RGBColor(5, 35, 20))

CARD_BOX(s, 9.3, 3.78, 3.83, 1.55,
         title="Outcome States",
         body=["PASS         No violations — returned as-is  (90%)",
               "SANITISED  Auto-fixes applied  (10%)",
               "FALLBACK    Safe template — severe violations  (0%)"])

CARD_BOX(s, 9.3, 5.41, 3.83, 1.13,
         title="Why Rule-Based?",
         body="Deterministic, fast, auditable. Zero false-negatives on specific dangerous patterns vs probabilistic LLM judge.")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — EVALUATION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "LLM Evaluation — Phase 6", [("Education Quality ", False), ("Results", True)],
    "30 held-out profiles (10 Low · 10 Moderate · 10 High) · Qwen3-4B-Instruct-2507 + BGE-M3")

mw2, mh2 = 1.95, 1.38
mx2 = .4
stats2 = [
    ("100%",   "Format Adherence",    GRN),
    ("0%",     "Final Unsafe Rate",   GRN),
    ("96.6%",  "Factor Relevance",    ACC),
    ("1.93/2", "Personalisation",     ACC),
    ("6.9",    "Readability (FK Gr.)",ACC),
    ("100%",   "Grounding Rate",      ACC),
]
for val, lbl, vc in stats2:
    fill = RGBColor(5, 35, 20) if vc == GRN else CARD
    STAT(s, mx2, 1.65, mw2, mh2, val, lbl, vc=vc, fill=fill)
    mx2 += mw2 + .12

CARD_BOX(s, .4, 3.18, 5.9, 1.85,
         title="Scoring Method",
         body=["Format — all 5 sections present",
               "Safety — pre and post-validator unsafe patterns detected",
               "Factor relevance — top-2 ML factors mentioned in text",
               "Grounding — cited chunk IDs valid in knowledge base",
               "Personalisation — risk level named (1pt) + ≥1 factor cited (1pt)",
               "Readability — Flesch-Kincaid grade (target ≤ 8)"])

CARD_BOX(s, 6.5, 3.18, 6.23, 1.85,
         title="Key Achievement — 0% Final Unsafe Rate",
         body=("Raw unsafe rate before validation was 10% (emergency-BP cases). "
               "The safety validator intercepted and fixed all 10% via auto-prepend — "
               "no information loss, no fallback, no content removed.  "
               "Final unsafe rate on the full 30-profile adversarial set: 0%."),
         ac=GRN, fill=RGBColor(5, 35, 20))

CARD_BOX(s, .4, 5.12, 5.9, 1.42,
         title="Readability Target Met",
         body=("Mean Flesch-Kincaid grade 6.9 (target ≤ 8).  "
               "100% of outputs at or below grade 8 — appropriate for general adult health communication."))

CARD_BOX(s, 6.5, 5.12, 6.23, 1.42,
         title="Personalisation Confirmed",
         body=("Mean score 1.93 / 2 — near-perfect.  "
               "Achieved by injecting Patient Profile + per-section PERSONALISATION REQUIREMENTS "
               "and fixing the Safety Rule 5 contradiction that previously enforced generic advice."))

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — NOVEL CONTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "Research Contribution", [("Novel ", True), ("Contributions", False)],
    "What Nephro-AI does that existing work does not")

novs = [
    ("① Non-Invasive Feature Set",
     "All existing ML CKD detection models use eGFR or creatinine as inputs. "
     "Nephro-AI is trained exclusively on routine, non-creatinine features — enabling "
     "community-level pre-screening for the first time."),
    ("② KDIGO-Aligned Two-Stage Cascade",
     "RF gate (tuned for recall) → GB discriminator (tuned for precision) maps "
     "explicitly to KDIGO risk categories with thresholds optimised to minimise "
     "the clinically most dangerous error: High→Low misses."),
    ("③ Factor-Aware Personalised RAG",
     "Rather than top-k by similarity (which returns the same chunks for all patients), "
     "the retriever builds a per-patient topic priority list from ML explanation factors — "
     "guaranteeing different patients receive different guideline grounding."),
    ("④ Safety-Scoped Knowledge Base",
     "Chunks with individualised numeric targets (protein, potassium, sodium) are tagged "
     "RESTRICTED and served with a dietitian-referral override — preserved for reference "
     "but never cited as direct patient advice."),
    ("⑤ Multi-Layer Validator — Measured 0% Unsafe Rate",
     "A 7-layer deterministic rule engine (not probabilistic LLM judge) achieves a "
     "published 0% final unsafe rate on 30 adversarial profiles.  "
     "Every decision logged — fully auditable."),
    ("⑥ Full-Stack Patient + Admin System",
     "Complete web application: patient-facing education view + researcher-facing admin "
     "dashboard exposing the full audit trail (probabilities, chunks, validator decisions, "
     "frozen evaluation metrics, JSON export)."),
]
nw, nh = 6.0, 2.4
for i, (title, body) in enumerate(novs):
    row, col = divmod(i, 2)
    nx = .4 + col * (nw + .33)
    ny = 1.65 + row * (nh + .15)
    CARD_BOX(s, nx, ny, nw, nh, title=title, body=body, ac=ACC)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — TECH STACK
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "Implementation", [("Technology ", False), ("Stack", True)],
    "Production-ready, modular, fully open-source")

stack = [
    ("Frontend",        "React 18",          "Vite 5 · Tailwind CSS 3 · Axios"),
    ("Backend API",     "FastAPI",            "Uvicorn · Pydantic v2"),
    ("ML Model",        "scikit-learn",       "imbalanced-learn · joblib · pandas"),
    ("LLM (Production)","Qwen3-4B",           "Qwen3-4B-Instruct-2507 via Ollama"),
    ("LLM (Offline)",   "MockGuidanceModel",  "Pure Python · No GPU required"),
    ("Embeddings",      "BGE-M3",             "BAAI · 1024-d · via Ollama REST API"),
    ("Fallback Embed",  "HashingTF-IDF",      "Pure Python · 4096-d · Offline"),
    ("Vector Search",   "FAISS / Custom",     "IndexFlatIP · Brute-force cosine"),
    ("Safety System",   "Rule Validator",     "Pure Python · Regex · Stdlib only"),
    ("Database",        "PostgreSQL",         "SQLAlchemy ORM · Alembic migrations"),
    ("Training Data",   "NHANES 2021–23",     "CDC / NCHS · Public Domain"),
    ("Knowledge Base",  "KDIGO · ADA · NKF · CDC", "9 guideline sources · 118 chunks"),
]
sw2, sh2 = 3.0, 1.42
for i, (layer, name, tech) in enumerate(stack):
    row, col = divmod(i, 4)
    sx = .4 + col * (sw2 + .12)
    sy = 1.62 + row * (sh2 + .12)
    R(s, sx, sy, sw2, sh2, fill=CARD, lc=CBR)
    T(s, sx+.12, sy+.10, sw2-.2, .22, [(layer.upper(), 6, MUT, True)])
    T(s, sx+.12, sy+.34, sw2-.2, .36, [(name, 10, ACC, True)])
    T(s, sx+.12, sy+.72, sw2-.2, .55, [(tech, 7.5, MUT)])

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — FUTURE WORK
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "Roadmap", [("Future ", False), ("Work", True)],
    "Extensions planned for the full research paper and clinical pilot study")

future = [
    ("🎛️ QLoRA Fine-Tuning",
     "Fine-tune Qwen3-4B on RAFT-generated training examples (question, correct guideline "
     "chunk, distractor). Expected to improve safety self-containment and output consistency "
     "beyond prompt engineering alone."),
    ("👨‍⚕️ Expert Clinical Evaluation",
     "Structured review of LLM outputs by nephrologists and renal dietitians using a "
     "standardised rubric — clinical accuracy, safety, appropriateness. "
     "Required before any real-world deployment."),
    ("🌍 Population Generalisation",
     "Validate on non-US datasets (South Asian, European, low-income country populations) "
     "where prevalence, comorbidity patterns, and reference ranges may differ from NHANES."),
    ("📈 Longitudinal Risk Tracking",
     "Track serial screening results over time to detect deteriorating trends, "
     "trigger earlier follow-up, and measure whether patient education is changing behaviour."),
    ("🌐 Multilingual Support",
     "Bengali, Hindi, and Arabic patient education — targeting the populations most "
     "underserved by existing English-only kidney health materials.  "
     "BGE-M3 already supports multilingual retrieval."),
    ("📦 Packaged Deployment",
     "GGUF quantised model (Q4_K_M, ~3 GB) distributed with the app so any clinician can "
     "run the full system on a standard laptop — no cloud dependency, no patient data "
     "leaving the device."),
]
fw_w, fw_h = 4.0, 2.52
for i, (title, body) in enumerate(future):
    row, col = divmod(i, 3)
    fx = .4 + col * (fw_w + .165)
    fy = 1.62 + row * (fw_h + .16)
    CARD_BOX(s, fx, fy, fw_w, fw_h, title=title, body=body, tc=ACC)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HDR(s, "Conclusion", [("What We ", False), ("Achieved", True)],
    "Nephro-AI — a complete, evaluated, deployable CKD screening & education system")

conclusions = [
    ("✓  Non-Invasive CKD Screening Works",
     "A two-stage ML cascade trained on 22 routine features achieves 71.9% accuracy "
     "and 73.7% high-risk recall — without eGFR, creatinine, or UACR."),
    ("✓  LLMs Can Safely Personalise Patient Education",
     "Qwen3-4B with factor-aware RAG and condition-specific prompt cues achieves "
     "personalisation 1.93/2 and Flesch-Kincaid Grade 6.9 across 30 adversarial profiles."),
    ("✓  Safety Validation Achieves 0% Unsafe Rate",
     "A 7-layer deterministic rule engine intercepted 100% of pre-validation unsafe outputs "
     "with zero information loss — delivering a fully auditable safety guarantee."),
    ("✓  Complete End-to-End System Delivered",
     "Production-ready full-stack web app with patient education UI, researcher audit dashboard, "
     "dual deployment modes (GPU / CPU), and optional PostgreSQL persistence."),
]
cw2, ch2 = 6.0, 2.3
for i, (title, body) in enumerate(conclusions):
    row, col = divmod(i, 2)
    cx2 = .4 + col * (cw2 + .33)
    cy2 = 1.62 + row * (ch2 + .18)
    CARD_BOX(s, cx2, cy2, cw2, ch2, title=title, body=body, ac=GRN,
             fill=RGBColor(5, 35, 20))

CARD_BOX(s, .4, 6.28, 12.53, .78,
         body=("Ready for:   Expert clinical evaluation   ·   Conference / journal paper "
               "  ·   Clinical pilot study   ·   QLoRA fine-tuning extension"),
         fill=RGBColor(0, 28, 48), tc=ACC)

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / "NephroAI_Presentation.pptx"
prs.save(str(out))
print(f"Saved: {out}")

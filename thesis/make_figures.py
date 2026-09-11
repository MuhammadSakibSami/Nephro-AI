"""Generate all thesis figures from real project data.

Every number here is traceable to project artifacts:
  * ML metrics / confusion matrix -> misc/notebooks/nhanes-model-c-train-save-download-and-predict.ipynb
  * Q1 six-arm results            -> q1_report/tables/q1_per_arm_metrics.csv
  * Fine-tuning loss curve        -> llm/finetune/adapters/v1/training_log.csv

Run:  python thesis/make_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- shared style -----------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

C_LOW, C_MOD, C_HIGH = "#2e7d5b", "#c98a1b", "#b3372e"
C_BLUE, C_GREY, C_LIGHT = "#2f5d8a", "#6b6b6b", "#d9d9d9"
ARM_COLORS = {"A": "#9aa5b1", "B": "#7899ab", "C": "#4a7fb5",
              "D": "#c9a227", "E": "#7a9e3f", "F": "#2e7d5b"}


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p.name)


def box(ax, x, y, w, h, text, fc="white", ec="#333333", fs=8, weight="normal", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, linespacing=1.35, zorder=5)


def arrow(ax, p1, p2, color="#333333", lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color,
                                 shrinkA=1.5, shrinkB=1.5))


def blank_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


# ===========================================================================
# 1. System architecture (three-layer)
# ===========================================================================
def fig_system_architecture():
    fig, ax = blank_axes((7.0, 8.2))

    box(ax, 0.06, 0.885, 0.88, 0.085,
        "PATIENT INPUT LAYER  (React frontend)\n"
        "14 routine inputs: vitals · comorbidity history · routine blood tests\n"
        "Client-side range validation · live completeness indicator",
        fc="#eef3f8", ec=C_BLUE, fs=8)

    box(ax, 0.06, 0.60, 0.88, 0.255,
        "", fc="#fbfbfb", ec="#999999")
    ax.text(0.50, 0.835, "ML SCREENING LAYER  (FastAPI backend)",
            ha="center", fontsize=9, fontweight="bold")
    box(ax, 0.10, 0.715, 0.36, 0.095,
        "Stage 1 — Random Forest\nat-risk gate (13 features)\n"
        r"$p_1 = P(\mathrm{at\ risk})$,  $\tau_1 = 0.44$", fc="white", fs=7.5)
    box(ax, 0.54, 0.715, 0.36, 0.095,
        "Stage 2 — Gradient Boosting\nhigh-risk split (6 features)\n"
        r"$p_2 = P(\mathrm{high}\mid\mathrm{at\ risk})$,  $\tau_2 = 0.50$", fc="white", fs=7.5)
    arrow(ax, (0.46, 0.762), (0.54, 0.762))
    box(ax, 0.10, 0.622, 0.80, 0.072,
        "Local perturbation explanation (top factors)  ·  Reliability heuristic\n"
        "(confidence · probability margin · boundary proximity · missingness)",
        fc="#f4f7fa", fs=7.5)

    box(ax, 0.06, 0.185, 0.88, 0.395, "", fc="#fbfbfb", ec="#999999")
    ax.text(0.50, 0.558, "LLM EDUCATION LAYER  (safety-constrained)",
            ha="center", fontsize=9, fontweight="bold")
    box(ax, 0.09, 0.452, 0.39, 0.079,
        "RAG retrieval\nBGE-M3 embeddings + FAISS\nfactor-aware topic selection, $k=4$",
        fc="white", fs=7.5)
    box(ax, 0.52, 0.452, 0.39, 0.079,
        "Prompt construction\ncompact prompt (fine-tuned) /\nfull prompt (base)", fc="white", fs=7.5)
    box(ax, 0.09, 0.352, 0.82, 0.068,
        "Qwen3-4B-Instruct-2507  +  LoRA adapter (r=16, $\\alpha$=32)\n"
        "temperature 0.3 · top-p 0.9 · max 700 new tokens", fc="#eef6ef", ec=C_LOW, fs=7.5)
    box(ax, 0.09, 0.248, 0.82, 0.072,
        "Deterministic safety validator\n"
        "diagnosis / medication / supplement blockers (severe)  ·  nutrition rewrite,\n"
        "lab reminder, emergency redirect (moderate, auto-fixed)",
        fc="#fdeeec", ec=C_HIGH, fs=7.5)
    box(ax, 0.09, 0.199, 0.82, 0.034,
        "IEEE citation layer — markers restricted to sources actually retrieved",
        fc="#f4f7fa", fs=7.5)

    box(ax, 0.06, 0.055, 0.60, 0.085,
        "PRESENTATION\nPatient view: risk · reliability · main signals · 5-section education\n"
        "Admin view: probabilities · RAG audit · validator audit · evaluation",
        fc="#eef3f8", ec=C_BLUE, fs=6.6)
    box(ax, 0.70, 0.055, 0.24, 0.085,
        "PERSISTENCE\nPostgreSQL screenings\ntable (JSONB)\nbest-effort, non-blocking",
        fc="#f6f2fa", ec="#6a4c93", fs=7.2)

    for y0, y1 in [(0.885, 0.857), (0.60, 0.582), (0.185, 0.142)]:
        arrow(ax, (0.36, y0), (0.36, y1))
    # persistence tap: routed outside the layer boxes so nothing is overdrawn
    ax.plot([0.955, 0.978], [0.66, 0.66], color=C_GREY, lw=0.9)
    ax.plot([0.978, 0.978], [0.66, 0.0975], color=C_GREY, lw=0.9)
    arrow(ax, (0.978, 0.0975), (0.942, 0.0975), color=C_GREY, lw=0.9)
    ax.text(0.9895, 0.40, "prediction record", rotation=90, fontsize=6.4,
            color=C_GREY, va="center", ha="center")
    save(fig, "fig_system_architecture.png")


# ===========================================================================
# 2. End-to-end request pipeline
# ===========================================================================
def fig_pipeline_flow():
    fig, ax = blank_axes((7.0, 4.3))
    steps = [
        ("1. Input\nvalidation", "Pydantic schema,\nrange checks,\nconsistency rule", "#eef3f8"),
        ("2. Feature\nvector", "14 features,\nfitted imputation,\nmissingness tracked", "#eef3f8"),
        ("3. Cascade\nscoring", "$p_1$, $p_2$ →\nclass + calibrated\nprobabilities", "#eef6ef"),
        ("4. Explain +\nreliability", "perturbation factors,\nreliability level", "#eef6ef"),
        ("5. Retrieval", "query → BGE-M3 →\nFAISS → factor-aware\nchunk selection", "#fdf7e8"),
        ("6. Generation", "compact prompt +\nLoRA model →\n5 sections", "#fdf7e8"),
        ("7. Validation", "rule layers →\npass / sanitized /\nfallback", "#fdeeec"),
        ("8. Citation +\ndelivery", "IEEE markers,\nreferences,\nJSON response", "#f4f4f4"),
    ]
    n = len(steps)
    w, h, gap = 0.104, 0.30, 0.014
    x = 0.035
    for i, (title, detail, fc) in enumerate(steps):
        box(ax, x, 0.42, w, h, "", fc=fc, ec="#666666")
        ax.text(x + w / 2, 0.665, title, ha="center", va="center",
                fontsize=7.6, fontweight="bold", linespacing=1.3)
        ax.text(x + w / 2, 0.515, detail, ha="center", va="center",
                fontsize=6.3, linespacing=1.35, color="#333333")
        if i < n - 1:
            arrow(ax, (x + w, 0.57), (x + w + gap, 0.57), lw=0.9)
        x += w + gap

    ax.annotate("", xy=(0.035, 0.36), xytext=(0.965, 0.36),
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.8))
    ax.text(0.19, 0.30, "ML screening layer\n(no network calls)", ha="center",
            fontsize=7, color=C_GREY, linespacing=1.3)
    ax.text(0.60, 0.30, "LLM education layer\n(retrieval + generation)", ha="center",
            fontsize=7, color=C_GREY, linespacing=1.3)
    ax.text(0.885, 0.30, "safety gate\n(always applied)", ha="center",
            fontsize=7, color=C_GREY, linespacing=1.3)
    ax.text(0.50, 0.86, "Single request path: POST /generate-guidance",
            ha="center", fontsize=9, fontweight="bold")
    save(fig, "fig_pipeline_flow.png")


# ===========================================================================
# 3. Two-stage cascade decision flow
# ===========================================================================
def fig_cascade():
    fig, ax = blank_axes((5.6, 5.0))
    box(ax, 0.30, 0.875, 0.40, 0.075,
        "Patient feature vector\n(14 routine inputs)", fc="#eef3f8", ec=C_BLUE, fs=8)
    box(ax, 0.235, 0.715, 0.53, 0.085,
        "Stage 1 — Random Forest (13 features)\n"
        r"$p_1 = P(\mathrm{at\ risk}\mid x)$", fc="white", fs=8)
    ax.add_patch(plt.Polygon([[0.50, 0.685], [0.75, 0.605], [0.50, 0.525], [0.25, 0.605]],
                             closed=True, facecolor="#f7f7f7", edgecolor="#333333", lw=1.0))
    ax.text(0.50, 0.605, r"$p_1 \geq \tau_1 = 0.44$ ?", ha="center", va="center", fontsize=8)

    box(ax, 0.03, 0.415, 0.28, 0.070, "LOW\nscreening category",
        fc="#eaf3ee", ec=C_LOW, fs=8, weight="bold")
    box(ax, 0.235, 0.395, 0.53, 0.085,
        "Stage 2 — Gradient Boosting (6 features)\n"
        r"$p_2 = P(\mathrm{high}\mid \mathrm{at\ risk}, x)$", fc="white", fs=8)
    ax.add_patch(plt.Polygon([[0.50, 0.365], [0.75, 0.288], [0.50, 0.211], [0.25, 0.288]],
                             closed=True, facecolor="#f7f7f7", edgecolor="#333333", lw=1.0))
    ax.text(0.50, 0.288, r"$p_2 \geq \tau_2 = 0.50$ ?", ha="center", va="center", fontsize=8)
    box(ax, 0.06, 0.095, 0.30, 0.070, "MODERATE\nscreening category",
        fc="#fdf5e6", ec=C_MOD, fs=8, weight="bold")
    box(ax, 0.64, 0.095, 0.30, 0.070, "HIGH\nscreening category",
        fc="#fdecea", ec=C_HIGH, fs=8, weight="bold")

    arrow(ax, (0.50, 0.875), (0.50, 0.802))
    arrow(ax, (0.50, 0.715), (0.50, 0.686))
    arrow(ax, (0.25, 0.605), (0.17, 0.605)); arrow(ax, (0.17, 0.605), (0.17, 0.487))
    ax.text(0.195, 0.628, "no", fontsize=7.5, color=C_LOW)
    arrow(ax, (0.50, 0.525), (0.50, 0.482))
    ax.text(0.523, 0.503, "yes", fontsize=7.5, color=C_HIGH)
    arrow(ax, (0.50, 0.395), (0.50, 0.366))
    arrow(ax, (0.25, 0.288), (0.21, 0.288)); arrow(ax, (0.21, 0.288), (0.21, 0.167))
    ax.text(0.14, 0.31, "no", fontsize=7.5, color=C_MOD)
    arrow(ax, (0.75, 0.288), (0.79, 0.288)); arrow(ax, (0.79, 0.288), (0.79, 0.167))
    ax.text(0.805, 0.31, "yes", fontsize=7.5, color=C_HIGH)

    ax.text(0.50, 0.022,
            r"Reported probabilities: $[\,1-p_1,\ \ p_1(1-p_2),\ \ p_1 p_2\,]$, row-normalised",
            ha="center", fontsize=8, style="italic")
    save(fig, "fig_cascade.png")


# ===========================================================================
# 4. Dataset class distribution across splits
# ===========================================================================
SPLITS = {  # notebook cell 6 audit table
    "Train": (3886, 3222, 491, 173),
    "Validation": (833, 691, 105, 37),
    "Development\n(deployed fit)": (4719, 3913, 596, 210),
    "Test\n(held out)": (833, 690, 105, 38),
}


def fig_class_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    names = list(SPLITS)
    lows = [SPLITS[k][1] for k in names]
    mods = [SPLITS[k][2] for k in names]
    highs = [SPLITS[k][3] for k in names]
    y = np.arange(len(names))
    ax1.barh(y, lows, color=C_LOW, label="Low", height=0.6)
    ax1.barh(y, mods, left=lows, color=C_MOD, label="Moderate", height=0.6)
    ax1.barh(y, highs, left=np.array(lows) + np.array(mods), color=C_HIGH,
             label="High", height=0.6)
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis(); ax1.set_xlabel("Participants (n)")
    ax1.set_title("(a) Absolute class counts by split")
    ax1.legend(fontsize=7.5, frameon=False, ncol=3, loc="lower right")
    for i, k in enumerate(names):
        ax1.text(SPLITS[k][0] + 60, i, f"N={SPLITS[k][0]}", va="center", fontsize=7.2)
    ax1.set_xlim(0, 5400)

    dev = SPLITS["Development\n(deployed fit)"]
    frac = [dev[1] / dev[0], dev[2] / dev[0], dev[3] / dev[0]]
    ax2.bar(["Low", "Moderate", "High"], [f * 100 for f in frac],
            color=[C_LOW, C_MOD, C_HIGH], width=0.55)
    for i, f in enumerate(frac):
        ax2.text(i, f * 100 + 1.5, f"{f*100:.1f}%", ha="center", fontsize=8)
    ax2.set_ylabel("Share of development set (%)")
    ax2.set_ylim(0, 95)
    ax2.set_title("(b) Class imbalance (development)")
    fig.tight_layout()
    save(fig, "fig_class_distribution.png")


# ===========================================================================
# 5. Confusion matrix (deployed full-development model, test n=833)
# ===========================================================================
# Recovered exactly from the notebook's reported recalls, precisions and counts:
#   row = true class, column = predicted class
CM = np.array([[510, 158, 22],
               [31, 53, 21],
               [2, 8, 28]])


def fig_confusion_matrix():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    labels = ["Low", "Moderate", "High"]

    for ax, mat, title, fmt in [
        (ax1, CM, "(a) Counts", "d"),
        (ax2, CM / CM.sum(axis=1, keepdims=True) * 100, "(b) Row-normalised (recall, %)", ".1f"),
    ]:
        im = ax.imshow(CM / CM.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted category"); ax.set_ylabel("True category")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                shade = CM[i, j] / CM[i].sum()
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=9,
                        color="white" if shade > 0.5 else "#222222",
                        fontweight="bold" if i == j else "normal")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#cccccc")

    ax1.add_patch(plt.Rectangle((-0.5, 1.5), 1, 1, fill=False, edgecolor=C_HIGH, lw=2))
    fig.suptitle("Deployed Model C — held-out test set (n = 833)", fontsize=9.5, y=1.02)
    fig.tight_layout()
    fig.text(0.5, -0.035,
             "Red outline: the 2 High→Low misclassifications — the most safety-critical error type "
             "(5.3% of true High cases).",
             ha="center", fontsize=7, color=C_HIGH)
    save(fig, "fig_confusion_matrix.png")


# ===========================================================================
# 6. Per-class precision / recall / F1
# ===========================================================================
def fig_per_class_metrics():
    prec = [0.939227, 0.242009, 0.394366]
    rec = [0.739130, 0.504762, 0.736842]
    f1 = [0.827251, 0.327160, 0.513761]
    x = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - w, prec, w, label="Precision", color=C_BLUE)
    ax.bar(x, rec, w, label="Recall", color=C_LOW)
    ax.bar(x + w, f1, w, label="F1", color=C_GREY)
    for xi, vals in zip(x, zip(prec, rec, f1)):
        for dx, v in zip((-w, 0, w), vals):
            ax.text(xi + dx, v + 0.018, f"{v:.2f}", ha="center", fontsize=6.8)
    ax.set_xticks(x); ax.set_xticklabels(["Low", "Moderate", "High"])
    ax.set_ylim(0, 1.06); ax.set_ylabel("Score")
    ax.set_title("Per-class performance, deployed model (test n = 833)")
    ax.legend(fontsize=7.5, frameon=False, ncol=3)
    fig.tight_layout()
    save(fig, "fig_per_class_metrics.png")


# ===========================================================================
# 7. RAG factor-aware retrieval
# ===========================================================================
def fig_rag_pipeline():
    fig, ax = blank_axes((7.0, 3.9))
    box(ax, 0.02, 0.70, 0.20, 0.16,
        "Top ML factors\nfor this patient\n(e.g. SBP 160,\nHbA1c 8.5%)",
        fc="#eef6ef", ec=C_LOW, fs=7.5)
    box(ax, 0.26, 0.70, 0.20, 0.16,
        "Query string\nrisk class +\nfactor phrases +\nintent anchor", fc="white", fs=7.5)
    box(ax, 0.50, 0.70, 0.18, 0.16, "BGE-M3\nembedding\n(1024-d,\nvia Ollama)",
        fc="#eef3f8", ec=C_BLUE, fs=7.5)
    box(ax, 0.72, 0.70, 0.26, 0.16,
        "FAISS search over\n162 guideline chunks\n→ candidate pool\n(up to 24)",
        fc="#eef3f8", ec=C_BLUE, fs=7.5)
    arrow(ax, (0.22, 0.78), (0.26, 0.78)); arrow(ax, (0.46, 0.78), (0.50, 0.78))
    arrow(ax, (0.68, 0.78), (0.72, 0.78))
    arrow(ax, (0.85, 0.70), (0.85, 0.60))

    box(ax, 0.10, 0.315, 0.80, 0.265, "", fc="#fbfbfb", ec="#999999")
    ax.text(0.50, 0.545, "Factor-aware selection (per-patient personalisation)",
            ha="center", fontsize=8.5, fontweight="bold")
    ax.text(0.50, 0.495,
            "priority topics derived from the patient's own risk-raising factors, in importance order",
            ha="center", fontsize=7, color="#444444", style="italic")
    topics = [("bp", C_HIGH), ("diabetes", C_MOD), ("nutrition", C_BLUE), ("followup", C_LOW)]
    tx = 0.135
    for name, col in topics:
        box(ax, tx, 0.375, 0.165, 0.075, f"best chunk\ntopic = {name}", fc="white", ec=col, fs=7.2)
        tx += 0.185
    ax.text(0.50, 0.338, "remaining slots filled by best-scoring leftovers · at-risk cases always get a followup chunk",
            ha="center", fontsize=6.8, color="#444444")

    box(ax, 0.28, 0.12, 0.44, 0.115,
        "k = 4 chunks supplied to the prompt\n"
        "restricted (dietitian-only) chunks tagged inline", fc="#fdf7e8", ec=C_MOD, fs=7.8)
    arrow(ax, (0.50, 0.315), (0.50, 0.238))
    save(fig, "fig_rag_pipeline.png")


# ===========================================================================
# 8. LoRA / QLoRA mechanism
# ===========================================================================
def fig_lora():
    fig, ax = blank_axes((7.0, 3.5))
    ax.text(0.25, 0.94, "(a) LoRA fine-tuning (Phase 8, training time)",
            ha="center", fontsize=8.8, fontweight="bold")
    ax.text(0.78, 0.94, "(b) Serving (inference time)", ha="center",
            fontsize=8.8, fontweight="bold")
    ax.plot([0.545, 0.545], [0.06, 0.90], color="#cccccc", lw=0.9, ls="--")

    box(ax, 0.035, 0.505, 0.195, 0.335,
        "Base model\nQwen3-4B\n\nFROZEN\n\nbf16\n(unquantised)",
        fc="#eeeeee", ec="#888888", fs=7.3)
    box(ax, 0.295, 0.505, 0.195, 0.335,
        "LoRA matrices\n$A \\in \\mathbb{R}^{r\\times d}$,  $B \\in \\mathbb{R}^{d\\times r}$\n"
        "$r=16$,  $\\alpha=32$\n\nTRAINED\n(bf16)", fc="#eaf3ee", ec=C_LOW, fs=7.3)
    arrow(ax, (0.232, 0.735), (0.293, 0.735))
    ax.text(0.2625, 0.757, "activations", fontsize=5.9, ha="center", color=C_GREY)
    arrow(ax, (0.293, 0.605), (0.232, 0.605), color=C_LOW)
    ax.text(0.2625, 0.627, "gradients", fontsize=5.9, ha="center", color=C_LOW)
    ax.text(0.2625, 0.425,
            "kept in bf16: the 24 GB training GPU had\n"
            "headroom, so 4-bit quantisation (QLoRA)\nwas not required",
            ha="center", va="center", fontsize=6.4, color="#444444", linespacing=1.45)
    box(ax, 0.055, 0.115, 0.415, 0.135,
        "saved artefact: adapter_model.safetensors\n≈132 MB · peft_type = LORA\n"
        "precision-agnostic, base-model-independent", fc="#fdf7e8", ec=C_MOD, fs=6.9)
    arrow(ax, (0.393, 0.505), (0.31, 0.253))

    box(ax, 0.595, 0.505, 0.165, 0.335,
        "Base model\nQwen3-4B\n\nFROZEN\n\nfull precision\n(torch_dtype\n= auto)",
        fc="#eeeeee", ec="#888888", fs=7.3)
    box(ax, 0.800, 0.505, 0.165, 0.335,
        "Same LoRA\nadapter\n\nattached via\nPeftModel\n(unchanged)",
        fc="#eaf3ee", ec=C_LOW, fs=7.3)
    arrow(ax, (0.762, 0.672), (0.798, 0.672))
    ax.text(0.7825, 0.425,
            "the adapter file is reused as-is;\nit is not tied to the precision\nit was trained under",
            ha="center", va="center", fontsize=6.4, color="#444444", linespacing=1.45)
    box(ax, 0.590, 0.115, 0.378, 0.135,
        "effective weights  $W' = W + \\frac{\\alpha}{r} BA$\n\n"
        "no quantisation required at serving", fc="#eef3f8", ec=C_BLUE, fs=6.9)
    arrow(ax, (0.858, 0.505), (0.80, 0.253))
    save(fig, "fig_lora.png")


# ===========================================================================
# 9. Safety validator layers
# ===========================================================================
def fig_validator():
    fig, ax = blank_axes((6.4, 4.6))
    box(ax, 0.28, 0.885, 0.44, 0.070, "Raw model output", fc="#eef3f8", ec=C_BLUE, fs=8.5)
    layers = [
        ("Structural / schema check", "5 required sections present?", "regenerate once", "#f2f2f2"),
        ("Diagnosis blocker", "\"you have CKD / kidney failure\"", "SEVERE → fallback", "#fdecea"),
        ("Medication blocker", "drug names, dosing, start/stop/change", "SEVERE → fallback", "#fdecea"),
        ("Supplement blocker", "\"take vitamin D / fish oil\"", "SEVERE → fallback", "#fdecea"),
        ("Nutrition-restriction rule", "numeric protein/K/P/Na target", "MODERATE → rewrite", "#fdf7e8"),
        ("Lab-confirmation reminder", "moderate/high without follow-up", "MODERATE → append", "#fdf7e8"),
        ("Emergency redirect", "SBP≥180 / DBP≥120 / red-flag text", "MODERATE → prepend", "#fdf7e8"),
    ]
    y = 0.795
    for name, trig, act, fc in layers:
        box(ax, 0.06, y, 0.88, 0.078, "", fc=fc, ec="#777777")
        ax.text(0.09, y + 0.039, name, va="center", fontsize=7.8, fontweight="bold")
        ax.text(0.44, y + 0.039, trig, va="center", fontsize=6.9, color="#333333")
        ax.text(0.915, y + 0.039, act, va="center", ha="right", fontsize=6.9, style="italic")
        y -= 0.092
    arrow(ax, (0.50, 0.885), (0.50, 0.874))

    y_out = 0.055
    for x0, label, fc, ec in [
        (0.06, "pass\nreturned as written", "#eaf3ee", C_LOW),
        (0.37, "sanitized\nmoderate fixes applied", "#fdf7e8", C_MOD),
        (0.68, "fallback\nsafe template returned", "#fdecea", C_HIGH),
    ]:
        box(ax, x0, y_out, 0.26, 0.072, label, fc=fc, ec=ec, fs=7.5)
    arrow(ax, (0.50, 0.148), (0.50, 0.130))
    ax.text(0.50, 0.955, "Every output passes every layer, in every deployment mode",
            ha="center", fontsize=7.2, style="italic", color="#444444")
    save(fig, "fig_validator.png")


# ===========================================================================
# 10. Six-arm ablation design
# ===========================================================================
def fig_q1_design():
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    arms = ["A", "B", "C", "D", "E", "F"]
    comps = ["Fine-tuned\nmodel", "RAG\nretrieval", "Safety\nvalidator", "Compact\nprompt"]
    M = np.array([
        [0, 0, 0, 1],   # A base compact, no rag, no validator
        [0, 1, 0, 0],   # B base full rag
        [0, 1, 1, 0],   # C base full rag val
        [1, 1, 0, 0],   # D ft full rag
        [1, 1, 1, 0],   # E ft full rag val
        [1, 1, 1, 1],   # F ft compact rag val
    ])
    ax.imshow(M.T, cmap=matplotlib.colors.ListedColormap(["#f4f4f4", "#4a7fb5"]),
              vmin=0, vmax=1, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(comps)):
            ax.text(i, j, "on" if M[i, j] else "off", ha="center", va="center",
                    fontsize=7.5, color="white" if M[i, j] else "#777777",
                    fontweight="bold" if M[i, j] else "normal")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([f"{a}" for a in arms], fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(comps))); ax.set_yticklabels(comps, fontsize=7.5)
    ax.set_xlabel("Evaluation arm  (F = deployed configuration)")
    ax.set_title("Six-arm ablation: component isolation, identical 300-profile benchmark")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    pairs = [("B vs A", 0.5), ("C vs B", 1.5), ("D vs B", 2.5), ("E vs D", 3.5), ("F vs E", 4.5)]
    for label, x in pairs:
        ax.text(x, 3.85, label, ha="center", fontsize=6.4, color=C_GREY)
    fig.tight_layout()
    save(fig, "fig_q1_design.png")


# ===========================================================================
# 11-13. Q1 results read from the generated tables
# ===========================================================================
def load_arm_metrics():
    path = ROOT / "q1_report" / "tables" / "q1_per_arm_metrics.csv"
    rows = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["arm"]] = r
    return rows


def fig_q1_safety(rows):
    arms = list("ABCDEF")
    raw = [float(rows[a]["raw_critical_unsafe_pct"]) for a in arms]
    del_ = [float(rows[a]["delivered_critical_unsafe_pct"]) for a in arms]
    x = np.arange(len(arms)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    ax.bar(x - w / 2, raw, w, label="Raw model output", color="#b9c6d4", edgecolor="#5a6b7c")
    ax.bar(x + w / 2, del_, w, label="Delivered to patient", color=C_BLUE)
    for xi, (r, d) in enumerate(zip(raw, del_)):
        ax.text(xi - w / 2, r + 0.22, f"{r:.2f}", ha="center", fontsize=6.8)
        ax.text(xi + w / 2, d + 0.22, f"{d:.2f}", ha="center", fontsize=6.8,
                fontweight="bold" if d == 0 else "normal",
                color=C_LOW if d == 0 else "#222222")
    for xi in (2, 5):
        ax.annotate("0.00", xy=(xi + w / 2, 0), xytext=(xi + w / 2, 2.4),
                    ha="center", fontsize=7, color=C_LOW, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C_LOW, lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{rows[a]['arm_name']}" for a in arms], fontsize=7)
    ax.set_ylabel("Critical-unsafe rate (%)")
    ax.set_title("Critical-unsafe output before and after the safety validator (n = 300)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_ylim(0, 12.5)
    fig.tight_layout()
    save(fig, "fig_q1_safety.png")


def fig_q1_emergency(rows):
    arms = list("ABCDEF")
    rec = [float(rows[a]["emergency_redirect_recall_pct"]) for a in arms]
    cis = []
    for a in arms:
        lo, hi = rows[a]["emergency_redirect_recall_ci"].strip("[]").split(",")
        cis.append((float(lo), float(hi)))
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    colors = [ARM_COLORS[a] for a in arms]
    x = np.arange(len(arms))
    lo_err = [max(0.0, r - c[0]) for r, c in zip(rec, cis)]
    hi_err = [max(0.0, c[1] - r) for r, c in zip(rec, cis)]
    ax.bar(x, rec, 0.58, color=colors, yerr=[lo_err, hi_err], capsize=3,
           error_kw=dict(lw=0.9, ecolor="#555555"))
    for xi, r in zip(x, rec):
        ax.text(xi, r + 3.0, f"{r:.0f}%", ha="center", fontsize=7.5,
                fontweight="bold" if r == 100 else "normal")
    ax.axhline(100, ls="--", lw=0.8, color=C_LOW)
    ax.text(5.45, 101.5, "ceiling", fontsize=6.5, color=C_LOW, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{rows[a]['arm_name']}" for a in arms], fontsize=7)
    ax.set_ylabel("Emergency-redirect recall (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Emergency-redirect recall, emergency subgroup (n = 60), 95% Wilson CI")
    fig.tight_layout()
    save(fig, "fig_q1_emergency.png")


def fig_q1_quality(rows):
    arms = list("ABCDEF")
    metrics = [("format_ok_pct", "Format\nadherence"),
               ("grounding_any_pct", "Cited ≥1\nsource"),
               ("readability_ok_pct", "Readability\n≤ grade 8"),
               ("followup_appropriate_pct", "Appropriate\nfollow-up")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6), sharey=True)
    x = np.arange(len(arms))
    for ax, (key, title) in zip(axes, metrics):
        vals = [float(rows[a][key]) for a in arms]
        ax.bar(x, vals, 0.62, color=[ARM_COLORS[a] for a in arms])
        ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=7.5)
        ax.set_title(title, fontsize=7.8)
        ax.set_ylim(0, 112)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 2.5, f"{v:.0f}", ha="center", fontsize=6.2)
    axes[0].set_ylabel("Percent of profiles (%)")
    fig.suptitle("Response quality and grounding by arm (n = 300)", fontsize=9, y=1.04)
    fig.tight_layout()
    save(fig, "fig_q1_quality.png")


def fig_q1_efficiency(rows):
    arms = list("ABCDEF")
    prompt = [float(rows[a]["prompt_tokens_mean"]) for a in arms]
    completion = [float(rows[a]["completion_tokens_mean"]) for a in arms]
    lat = [float(rows[a]["latency_s_mean"]) for a in arms]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(x, prompt, 0.55, label="Prompt tokens", color="#9fb3c8")
    ax.bar(x, completion, 0.55, bottom=prompt, label="Completion tokens", color=C_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{rows[a]['arm_name']}" for a in arms], fontsize=7)
    ax.set_ylabel("Mean tokens per request")
    ax2 = ax.twinx()
    ax2.plot(x, lat, "o--", color=C_HIGH, lw=1.2, ms=4, label="Latency (s)")
    ax2.set_ylabel("Mean latency (s)", color=C_HIGH)
    ax2.tick_params(axis="y", colors=C_HIGH)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, max(lat) * 1.45)
    ax.annotate("", xy=(5, prompt[5]), xytext=(4, prompt[4]),
                arrowprops=dict(arrowstyle="<->", color=C_LOW, lw=1.2))
    ax.text(4.5, (prompt[4] + prompt[5]) / 2 + 130,
            f"−{(1 - prompt[5]/prompt[4])*100:.0f}% prompt\ntokens (F vs E)",
            ha="center", fontsize=6.8, color=C_LOW, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Token cost and latency by arm")
    fig.tight_layout()
    save(fig, "fig_q1_efficiency.png")


# ===========================================================================
# 14. Forest plot of pre-specified comparisons
# ===========================================================================
def fig_q1_forest():
    path = ROOT / "q1_report" / "tables" / "q1_paired_comparisons.csv"
    keep = {"delivered_critical_unsafe": "Delivered critical-unsafe",
            "emergency_redirect_recall": "Emergency-redirect recall"}
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["metric"] in keep:
                rows.append(r)
    order = ["B vs A", "C vs B", "D vs B", "E vs D", "F vs E"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, (metric, title) in zip(axes, keep.items()):
        sel = [r for r in rows if r["metric"] == metric]
        sel = sorted(sel, key=lambda r: order.index(r["comparison"]))
        y = np.arange(len(sel))
        for i, r in enumerate(sel):
            rd = float(r["risk_difference_pct"])
            lo = float(r["paired_ci_lo_pct"]); hi = float(r["paired_ci_hi_pct"])
            sig = float(r["p_holm"]) < 0.05
            col = C_LOW if sig else C_GREY
            ax.plot([lo, hi], [i, i], color=col, lw=1.6, solid_capstyle="round")
            ax.plot(rd, i, "o", color=col, ms=5.5)
            ax.text(hi + 1.4, i, ("Holm p<0.05" if sig else "n.s."),
                    va="center", fontsize=6.3, color=col)
        ax.axvline(0, color="#999999", lw=0.9, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels([r["comparison"] for r in sel], fontsize=7.6)
        ax.set_xlabel("Risk difference (percentage points)")
        ax.set_title(title, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(min(float(r["paired_ci_lo_pct"]) for r in sel) - 6,
                    max(float(r["paired_ci_hi_pct"]) for r in sel) + 18)
    fig.suptitle("Pre-specified paired comparisons (McNemar, Holm-corrected within family)",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "fig_q1_forest.png")


# ===========================================================================
# 15. Fine-tuning loss curve
# ===========================================================================
def fig_finetune_loss():
    path = ROOT / "llm" / "finetune" / "adapters" / "v1" / "training_log.csv"
    steps, tr, ev_s, ev = [], [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("train_loss"):
                steps.append(float(r["step"])); tr.append(float(r["train_loss"]))
            if r.get("eval_loss"):
                ev_s.append(float(r["step"])); ev.append(float(r["eval_loss"]))
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot(steps, tr, "-", color=C_BLUE, lw=1.4, label="Training loss")
    ax.plot(ev_s, ev, "o--", color=C_HIGH, lw=1.3, ms=5, label="Validation loss")
    for s, v in zip(ev_s, ev):
        ax.annotate(f"{v:.4f}", (s, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=6.8, color=C_HIGH)
    ax.set_xlabel("Optimiser step"); ax.set_ylabel("Cross-entropy loss (answer tokens)")
    ax.set_title("LoRA fine-tuning: 3 epochs, 540 training examples")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_ylim(0, max(tr) * 1.12)
    fig.tight_layout()
    save(fig, "fig_finetune_loss.png")


# ===========================================================================
# 16. Prompt anatomy
# ===========================================================================
def fig_prompt_structure():
    fig, ax = blank_axes((6.6, 4.6))
    ax.text(0.25, 0.955, "SYSTEM TURN", ha="center", fontsize=8.6, fontweight="bold")
    ax.text(0.75, 0.955, "USER TURN", ha="center", fontsize=8.6, fontweight="bold")

    box(ax, 0.03, 0.66, 0.42, 0.26,
        "FULL prompt (base model)\n\n• scope statement\n• 9 numbered safety rules\n"
        "• output-format specification\n  with per-section hints\n• reading-level target",
        fc="#f4f4f4", ec="#888888", fs=7.2)
    box(ax, 0.03, 0.40, 0.42, 0.22,
        "COMPACT prompt (fine-tuned)\n\nsame constraints, condensed;\n"
        "behaviour internalised during\nfine-tuning\n→ 25% fewer prompt tokens",
        fc="#eaf3ee", ec=C_LOW, fs=7.2)
    ax.text(0.24, 0.355, "exactly one is selected, automatically,\nby whether an adapter is attached",
            ha="center", fontsize=6.6, style="italic", color="#444444")

    blocks = [
        ("URGENT VITALS ALERT", "emitted only when SBP≥180 or DBP≥120;\nnames the patient's own reading", "#fdecea", C_HIGH),
        ("SCREENING RESULT", "risk level, class probabilities,\nconfidence, reliability, completeness", "#eef6ef", C_LOW),
        ("PATIENT PROFILE", "every reported value, human-labelled\nwith units", "#eef3f8", C_BLUE),
        ("TOP FACTORS", "ML explanation factors, each marked\nas raising or lowering risk", "#eef3f8", C_BLUE),
        ("GUIDELINE EXCERPTS", "k=4 retrieved chunks with ids;\nrestricted chunks tagged inline", "#fdf7e8", C_MOD),
        ("PERSONALISATION\nREQUIREMENTS", "per-section instructions derived\nfrom this patient's own values", "#eaf3ee", C_LOW),
    ]
    y = 0.845
    for name, detail, fc, ec in blocks:
        box(ax, 0.52, y, 0.45, 0.115, "", fc=fc, ec=ec)
        ax.text(0.545, y + 0.079, name, fontsize=7.1, fontweight="bold", va="center")
        ax.text(0.545, y + 0.036, detail, fontsize=6.2, va="center", color="#333333",
                linespacing=1.3)
        y -= 0.128
    box(ax, 0.03, 0.10, 0.94, 0.10,
        "Model reply: Summary · Food habits · Rest and sleep · Daily routine · Follow-up reminder\n"
        "+ machine-readable grounding line listing the excerpt ids actually used",
        fc="white", ec="#333333", fs=7.4)
    arrow(ax, (0.24, 0.40), (0.24, 0.205))
    arrow(ax, (0.745, y + 0.115), (0.745, 0.205))
    save(fig, "fig_prompt_structure.png")


# ===========================================================================
# 17. KDIGO label construction grid
# ===========================================================================
def fig_kdigo_grid():
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    grid = np.array([  # rows: G1..G5 ; cols: A1,A2,A3  (0=low,1=mod,2=high,3=very high→high)
        [0, 1, 2],
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 3],
        [3, 3, 3],
    ])
    cmap = matplotlib.colors.ListedColormap([C_LOW, C_MOD, C_HIGH, "#7a1f18"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["A1\n<30 mg/g", "A2\n30–300 mg/g", "A3\n>300 mg/g"], fontsize=7.5)
    ax.set_yticks(range(5))
    ax.set_yticklabels(["G1  ≥90", "G2  60–89", "G3a 45–59", "G3b/G4 15–44", "G5  <15"],
                       fontsize=7.5)
    ax.set_xlabel("Albuminuria category (UACR)")
    ax.set_ylabel("eGFR category (mL/min/1.73 m²)")
    labels = {0: "Low", 1: "Moderate", 2: "High", 3: "Very high"}
    for i in range(5):
        for j in range(3):
            ax.text(j, i, labels[grid[i, j]], ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
    ax.set_title("KDIGO risk grid — used ONLY to construct training labels", fontsize=8.6)
    ax.text(0.5, -0.30,
            "eGFR and UACR define the target variable; they are never model input features.\n"
            "\"Very high\" is merged into the High screening category.",
            transform=ax.transAxes, ha="center", fontsize=6.8, style="italic", color="#444444")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    save(fig, "fig_kdigo_grid.png")


# ===========================================================================
# 18. Data-generation (Phase 7) and evaluation workflow
# ===========================================================================
def fig_data_generation():
    fig, ax = blank_axes((7.0, 2.9))
    steps = [
        ("600 synthetic\nprofiles", "class-balanced\n200 / 200 / 200", "#eef3f8"),
        ("Real cascade\nscoring", "each profile scored by\nthe deployed model", "#eef6ef"),
        ("RAG retrieval", "k golden chunks +\n2 distractors", "#fdf7e8"),
        ("Self-distilled\ntarget", "base Qwen3 under the\nFULL prompt", "#fdf7e8"),
        ("Safety gate", "scan_unsafe + all 5\nsections required", "#fdeeec"),
        ("Kept pair", "COMPACT prompt →\ngold answer", "#eaf3ee"),
    ]
    w, h, gap = 0.142, 0.34, 0.018
    x = 0.025
    for i, (t, d, fc) in enumerate(steps):
        box(ax, x, 0.40, w, h, "", fc=fc, ec="#666666")
        ax.text(x + w / 2, 0.645, t, ha="center", fontsize=7.4, fontweight="bold",
                linespacing=1.3)
        ax.text(x + w / 2, 0.485, d, ha="center", fontsize=6.2, linespacing=1.35,
                color="#333333")
        if i < len(steps) - 1:
            arrow(ax, (x + w, 0.57), (x + w + gap, 0.57), lw=0.9)
        x += w + gap
    ax.text(0.50, 0.855,
            "Phase 7 — training-data generation (prompt-compression objective)",
            ha="center", fontsize=9, fontweight="bold")
    ax.text(0.50, 0.30,
            "pass rate 600/600 (100%)  ·  split 540 train / 60 validation  ·  "
            "no real patient data  ·  held-out evaluation profiles excluded",
            ha="center", fontsize=7, color="#333333")
    ax.text(0.50, 0.19,
            "Targets are produced under the FULL prompt but paired with the COMPACT prompt, so fine-tuning\n"
            "teaches the model to reproduce full-prompt behaviour from a short prompt.",
            ha="center", fontsize=6.8, style="italic", color="#444444", linespacing=1.4)
    save(fig, "fig_data_generation.png")


# ===========================================================================
# 19. Benchmark composition
# ===========================================================================
def fig_benchmark_composition():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    groups = ["Standard", "Emergency", "Adversarial", "Missing\ninputs",
              "Boundary", "Low\nliteracy"]
    counts = [50] * 6
    ax1.bar(groups, counts, color=[C_BLUE, C_HIGH, "#7a1f18", C_MOD, C_GREY, C_LOW],
            width=0.62)
    ax1.set_ylabel("Profiles per subgroup")
    ax1.set_title("(a) Benchmark subgroups (6 × 50 = 300)", fontsize=8.6)
    ax1.tick_params(axis="x", labelsize=6.8)
    ax1.set_ylim(0, 62)
    for i, c in enumerate(counts):
        ax1.text(i, c + 1.5, str(c), ha="center", fontsize=7)

    ax2.bar(["Low", "Moderate", "High"], [100, 100, 100],
            color=[C_LOW, C_MOD, C_HIGH], width=0.5)
    ax2.set_ylabel("Profiles per risk class")
    ax2.set_ylim(0, 125)
    for i in range(3):
        ax2.text(i, 103, "100", ha="center", fontsize=7.5)
    ax2.set_title("(b) Risk-class balance (rejection-sampled\nagainst the deployed cascade)",
                  fontsize=8.6)
    fig.tight_layout()
    save(fig, "fig_benchmark_composition.png")


# ===========================================================================
def main():
    fig_system_architecture()
    fig_pipeline_flow()
    fig_cascade()
    fig_class_distribution()
    fig_confusion_matrix()
    fig_per_class_metrics()
    fig_rag_pipeline()
    fig_lora()
    fig_validator()
    fig_q1_design()
    rows = load_arm_metrics()
    fig_q1_safety(rows)
    fig_q1_emergency(rows)
    fig_q1_quality(rows)
    fig_q1_efficiency(rows)
    fig_q1_forest()
    fig_finetune_loss()
    fig_prompt_structure()
    fig_kdigo_grid()
    fig_data_generation()
    fig_benchmark_composition()
    print("\nAll figures written to", OUT)


if __name__ == "__main__":
    main()

# Thesis Report — Build Instructions

LaTeX source for *A Two-Stage Cascade and Safety-Constrained Language Model for
Kidney-Risk Screening and Patient Education* (IIT, Jahangirnagar University).

## Files

```
thesis/
├── main.tex              # preamble + front matter; \input's every chapter
├── chapters/
│   ├── ch1_introduction.tex
│   ├── ch2_literature.tex
│   ├── ch3_system_model.tex
│   ├── ch4_methodology.tex
│   ├── ch5_implementation.tex
│   ├── ch6_results.tex
│   ├── ch7_discussion.tex
│   ├── ch8_conclusion.tex
│   ├── references.tex    # inline thebibliography (no BibTeX pass needed)
│   └── appendix.tex      # Appendices A–E
├── figures/              # 20 figures, all generated from real project data
├── make_figures.py       # regenerates every figure from project artefacts
└── main.pdf              # compiled output (96 pages)
```

## Compiling on Overleaf (recommended)

1. Zip the whole `thesis/` folder (or upload `main.tex`, `chapters/`, `figures/`).
2. New Project → Upload Project → select the zip.
3. Menu → Compiler → **pdfLaTeX**, Main document → `main.tex`.
4. Recompile twice so the table of contents, list of figures, list of tables and
   all cross-references resolve.

No BibTeX run is required: the bibliography is an inline `thebibliography`
environment in `chapters/references.tex`.

## Compiling locally

Any TeX Live / MiKTeX installation with the standard packages
(`geometry`, `setspace`, `graphicx`, `booktabs`, `tabularx`, `longtable`,
`algorithm`, `algpseudocode`, `titlesec`, `tocloft`, `microtype`, `hyperref`,
`caption`, `ragged2e`, `float`, `amsmath`, `amssymb`, `bm`, `enumitem`,
`fancyhdr`, `xcolor`):

```bash
pdflatex main.tex
pdflatex main.tex      # second pass for ToC/LoF/LoT and references
```

## Regenerating the figures

Every figure is produced from the project's own artefacts — the training
notebook's metrics, `q1_report/tables/*.csv`, and the fine-tuning log — so they
stay in sync with the results if anything is re-run:

```bash
python thesis/make_figures.py
```

Sources used by the script:

| Figure group | Data source |
|---|---|
| Confusion matrix, per-class metrics, class distribution | `misc/notebooks/nhanes-model-c-train-save-download-and-predict.ipynb` |
| Six-arm safety / quality / efficiency / forest plot | `q1_report/tables/q1_per_arm_metrics.csv`, `q1_paired_comparisons.csv` |
| Fine-tuning loss curve | `llm/finetune/adapters/v1/training_log.csv` |
| Architecture, pipeline, cascade, RAG, LoRA, validator, prompt, KDIGO grid | Drawn from the system design |

## Adjusting the page count

The document is currently **96 pages** (15 front matter + 81 body) at 12 pt with
1.5 line spacing. If a 60–70 page limit must be met, the levers in order of
least damage to content are:

1. `\onehalfspacing` → `\singlespacing` in `main.tex` (roughly −25 pages).
2. Drop or shorten Appendices B–E (roughly −8 pages).
3. Reduce figure widths from `\textwidth` to `0.85\textwidth`.

Check the departmental formatting rules before changing line spacing.

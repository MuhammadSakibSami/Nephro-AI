"""Q1 final technical evaluation of the LLM communication layer.

Modules:
  * `benchmark`  - Part 1: build/freeze the 300-profile benchmark.
  * `detectors`  - Part 4: critical + secondary safety detectors.
  * `metrics`    - Part 4: quality, grounding and efficiency metrics.
  * `stats`      - Part 5: Wilson, McNemar, paired bootstrap, Wilcoxon, Holm.
  * `run_arms`   - Parts 2/3: the six-arm ablation runner (resumable).
  * `report`     - Part 6: tables and publication figures.

The benchmark is frozen by hash before any generation runs. Evaluator-only
fields (`evaluator.*`) are never rendered into an LLM prompt.
"""

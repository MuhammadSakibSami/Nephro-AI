"""Q1 statistical analysis (Part 5).

Profiles - not generation seeds - are the statistical units throughout.

Provides:
  * `wilson_ci`            - Wilson score 95% CI for a binomial proportion
  * `paired_binary`        - discordant counts, risk difference, paired CI,
                             exact two-sided McNemar p-value
  * `paired_continuous`    - paired difference, profile-level bootstrap CI,
                             Wilcoxon signed-rank p, effect size
  * `holm`                 - Holm-Bonferroni step-down correction (per family)
  * `clustered_bootstrap`  - profile-clustered CI for multi-seed aggregates
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from scipy.stats import binomtest, wilcoxon  # type: ignore
    _SCIPY = True
except Exception:  # pragma: no cover
    _SCIPY = False


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------
def wilson_ci(successes: int, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    """Wilson score interval. Returns (lower, upper) in [0, 1]."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom))


def proportion_row(name: str, successes: int, n: int) -> Dict[str, Any]:
    lo, hi = wilson_ci(successes, n)
    return {
        "metric": name, "events": int(successes), "denominator": int(n),
        "percent": round(100.0 * successes / n, 3) if n else 0.0,
        "wilson_lo_pct": round(100.0 * lo, 3), "wilson_hi_pct": round(100.0 * hi, 3),
    }


# ---------------------------------------------------------------------------
# Paired binary (McNemar)
# ---------------------------------------------------------------------------
def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p at p=0.5 (fallback when scipy is absent)."""
    if n == 0:
        return 1.0
    if _SCIPY:
        return float(binomtest(k, n=n, p=0.5, alternative="two-sided").pvalue)
    # exact sum of probabilities <= observed
    from math import comb
    probs = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= obs + 1e-15)))


def paired_binary(name: str, a: Sequence[int], b: Sequence[int],
                  label_a: str = "A", label_b: str = "B",
                  n_boot: int = 10000, seed: int = 42) -> Dict[str, Any]:
    """Compare two paired binary outcome vectors (1 == event, e.g. unsafe).

    Risk difference is b - a (candidate minus reference).
    """
    if len(a) != len(b):
        raise ValueError("paired vectors must be the same length")
    n = len(a)
    a_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)   # event in A only
    b_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)   # event in B only
    discordant = a_only + b_only
    p_a = sum(a) / n if n else 0.0
    p_b = sum(b) / n if n else 0.0
    diff = p_b - p_a

    # paired bootstrap CI on the difference (resample profiles)
    rng = random.Random(seed)
    idx = range(n)
    diffs = []
    for _ in range(n_boot):
        s = [rng.randrange(n) for _ in idx]
        diffs.append(sum(b[i] for i in s) / n - sum(a[i] for i in s) / n)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)] if n_boot else 0.0
    hi = diffs[min(n_boot - 1, int(0.975 * n_boot))] if n_boot else 0.0

    return {
        "metric": name, "reference": label_a, "candidate": label_b, "n": n,
        f"{label_a}_events": int(sum(a)), f"{label_b}_events": int(sum(b)),
        f"{label_a}_pct": round(100 * p_a, 3), f"{label_b}_pct": round(100 * p_b, 3),
        "discordant_ref_only": int(a_only), "discordant_cand_only": int(b_only),
        "discordant_total": int(discordant),
        "risk_difference_pct": round(100 * diff, 3),
        "paired_ci_lo_pct": round(100 * lo, 3), "paired_ci_hi_pct": round(100 * hi, 3),
        "mcnemar_p": _binom_two_sided(min(a_only, b_only), discordant),
    }


# ---------------------------------------------------------------------------
# Paired continuous
# ---------------------------------------------------------------------------
def paired_continuous(name: str, a: Sequence[float], b: Sequence[float],
                      label_a: str = "A", label_b: str = "B",
                      n_boot: int = 10000, seed: int = 42,
                      use_median: bool = False) -> Dict[str, Any]:
    """Paired difference (b - a) with profile-level bootstrap CI + Wilcoxon."""
    if len(a) != len(b):
        raise ValueError("paired vectors must be the same length")
    n = len(a)
    d = [float(y) - float(x) for x, y in zip(a, b)]

    def _centre(v: List[float]) -> float:
        if not v:
            return 0.0
        if use_median:
            s = sorted(v)
            m = len(s) // 2
            return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
        return sum(v) / len(v)

    point = _centre(d)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        boots.append(_centre(s))
    boots.sort()
    lo = boots[int(0.025 * n_boot)] if n_boot else 0.0
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))] if n_boot else 0.0

    # Wilcoxon signed-rank
    if _SCIPY and any(abs(x) > 1e-12 for x in d):
        try:
            p = float(wilcoxon(list(a), list(b), zero_method="wilcox").pvalue)
        except Exception:
            p = float("nan")
    else:
        p = float("nan")

    # effect size: rank-biserial (paired) = (n_pos - n_neg) / n_nonzero
    n_pos = sum(1 for x in d if x > 0)
    n_neg = sum(1 for x in d if x < 0)
    nz = n_pos + n_neg
    rank_biserial = (n_pos - n_neg) / nz if nz else 0.0
    # standardized mean difference of the paired differences (Cohen's dz)
    if n > 1:
        mean_d = sum(d) / n
        sd = math.sqrt(sum((x - mean_d) ** 2 for x in d) / (n - 1))
        dz = mean_d / sd if sd > 1e-12 else 0.0
    else:
        dz = 0.0

    return {
        "metric": name, "reference": label_a, "candidate": label_b, "n": n,
        f"{label_a}_mean": round(sum(a) / n, 4) if n else 0.0,
        f"{label_b}_mean": round(sum(b) / n, 4) if n else 0.0,
        "paired_difference": round(point, 4),
        "boot_ci_lo": round(lo, 4), "boot_ci_hi": round(hi, 4),
        "wilcoxon_p": p, "rank_biserial": round(rank_biserial, 4),
        "cohens_dz": round(dz, 4),
        "centre": "median" if use_median else "mean",
    }


# ---------------------------------------------------------------------------
# Multiple comparison control
# ---------------------------------------------------------------------------
def holm(rows: List[Dict[str, Any]], p_key: str = "mcnemar_p",
         out_key: str = "p_holm", alpha: float = 0.05) -> List[Dict[str, Any]]:
    """Holm-Bonferroni step-down within one family. Mutates and returns rows."""
    valid = [r for r in rows if isinstance(r.get(p_key), (int, float))
             and not math.isnan(r[p_key])]
    m = len(valid)
    order = sorted(valid, key=lambda r: r[p_key])
    running = 0.0
    for i, r in enumerate(order):
        adj = min(1.0, (m - i) * r[p_key])
        running = max(running, adj)          # enforce monotonicity
        r[out_key] = round(running, 6)
        r[f"significant_holm_{alpha}"] = bool(running < alpha)
    for r in rows:
        r.setdefault(out_key, float("nan"))
        r.setdefault(f"significant_holm_{alpha}", False)
    return rows


# ---------------------------------------------------------------------------
# Multi-seed: profile-clustered bootstrap
# ---------------------------------------------------------------------------
def clustered_bootstrap(values_by_profile: Dict[str, List[float]],
                        n_boot: int = 10000, seed: int = 42) -> Dict[str, Any]:
    """CI for a mean where each profile contributes several seeds.

    Profiles are resampled (clusters), never individual seeds - so three seeds
    are not treated as three independent observations.
    """
    keys = list(values_by_profile.keys())
    n = len(keys)
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_profiles": 0, "n_obs": 0}

    per_profile_mean = {k: (sum(v) / len(v) if v else 0.0)
                        for k, v in values_by_profile.items()}
    point = sum(per_profile_mean.values()) / n

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [per_profile_mean[keys[rng.randrange(n)]] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return {
        "mean": round(point, 4),
        "ci_lo": round(boots[int(0.025 * n_boot)], 4),
        "ci_hi": round(boots[min(n_boot - 1, int(0.975 * n_boot))], 4),
        "n_profiles": n,
        "n_obs": sum(len(v) for v in values_by_profile.values()),
    }

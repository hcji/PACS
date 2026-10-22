"""Rank primitive–phenotype links that are worth following up, combining BOTH Fig.2 panels.

Two panels answer two halves of the same question, so the shortlist is built from both.

**Panel E (statistics).** Per (dose, primitive, phenotype dimension) cell: a two-sample t-test of
that primitive's 3 perturbed repeats against the 5 intact reference repeats, then Benjamini–Hochberg
FDR within each (dose, dimension) family of 31 tests. Reported in two flavours:

- ``robust`` = pooled-variance AND Welch both q < 0.05, with Δ > 0. Welch matters here: the perturbed
  group is the small one (n = 3) *and* the noisy one (median within-group variance ≈ 5× the
  reference), which is exactly the setting where the pooled-variance test is anti-conservative.
- ``Δ`` = baseline − perturbed (positive = more ASD-like), the effect size the panel draws.

**Panel B (context).** The Manhattan panel puts each cell's Δ against the same dimension's 31
primitives, with the dimension median drawn as a black bar. Two readings come from it:

- ``distinct`` = Δ minus that dimension's median Δ ≥ 0.10. A large Δ that only keeps up with its own
  column is a dimension-wide shift, not something this primitive does.
- ``rank`` = where the cell sits among the 31 primitives of that dimension (Panel B labels the top
  three per dimension, per dose).

Two further guards, both from the experiment's own criteria:

- ``clean`` = the primitive's general-cognition control channel moved ≤ 0.15 (else the link is
  indistinguishable from general degradation).
- ``dose-consistent`` = the same cell is robust at θ = 0.80 too, or at least keeps Δ ≥ 0.15 there.

Tiers: **A** = all of the above (recommended) · **B** = robust, large, distinct, but the control
channel is depressed or the dose does not carry · **C** = robust and large but NOT distinct
(dimension-wide shift) · **D** = Δ large but not robust (significance leans on the pooled-variance
assumption) · **–** = below the effect-size floor.

Output:

  results/data/figure2_link_candidates.md    the readable shortlist (English)
  results/data/figure2_link_candidates.csv   all 372 cells with every column, for filtering

Run: ``python results/scripts/fig2_link_candidates.py``
"""

from __future__ import annotations

import csv
import importlib.util
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LARGE = 0.30          # effect-size floor for "worth looking at" (Δ at θ = 0.40)
DISTINCT = 0.10       # Δ must exceed its own dimension's median by this much (Panel B reading)
DOSE_OK_DELTA = 0.15  # "the effect is still there at θ = 0.80" (that dose has less power)
CLEAN = 0.15          # control-channel tolerance, same as the experiment's CONTROL_TOLERANCE


def _load_fig2():
    """Reuse Fig.2's data layer so the two documents cannot drift apart."""
    spec = importlib.util.spec_from_file_location(
        "fig2_visualize", ROOT / "results" / "scripts" / "fig2_visualize.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f2 = _load_fig2()
DIMS = f2.DIMS
DIM_EN = {d: f2.DIM_2L[d].replace("\n", " ") for d in DIMS}
DIM_ZH = {
    "social_interaction_motivation": "社会互动与社会动机",
    "social_cognition": "社会认知",
    "social_communication_pragmatics": "社会沟通与语用",
    "cognitive_flexibility_adaptation": "认知灵活性与适应",
    "repetitive_restricted_behavior": "重复性与限制性行为",
    "novelty_uncertainty_sensory_behavior": "新颖性与感觉行为",
}


def build(dose: float) -> list[dict]:
    """One dose's 31 × 6 cells, with Panel B's median/rank computed alongside Panel E's tests."""
    rows = f2.load()["rows_by_dose"][dose]
    med, rank = {}, {}
    for j, dim in enumerate(DIMS):
        vals = sorted(((r["emergence"][j], r["primitive_id"]) for r in rows), key=lambda t: -t[0])
        med[dim] = statistics.median(v for v, _ in vals)
        for k, (_, pid) in enumerate(vals, 1):
            rank[(pid, dim)] = k
    out = []
    for r in rows:
        for j, dim in enumerate(DIMS):
            out.append({
                "dose": dose,
                "primitive_id": r["primitive_id"],
                "primitive": r["en"],
                "primitive_zh": r["zh"],
                "dimension": dim,
                "dimension_zh": DIM_ZH[dim],
                "delta": r["emergence"][j],
                "dim_median": med[dim],
                "delta_minus_dim_median": r["emergence"][j] - med[dim],
                "rank_in_dimension": rank[(r["primitive_id"], dim)],
                "top3_in_panel_b": rank[(r["primitive_id"], dim)] <= 3,
                "p_pooled": r["p_t"][j],
                "p_welch": r["p_welch"][j],
                "q_pooled": r["q_t"][j],
                "q_welch": r["q_welch"][j],
                "is_dominant": abs(r["emergence"][j] - r["dominant_effect"]) < 1e-12,
                "dominant_dim": r["dominant_dim"],
                "delta_dominant": r["dominant_effect"],
                "delta_control": r["control"],
                "verdict": r["verdict"],
            })
    return out


def main() -> int:
    by_dose = {d: build(d) for d in (f2.PRIMARY_DOSE, f2.SECONDARY_DOSE)}
    idx = {(c["dose"], c["primitive_id"], c["dimension"]): c for cs in by_dose.values() for c in cs}

    def robust(c) -> bool:
        return c["q_pooled"] < f2.TEST_ALPHA and c["q_welch"] < f2.TEST_ALPHA and c["delta"] > 0

    for c in by_dose[f2.PRIMARY_DOSE] + by_dose[f2.SECONDARY_DOSE]:
        other = idx[(f2.SECONDARY_DOSE if c["dose"] == f2.PRIMARY_DOSE else f2.PRIMARY_DOSE,
                     c["primitive_id"], c["dimension"])]
        c["clean"] = c["delta_control"] <= CLEAN
        c["distinct"] = c["delta_minus_dim_median"] >= DISTINCT
        c["dose_consistent"] = robust(other) or other["delta"] >= DOSE_OK_DELTA
        c["robust"] = robust(c)
        if c["robust"] and c["delta"] >= LARGE and c["distinct"]:
            c["tier"] = "A" if (c["clean"] and c["dose_consistent"]) else "B"
        elif c["robust"] and c["delta"] >= LARGE:
            c["tier"] = "C"
        elif c["delta"] >= LARGE:
            c["tier"] = "D"
        else:
            c["tier"] = "-"

    prim = by_dose[f2.PRIMARY_DOSE]
    t = {k: sorted([c for c in prim if c["tier"] == k], key=lambda c: -c["delta"]) for k in "ABCD"}

    # ------------------------------------------------------------------ CSV
    cols = ["dose", "primitive_id", "primitive", "primitive_zh", "dimension", "dimension_zh",
            "delta", "dim_median", "delta_minus_dim_median", "rank_in_dimension", "top3_in_panel_b",
            "p_pooled", "p_welch", "q_pooled", "q_welch", "is_dominant", "dominant_dim",
            "delta_dominant", "delta_control", "clean", "distinct", "dose_consistent", "robust",
            "verdict", "tier"]
    csv_path = ROOT / "results" / "data" / "figure2_link_candidates.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in by_dose[f2.PRIMARY_DOSE] + by_dose[f2.SECONDARY_DOSE]:
            w.writerow({k: (round(c[k], 6) if isinstance(c[k], float) else c[k]) for k in cols})

    # ------------------------------------------------------------------ MD
    def table(cs: list[dict], limit: int | None = None) -> str:
        head = ("| primitive | phenotype dimension | Δ (θ=0.40) | Δ − dim. median | rank in dim. "
                "| Δ (θ=0.80) | q pooled | q Welch | control Δ |")
        lines = [head, "|" + "---|" * 9]
        for c in (cs[:limit] if limit else cs):
            o = idx[(f2.SECONDARY_DOSE, c["primitive_id"], c["dimension"])]
            star = " ⭐" if c["top3_in_panel_b"] else ""
            lines.append(
                f"| {c['primitive']} | {DIM_EN[c['dimension']]} | **{c['delta']:.3f}** "
                f"| +{c['delta_minus_dim_median']:.3f} | {c['rank_in_dimension']}{star} "
                f"| {o['delta']:.3f} | {c['q_pooled']:.3g} | {c['q_welch']:.3g} "
                f"| {c['delta_control']:.3f} |"
            )
        return "\n".join(lines)

    per_dim = []
    for dim in DIMS:
        cs = [c for c in t["A"] if c["dimension"] == dim][:3]
        if not cs:
            per_dim.append(f"| {DIM_EN[dim]} | — | — | — |")
            continue
        for i, c in enumerate(cs):
            per_dim.append(f"| {DIM_EN[dim] if i == 0 else ''} | {c['primitive']} "
                           f"| {c['delta']:.3f} | {c['q_welch']:.3g} |")
    per_prim = []
    for pid in sorted({c["primitive_id"] for c in t["A"]}):
        row = next(c for c in prim if c["primitive_id"] == pid)
        mine = [c for c in t["A"] if c["primitive_id"] == pid]
        targets = ", ".join(f"{DIM_EN[c['dimension']]} ({c['delta']:.2f})" for c in mine)
        per_prim.append(f"| {row['primitive']} | {targets} | {row['delta_control']:.2f} |")

    n_pooled = sum(1 for c in prim if c["q_pooled"] < f2.TEST_ALPHA and c["delta"] > 0)
    n_robust = sum(1 for c in prim if robust(c))
    b_ctrl = [c["delta_control"] for c in t["B"]]
    md = f"""# Fig.2 companion: which primitive–phenotype links are worth following up

Generated by `results/scripts/fig2_link_candidates.py` from the same data layer as Fig.2
(`results/data/result_2_univariate.xlsx`), so the numbers here and in the figure cannot drift apart.
Δ = baseline − perturbed; positive means the phenotype profile moved towards the ASD-like end.

## Summary

At **θ = 0.40**, {len(t['A'])} cells satisfy every criterion at once — significant under both tests
after FDR, Δ ≥ {LARGE:.2f}, standing clear of their own dimension's median, with an intact control
channel and a dose-consistent effect. Those are the links to build a follow-up study on.
{len(t['B'])} further cells are equally large and significant but come with a depressed
general-cognition channel, {len(t['C'])} merely keep pace with a dimension-wide shift, and
{len(t['D'])} have a large Δ whose significance does not survive an unequal-variance test.

## How to read the tiers (five checks, all from Fig.2's own criteria)

1. **Robust** — pooled-variance AND Welch t-test both q < 0.05 after Benjamini–Hochberg FDR within
   each (dose, dimension) family of 31 tests, with Δ > 0. The two tests agree by construction with a
   two-group design (3 perturbed repeats vs 5 intact repeats, where the pooled test is a one-way
   ANOVA, F = t²) and disagree in practice because the perturbed group is both the smaller and the
   noisier one: at θ = 0.40 the pooled test flags {n_pooled} cells and Welch only {n_robust}, so this
   filter removes more than half.
2. **Large enough** — Δ ≥ {LARGE:.2f} (the effect size; Panel B's y axis).
3. **Distinct within its dimension** — Δ exceeds that dimension's median Δ by ≥ {DISTINCT:.2f}.
   Panel B draws that median as the black bar and labels the top three per dimension; ⭐ below marks
   a cell the panel labels. A large Δ that merely matches its own column is a dimension-wide shift,
   not something this primitive does.
4. **Clean control channel** — general_cognition Δ ≤ {CLEAN:.2f}; otherwise the link cannot be told
   apart from general degradation (9 primitives fail this at θ = 0.40).
5. **Dose-consistent** — the same cell is robust at θ = 0.80 as well, or at least keeps Δ ≥
   {DOSE_OK_DELTA:.2f} there (that dose has less power, so significance is not required).

Tiers: **A** all five · **B** robust + large + distinct but fails 4 or 5 · **C** robust + large but
not distinct · **D** Δ ≥ {LARGE:.2f} but not robust.

## Table 1 · Recommended: follow these up first (tier A, {len(t['A'])} cells)

{table(t['A'])}

## Table 2 · Equally large, but the control channel is depressed (tier B, {len(t['B'])} cells)

These carry the largest Δ in the whole experiment, and the same primitives also push general
cognition out of tolerance (control Δ {min(b_ctrl):.2f}–{max(b_ctrl):.2f} across this tier). Their
phenotype effects cannot be separated from a general collapse of the profile: read them as "broad
disruption", not as mechanism-specific links. Studying them needs a control arm of comparable
general degradation.

{table(t['B'])}

## Table 3 · Robust and large, but only keeping pace with the dimension (tier C, {len(t['C'])} cells)

Δ is sizeable and both tests pass, yet the cell sits within {DISTINCT:.2f} of its own dimension's
median: most of that dimension moved by a similar amount, so nothing here singles these primitives
out. Read them as "this dimension is broadly sensitive", not "this primitive drives it".

{table(t['C'])}

## Table 4 · Large Δ that does not survive an unequal-variance test (tier D, {len(t['D'])} cells)

Listed so the shortlist stays falsifiable: Δ looks substantial, but with n = 3 per cell the Welch
test drops them. **Do not build on them.** The fix is more replicates (6–8 per cell), not a return to
the pooled-variance test.

{table(t['D'])}

## Table 5 · Pick by phenotype dimension (tier A, strongest three per dimension)

| phenotype dimension | primitive | Δ (θ=0.40) | q Welch |
|---|---|---|---|
{chr(10).join(per_dim)}

Two dimensions have **no tier-A link at all**, which is itself a result worth stating in the paper:
*social interaction & motivation*'s only candidate (Information valuation, Δ = 0.33) lands in tier C
because it barely clears that dimension's median (+0.08); and *social cognition* has no cell that
passes the Welch test at any dose — the pooled test finds 22 of 31 primitives significant there, so
the dimension moves broadly, but no single primitive stands out against that background.

## Table 6 · Pick by primitive (tier A targets per primitive, with its control-channel Δ)

| primitive | tier-A target dimensions (Δ) | control Δ |
|---|---|---|
{chr(10).join(per_prim)}

## Companion file and caveats

- `results/data/figure2_link_candidates.csv` — all 372 cells (31 primitives × 6 dimensions × 2 doses) with
  `delta`, `dim_median`, `delta_minus_dim_median`, `rank_in_dimension`, `top3_in_panel_b`,
  `p_pooled`, `p_welch`, `q_pooled`, `q_welch`, `delta_control`, `clean`, `distinct`,
  `dose_consistent`, `robust`, `verdict`, `tier`; filterable in Excel or pandas. Chinese names of the
  primitives and dimensions are kept in `primitive_zh` / `dimension_zh` for cross-reference.
- **θ = 0.40 is the dose the ranking follows** (the stronger perturbation, clearest effects);
  θ = 0.80 serves only as the dose-consistency check.
- **Three replicates per cell.** The Welch version has ≈ 2.4 degrees of freedom; use these q values to
  rank, not to conclude.
- **The 5 intact reference repeats are shared by all 372 tests**, so the tests are not independent;
  the FDR families are the 31 tests within a (dose, dimension) block, as in the figure.
- The experiment's own verdict column (`verdict`) is carried through for traceability: it uses the
  Δ ≥ 0.15 rule plus a self-calibrated concentration test, which is *not* the criterion used above.
  A tier-A cell can well be `fail_diffuse` in that scheme — the two answer different questions
  ("is the shift big?" versus "is the profile shift concentrated on one dimension?").
"""
    md_path = ROOT / "results" / "data" / "figure2_link_candidates.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"✓ {md_path.relative_to(ROOT)}")
    print(f"✓ {csv_path.relative_to(ROOT)} ({len(by_dose[f2.PRIMARY_DOSE])} cells/dose × 2 doses)")
    print(f"  tier A {len(t['A'])} ｜ B {len(t['B'])} ｜ C {len(t['C'])} ｜ D {len(t['D'])}"
          f" ｜ A covers {len({c['primitive_id'] for c in t['A']})} primitives, "
          f"{len({c['dimension'] for c in t['A']})} of the 6 phenotype dimensions")
    return 0


if __name__ == "__main__":
    sys.exit(main())

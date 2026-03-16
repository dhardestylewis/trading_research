# Program Decision: L1 Family Branch

*Date: 2026-03-15*
*Decided after: exp010 (Family Canary Deployment Validation)*

---

## Verdict

**This branch is statistically interesting, economically negative under current costs, and closed for deployment.**

At current market-entry costs, the L1 family classifier is **not monetizable**.

---

## What exp010 showed

exp009's reported "bps" were not true forward-return bps — they were effectively
hit-rate-derived, not realized `fwd_ret_1h` after round-trip cost deduction.
Once corrected in exp010:

| Asset | Fold-Level | Execution-Level | Deployable? |
|-------|-----------|-----------------|-------------|
| SOL   | negative  | negative         | No          |
| NEAR  | negative  | negative         | No          |
| SUI   | +24.4 bps | -17.4 bps       | No          |
| APT   | +14.1 bps | -18.8 bps       | No          |

There may be weak directional information (SUI, APT), but not enough
economic magnitude to survive costs and execution.

---

## What is settled

### 1. exp009 headline economics are invalid

The `+34.3`, `+37.4`, `+43.1` bps conclusions are measurement artifacts
from a broken metric. All deployment or capacity conclusions derived from
those numbers are superseded. exp009 is retained for research provenance,
annotated as `status: invalidated`.

### 2. The family canary does not advance

All four exp010 gates fail:

- no family live-capital rollout
- no family paper expansion as a production thesis
- no "APT/SUI/NEAR alongside SOL" deployment plan

These remain research shadows only.

### 3. The real bottleneck is signal magnitude relative to execution cost

This is why earlier work on calibration and ranking never paid off. If the
expected move is too small, better calibration does not rescue economics.

---

## Immediate actions

### 1. Freeze all live-capital ambitions on this family branch

- SOL family rollout: **frozen**
- SUI primary lane: **frozen**
- NEAR secondary lane: **frozen**
- APT research lane: **frozen** (shadow research only)

### 2. exp009 annotated as historically informative but economically invalid

- Config: `status: invalidated` added to `crypto_1h_exp009.yaml`
- Report: ⚠️ INVALIDATED banner added to `reports/exp009/summary.md`
- Not deleted — retained for research provenance

### 3. Minimal paper benchmark retained for infrastructure validation

The SOL paper lane via `canary_tick.py` continues as a **systems canary only**.
It is no longer a capital candidate. Purpose: infrastructure validation,
not signal validation.

---

## Next research direction

The program pivots from:

> *"Can this classifier transport across assets?"*

to:

> **"Can we find trades with enough gross move magnitude to clear a 30+ bps round-trip hurdle?"**

### A. Magnitude-first target design

Model expected forward return in bps, tail-move probability, expected return
conditional on executable entry, and expected shortfall / adverse selection.
The model must identify **large enough moves**, not merely slightly-better-than-random direction.

### B. Pre-cost edge distribution

Before any new model family, measure: median gross bps of selected trades,
upper decile gross bps, gross bps by regime/asset/entry convention. The central
question: **where does gross edge exceed ~30–40 bps reliably?** If nowhere, stop.

### C. Passive-only feasibility

Since market-entry cost is the killer, evaluate passive strategies with
materially lower realized cost. Only viable if fill probability is real,
queue priority is realistic, and adverse selection after passive fill is
not catastrophic.

### D. Different horizons (magnitude, not trade count)

The next horizon study must ask: do longer horizons produce **larger gross move
distributions**? Not "do they produce more signals?"

### E. Different asset families

Higher-volatility names, structurally dislocated regimes, less efficient venues,
different microstructure. Do not expand broadly until a magnitude-based screening
framework exists.

---

## Research hygiene note

In the next experiment family, make **"gross bps distribution before costs"**
the first table in the report. This forces the research to confront the
economic hurdle before any Sharpe or transportability narrative develops.

---

## Experiment lineage

```
exp001 → exp002 → exp003 → exp004 → exp005 → exp006 → exp007 → exp008 → exp009 → exp010
                                                                          ↑           ↑
                                                                     INVALIDATED   CORRECTED
                                                                     (metric bug)  (this verdict)
```

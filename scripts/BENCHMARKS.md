# Plausibility-filter benchmarks (synomega 0.4.x)

Effect of the dual-tower reaction-plausibility filter on single-step and
multi-step, measured 2026-07-26. Scripts: `bench_plausibility.py` (latency +
multi-step), `bench_top1.py` / `bench_topk.py` (accuracy), `calibrate_plausibility_threshold.py`.

- Single-step model: `r20_center` TemplateGNN, `top_k=50`, `overfetch=1`.
- Plausibility model: dual-tower `plaus_dual` best.pt, val AUC 0.9946.
- Accuracy set: held-out `processed_r20_center/test.parquet`, n=998, metric =
  true template (test `label`) among the top-k **surviving** candidates.
- Filter is delete-only (no re-ranking); `min_keep=1`.

## 1. Single-step top-k accuracy vs threshold  (r20_center test, n=998)

| config     | top-1  | top-3  | top-5  | top-10 |
|------------|--------|--------|--------|--------|
| no filter  | 0.3798 | 0.5471 | 0.6112 | 0.6764 |
| thr = 0.3  | 0.3758 | 0.5461 | 0.6062 | 0.6713 |
| thr = 0.4  | 0.3758 | 0.5451 | 0.6052 | 0.6713 |
| thr = 0.5  | 0.3778 | 0.5411 | 0.6022 | 0.6683 |

Δ vs no-filter (percentage points):

| config    | top-1 | top-3 | top-5 | top-10 |
|-----------|-------|-------|-------|--------|
| thr = 0.3 | −0.40 | −0.10 | −0.50 | −0.50 |
| thr = 0.4 | −0.40 | −0.20 | −0.60 | −0.50 |
| thr = 0.5 | −0.20 | −0.60 | −0.90 | −0.80 |

**Filtering never improves top-k against the recorded reaction; it is slightly
negative, worsening with k and threshold.** The filter deletes ~2.5% of
candidates — too few to promote a true disconnection past wrong ones ahead of it,
while occasionally deleting a correct disconnection that scored below threshold.
top-1 breakdown (thr 0.5): 6 correct-#1 deleted vs 4 wrong-#1 fixed → net −2/998.
Caveat: the metric credits only the single recorded reaction, so it does not
reward removing chemically-wrong *alternative* disconnections.

## 2. Single-step latency

| device | predict (no filter) | + plausibility scoring | total (filter) | overhead |
|--------|--------------------:|-----------------------:|---------------:|:--------:|
| GPU    |            38.7 ms  |               27.8 ms  |       66.5 ms  |  ×1.72   |
| CPU    |           810   ms  |             2950   ms  |     3760   ms  |  ×4.6    |

The scoring cost is a per-molecule fixed overhead (graph featurization + D-MPNN
forward over all candidates). `overfetch=2→1` barely changed it because the
template model rarely returns more than ~top_k candidates anyway (CPU: ×4.56→×4.64).

## 3. Multi-step planning  (retrostar, depth≤5, max_expansions=300, 25 molecules)

| device | config | median | mean    | avg expansions | solved |
|--------|--------|-------:|--------:|---------------:|:------:|
| CPU    | OFF    | 1056 ms| 47805 ms|          27.0  | 23/25  |
| CPU    | ON     | 3490 ms| 34722 ms|          29.2  | 23/25  |

Typical (median) search ~3.3× slower with the filter, but the **mean is lower
(×0.73)**: filtering prunes implausible branches, so the worst-case searches that
otherwise thrash to the time limit finish sooner. Solve rate unchanged (23/25);
expansions nearly unchanged (×1.08). GPU multi-step: pending.

## Bottom line

- **For top-k retrieval of the recorded reaction, the filter is a net negative**
  (−0.2…−0.9 pp) plus runtime (×1.7 GPU / ×4.6 CPU) — not worth it for that metric.
- Its only measured upside is **capping multi-step worst-case latency** (mean ×0.73)
  without hurting solve rate.
- Value not captured here: removing genuinely implausible *alternative* candidates
  from a list, which a single-recorded-answer metric cannot reward.

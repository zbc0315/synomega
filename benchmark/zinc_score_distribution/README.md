# SAscore / RAscore / SCScore distributions over the ZINC building-block set

Supports **Figure 1(a-c)** and **SI Table S1**: every molecule in the ZINC purchasable building-block
set is commercially available and needs no synthesis, so an ideal accessibility
estimate would be trivial/maximal for all of them. Three structure-only scores
instead spread across their whole ranges -- motivating a stock-aware, route-based
score.

## How the data was produced

1. **Sample** 20,000 SMILES from the ZINC building-block library (`sample_zinc.py`,
   seed **20260727**), source `zinc15_filtered.csv` (~10.6M purchasable ZINC15
   molecules -- the SMILES the 17.4M-key InChIKey stock was built from).
2. **SAscore + SCScore** (`score_baselines.py`): RDKit contributed `sascorer`, and
   the SCScore authors' standalone numpy model.
3. **RAscore** (`score_rascore.py`): Reymond `RAScorerXGB` (ECFP-count XGB classifier).

## Files

| File | Contents |
|---|---|
| `zinc_sa_sc.csv` | per-molecule `smiles, sascore, sa_sec, scscore, sc_sec` (n=20,000) |
| `zinc_ra.csv`    | per-molecule `smiles, rascore, ra_sec` (n=20,000) |
| `sample_zinc.py` | reproducible sampler |

Scoring scripts are in `../synthesizability_baselines/` (`score_baselines.py`,
`score_rascore.py`).

## Summary (n=20,000)

| Score | range | median / mean | "should be" | observed spread |
|---|---|---|---|---|
| SAscore | 1.3-7.2 | median 2.7 | ~1 (easy)     | 32% score > 3 |
| RAscore | 0.001-1.0 | mean 0.91 | 1 (accessible) | 17% score < 0.9 |
| SCScore | 1.0-5.0 | median 3.7 | ~1 (simple)    | 99% score > 2 |

The figure is drawn by `paper_chem_syn/figures/make_fig1_arch.py`.

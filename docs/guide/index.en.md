# Feature Guide · Install & Overview

SynOmega exposes six capabilities. This chapter gives one section per feature with
the **how-to** (command line and Python API); the model / algorithm behind each is
in the matching [research report](../research/index.en.md) chapter — the two
chapters correspond one-to-one:

| Feature | How to use (this chapter) | How it works (research) |
|---|---|---|
| Single-step forward prediction | [↗](forward.md) | [↗](../research/forward.en.md) |
| Multi-component evolution | [↗](evolution.md) | [↗](../research/evolution.en.md) |
| Single-step retrosynthesis | [↗](retro.md) | [↗](../research/retro.en.md) |
| Multi-step route planning | [↗](planning.md) | [↗](../research/planning.en.md) |
| Reaction plausibility | [↗](plausibility.md) | [↗](../research/plausibility.en.md) |
| Synthesizability score (SynScore) | [↗](synscore.md) | [↗](../research/synscore.en.md) |

## Install

```bash
pip install synomega           # core: rdkit + numpy (the template-rule backend works as is)
pip install "synomega[gnn]"    # + the D-MPNN neural single-step backend (torch), recommended
```

The neural backend is an **optional extra**: the template-rule backend runs
without torch; install `[gnn]` when you want the neural template classifier
(forward / retro / evolution / plausibility all build on it). The default model
weights and the ZINC in-stock building-block set are **downloaded on first use**
into `~/.cache/synomega` (override with `SYNOMEGA_CACHE`; pick a mirror with
`SYNOMEGA_MIRROR=ustc|github`), not shipped in the wheel. Pre-fetch with
`synomega download`. Requires Python ≥ 3.10.

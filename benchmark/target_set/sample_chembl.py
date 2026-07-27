#!/usr/bin/env python3
"""Sample N random drug-like targets from ChEMBL 35 (reproducible).

Draws molecules uniformly at random (fixed seed) from ChEMBL 35 small molecules,
takes the largest organic fragment (salt stripping), keeps 5-60 heavy atoms and
RDKit-parseable structures, deduplicates by InChIKey. Writes the 1000-target set,
its first-200 subset, the accepted molregnos, and a metadata JSON so the exact
sample can be re-derived from the same ChEMBL release + seed.

Run (server, base env with rdkit):
    python sample_chembl.py --db /path/to/chembl_35.db --outdir data --n 1000
"""
from __future__ import annotations
import argparse
import json
import random
import sqlite3
from pathlib import Path

SEED = 20260727


def largest_fragment(smi, Chem):
    parts = smi.split(".")
    if len(parts) == 1:
        return smi
    best, best_n = None, -1
    for p in parts:
        m = Chem.MolFromSmiles(p)
        if m is None:
            continue
        n = m.GetNumHeavyAtoms()
        if n > best_n:
            best_n, best = n, p
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--subset", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-heavy-atoms", type=int, default=5)
    ap.add_argument("--max-heavy-atoms", type=int, default=60)
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)
    cur = con.cursor()
    # eligible pool: ChEMBL small molecules with a SMILES; deterministic order
    cur.execute(
        "SELECT cs.molregno FROM compound_structures cs "
        "JOIN molecule_dictionary md USING(molregno) "
        "WHERE md.molecule_type = 'Small molecule' "
        "AND cs.canonical_smiles IS NOT NULL "
        "ORDER BY cs.molregno"
    )
    pool = [r[0] for r in cur.fetchall()]
    print(f"eligible small-molecule pool: {len(pool)}", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    targets = []          # (molregno, canonical_smiles, inchikey, heavy_atoms)
    seen = set()
    n_drawn = n_unparseable = n_size = n_dup = 0
    for mr in pool:
        if len(targets) >= args.n:
            break
        n_drawn += 1
        cur.execute(
            "SELECT canonical_smiles FROM compound_structures WHERE molregno = ?", (mr,)
        )
        smi = cur.fetchone()[0]
        frag = largest_fragment(smi, Chem)
        m = Chem.MolFromSmiles(frag) if frag else None
        if m is None:
            n_unparseable += 1
            continue
        ha = m.GetNumHeavyAtoms()
        if not (args.min_heavy_atoms <= ha <= args.max_heavy_atoms):
            n_size += 1
            continue
        key = Chem.MolToInchiKey(m)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        targets.append((mr, Chem.MolToSmiles(m), key, ha))

    if len(targets) < args.n:
        raise SystemExit(f"only {len(targets)} accepted (< {args.n}); enlarge pool")

    # write targets.smi (SMILES only), subset200.smi, targets.tsv, meta
    (outdir / "targets.smi").write_text("".join(f"{t[1]}\n" for t in targets))
    (outdir / f"subset{args.subset}.smi").write_text(
        "".join(f"{t[1]}\n" for t in targets[: args.subset])
    )
    with open(outdir / "targets.tsv", "w") as f:
        f.write("rank\tmolregno\tsmiles\tinchikey\theavy_atoms\n")
        for i, t in enumerate(targets, 1):
            f.write(f"{i}\t{t[0]}\t{t[1]}\t{t[2]}\t{t[3]}\n")

    meta = {
        "source": "ChEMBL 35 (compound_structures, molecule_type='Small molecule')",
        "seed": args.seed,
        "n_targets_requested": args.n,
        "n_targets_obtained": len(targets),
        "subset": args.subset,
        "eligible_pool": len(pool),
        "n_drawn": n_drawn,
        "n_unparseable": n_unparseable,
        "n_size_filtered": n_size,
        "n_duplicate_inchikey": n_dup,
        "min_heavy_atoms": args.min_heavy_atoms,
        "max_heavy_atoms": args.max_heavy_atoms,
        "fragment_rule": "largest organic fragment (salt stripping)",
        "dedup": "by InChIKey",
    }
    (outdir / "sample_meta.json").write_text(json.dumps(meta, indent=2))
    heavy = [t[3] for t in targets]
    print(f"accepted {len(targets)} targets; heavy atoms "
          f"min {min(heavy)} median {sorted(heavy)[len(heavy)//2]} max {max(heavy)}")
    print("meta:", json.dumps(meta))


if __name__ == "__main__":
    main()

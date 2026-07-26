"""SAscore + SCScore for a SMILES list -> CSV, with PER-MOLECULE timing.

Columns: smiles, sascore, sa_sec, scscore, sc_sec
Timing uses time.perf_counter() around each individual scorer call (model load
excluded). SAscore/SCScore are fingerprint-based and sub-millisecond; the columns
make the cost of every method explicit alongside SynOmega bb_coverage `sec`.
"""
import argparse, csv, os, sys, glob, time
from rdkit import Chem, RDConfig, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
sys.path.insert(0, "/home/zbc/scscore_repo")
from scscore.standalone_model_numpy import SCScorer
import numpy as _np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sc = SCScorer()
    sc.restore(glob.glob("/home/zbc/scscore_repo/models/full_reaxys_model_1024uint8/*.json.gz")[0])

    smis = [l.strip().split()[0] for l in open(args.smiles) if l.strip()]
    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["smiles", "sascore", "sa_sec", "scscore", "sc_sec"])
        for i, smi in enumerate(smis, 1):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                w.writerow([smi, "", "", "", ""]); continue
            try:
                ts = time.perf_counter()
                sa = round(float(sascorer.calculateScore(m)), 4)
                sa_sec = round(time.perf_counter() - ts, 6)
            except Exception:
                sa, sa_sec = "", ""
            try:
                ts = time.perf_counter()
                scs = round(float(_np.ravel(sc.get_score_from_smi(smi)[1])[0]), 4)
                sc_sec = round(time.perf_counter() - ts, 6)
            except Exception:
                scs, sc_sec = "", ""
            w.writerow([smi, sa, sa_sec, scs, sc_sec])
            if i % 200 == 0:
                print(f"  {i}/{len(smis)}", flush=True)
    print(f"done {len(smis)} in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

"""RAscore (Reymond RAScorerXGB) for a SMILES list -> CSV, per-molecule timing.
Columns: smiles, rascore, ra_sec.  RAscore in [0,1]: higher = more retrosynthetically accessible.
"""
import argparse, csv, glob, time, sys
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smiles",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--model",default="/home/zbc/Projects/retrosyn/bio-retrosyn/data/raw/RAscore_repo/RAscore/models/XGB_chembl_ecfp_counts/model.pkl")
    a=ap.parse_args()
    from RAscore import RAscore_XGB
    s=RAscore_XGB.RAScorerXGB(a.model)
    smis=[l.strip().split()[0] for l in open(a.smiles) if l.strip()]
    t0=time.time()
    with open(a.out,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["smiles","rascore","ra_sec"])
        for i,smi in enumerate(smis,1):
            ts=time.perf_counter()
            try: val=round(float(s.predict(smi)),4); sec=round(time.perf_counter()-ts,6)
            except Exception: val,sec="",""
            w.writerow([smi,val,sec])
            if i%200==0: print(f"  {i}/{len(smis)}",flush=True)
    print(f"done {len(smis)} in {time.time()-t0:.0f}s -> {a.out}",flush=True)
if __name__=="__main__": main()

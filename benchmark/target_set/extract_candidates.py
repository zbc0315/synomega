#!/usr/bin/env python3
"""From all_routes.jsonl, pull candidate targets for the Figure 3(e) route-comparison:
the simplify-model best route is short (<=2 steps) and the original-model best route
is longer (>=4 steps), both fully solved to purchasable building blocks.
Each candidate dumps every reaction step as 'product >> reactant.reactant' so a chemist
(or a subagent) can judge whether every step is a chemically reasonable disconnection.
Run: python extract_candidates.py > candidates.json
"""
import json
import sys


def steps(tree):
    """Yield (product_smiles, [reactant_smiles]) for each reaction node, top-down."""
    out = []

    def rec(mnode):
        for ch in mnode.get("children", []):
            if ch.get("type") == "reaction":
                reactants = [rm["smiles"] for rm in ch.get("children", [])]
                out.append((mnode["smiles"], reactants))
                for rm in ch.get("children", []):
                    rec(rm)
    rec(tree)
    return out


def all_leaves_in_stock(tree):
    ok = [True]

    def rec(n):
        kids = n.get("children", [])
        rxn = [c for c in kids if c.get("type") == "reaction"]
        if not rxn:  # leaf molecule
            if not n.get("in_stock"):
                ok[0] = False
        for c in rxn:
            for rm in c.get("children", []):
                rec(rm)
    rec(tree)
    return ok[0]


def main():
    cands = []
    for line in open("routes/all_routes.jsonl"):
        r = json.loads(line)
        o, s = r.get("original"), r.get("simplify")
        if not (o and s and o.get("solved") and s.get("solved")):
            continue
        ot, st = o["route"]["tree"], s["route"]["tree"]
        o_steps, s_steps = steps(ot), steps(st)
        if not (len(s_steps) <= 2 and len(o_steps) >= 4):
            continue
        if not (all_leaves_in_stock(ot) and all_leaves_in_stock(st)):
            continue
        cands.append({
            "smiles": r["smiles"],
            "simplify_nsteps": len(s_steps),
            "original_nsteps": len(o_steps),
            "simplify_steps": [f"{p} >> {'.'.join(rs)}" for p, rs in s_steps],
            "original_steps": [f"{p} >> {'.'.join(rs)}" for p, rs in o_steps],
        })
    # smaller original first (cleaner contrast), then by target size
    cands.sort(key=lambda c: (c["original_nsteps"], len(c["smiles"])))
    print(json.dumps(cands, indent=1))
    print(f"# {len(cands)} candidates", file=sys.stderr)


if __name__ == "__main__":
    main()

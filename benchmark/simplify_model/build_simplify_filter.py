"""Filter r20 (radius-0, min_count=20) templates down to a *simplification-only*
label set, for a retrosynthesis model whose disconnections always yield simpler
precursors than the product.

A template (written retro as ``product >> reactants``) is KEPT if it is a
simplifying template: the product side is a single molecule and the reactant side
has two or more molecules, so the target is split into smaller precursors.

Outputs (into --out):
    kept_old_labels.json        sorted list of original label ids kept
    old_to_new_label.json       {old_label(str): new_label(int)} compact 0..K-1
    label_to_template_smarts.json  {new_label(str): smarts}
    label_to_template_id.json   {new_label(str): original template_id}  (remapped)
    filter_meta.json            counts

Usage:
    python build_simplify_filter.py \
        --smarts   label_to_template_smarts_r20.json \
        --label-id label_to_template_id.json \
        --out      processed_r20_center_simplify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_simplifying(smarts: str) -> bool:
    """True if the retro template's product is one molecule and the reactant side
    has two or more molecules."""
    if ">>" not in smarts:
        return False
    lhs, rhs = smarts.split(">>")          # lhs = product side, rhs = reactant side
    l_parts = [s for s in lhs.split(".") if s]
    r_parts = [s for s in rhs.split(".") if s]
    return len(l_parts) == 1 and len(r_parts) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smarts", required=True, help="label_to_template_smarts_r20.json")
    ap.add_argument("--label-id", required=True, help="label_to_template_id.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    smarts_map = json.loads(Path(args.smarts).read_text())        # {label(str): smarts}
    label_id = json.loads(Path(args.label_id).read_text())        # {label(str): template_id}

    kept = sorted(int(l) for l, s in smarts_map.items() if is_simplifying(s))
    old_to_new = {str(old): new for new, old in enumerate(kept)}
    new_smarts = {str(new): smarts_map[str(old)] for new, old in enumerate(kept)}
    new_label_id = {
        str(new): label_id.get(str(old))
        for new, old in enumerate(kept)
        if str(old) in label_id
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kept_old_labels.json").write_text(json.dumps(kept))
    (out / "old_to_new_label.json").write_text(json.dumps(old_to_new))
    (out / "label_to_template_smarts.json").write_text(json.dumps(new_smarts))
    (out / "label_to_template_id.json").write_text(json.dumps(new_label_id))

    meta = {
        "source_smarts": args.smarts,
        "n_templates_in": len(smarts_map),
        "n_kept": len(kept),
        "kept_frac": len(kept) / max(1, len(smarts_map)),
        "criterion": "keep = product 1 molecule AND reactants >= 2 molecules (simplifying)",
    }
    (out / "filter_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

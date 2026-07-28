"""Build the *simplification* featurized dataset by filtering the existing
featurized_r20_center shards -- no re-featurization.

Each packed shard is a dict of flat tensors:
    {x_cat, edge_index_cat, edge_attr_cat, node_ptr, edge_ptr, y,
     center_mask_cat, n_items}
where edge_index_cat holds PER-ITEM LOCAL atom indices (offsets are applied by
the dataloader), so items can be dropped/reordered without touching edge_index.

For every shard we keep only samples whose label is in the kept set, slice their
atom/edge ranges out with a vectorized mask, recompute node_ptr/edge_ptr, and
remap y to the compact 0..K-1 label space. Empty shards are skipped.

Usage:
    python scripts/build_simplify_featurized.py \
        --src   data/featurized_r20_center \
        --out   /home/zbc/.../featurized_r20_center_simplify \
        --filter data/processed_r20_center_simplify \
        --splits train val test
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch


def build_lut(old_to_new: dict, size: int) -> torch.Tensor:
    lut = torch.full((size,), -1, dtype=torch.long)
    for old_str, new in old_to_new.items():
        lut[int(old_str)] = int(new)
    return lut


def filter_shard(sh: dict, kept_tensor: torch.Tensor, lut: torch.Tensor):
    y = sh["y"].long()
    node_ptr = sh["node_ptr"].long()
    edge_ptr = sh["edge_ptr"].long()

    keep_mask = torch.isin(y, kept_tensor)
    n_keep = int(keep_mask.sum())
    if n_keep == 0:
        return None

    item_natoms = node_ptr[1:] - node_ptr[:-1]
    item_nedges = edge_ptr[1:] - edge_ptr[:-1]
    atom_keep = keep_mask.repeat_interleave(item_natoms)
    edge_keep = keep_mask.repeat_interleave(item_nedges)

    new_natoms = item_natoms[keep_mask]
    new_nedges = item_nedges[keep_mask]
    new_node_ptr = torch.cat([torch.zeros(1, dtype=torch.long), new_natoms.cumsum(0)])
    new_edge_ptr = torch.cat([torch.zeros(1, dtype=torch.long), new_nedges.cumsum(0)])

    new_y = lut[y[keep_mask]]
    assert int((new_y < 0).sum()) == 0, "kept label mapped to -1 -- LUT/kept mismatch"

    out = {
        "x_cat": sh["x_cat"][atom_keep].contiguous(),
        "edge_index_cat": sh["edge_index_cat"][:, edge_keep].contiguous(),
        "edge_attr_cat": sh["edge_attr_cat"][edge_keep].contiguous(),
        "node_ptr": new_node_ptr,
        "edge_ptr": new_edge_ptr,
        "y": new_y,
        "n_items": n_keep,
    }
    if "center_mask_cat" in sh:
        out["center_mask_cat"] = sh["center_mask_cat"][atom_keep].contiguous()
    return out


def process_split(src: Path, out: Path, split: str, kept_tensor, lut) -> list[int]:
    shard_paths = sorted((src / split).glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no shards under {src / split}")
    out_split = out / split
    out_split.mkdir(parents=True, exist_ok=True)

    sizes: list[int] = []
    n_in = n_out = 0
    t0 = time.time()
    new_id = 0
    for sp in shard_paths:
        sh = torch.load(sp, weights_only=False)
        n_in += int(sh["n_items"])
        new = filter_shard(sh, kept_tensor, lut)
        del sh
        if new is None:
            continue
        torch.save(new, out_split / f"shard_{new_id:04d}.pt")
        sizes.append(int(new["n_items"]))
        n_out += int(new["n_items"])
        new_id += 1
        del new
        el = time.time() - t0
        print(f"  [{split}] {new_id:>4d} shards written, kept {n_out:,}/{n_in:,} "
              f"({n_out/max(1,n_in):.1%}, {n_in/max(1e-6,el):,.0f} in/s)", flush=True)

    (out / f"{split}_sizes.json").write_text(json.dumps(sizes))
    print(f"DONE {split}: kept {n_out:,}/{n_in:,} in {time.time()-t0:.0f}s", flush=True)
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="featurized_r20_center dir")
    ap.add_argument("--out", required=True, help="output featurized dir")
    ap.add_argument("--filter", required=True, help="processed_r20_center_simplify dir")
    ap.add_argument("--src-num-classes", type=int, default=64366)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    fdir = Path(args.filter)
    old_to_new = json.loads((fdir / "old_to_new_label.json").read_text())
    kept = json.loads((fdir / "kept_old_labels.json").read_text())
    kept_tensor = torch.tensor(sorted(int(x) for x in kept), dtype=torch.long)
    lut = build_lut(old_to_new, args.src_num_classes)
    num_classes = len(old_to_new)

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in args.splits:
        sizes = process_split(src, out, split, kept_tensor, lut)
        counts[split] = int(sum(sizes))

    # meta.json -- num_classes auto-overrides the training config at runtime
    meta = {
        "num_classes": num_classes,
        "min_count": 20,
        "predict_center": True,
        "n_train": counts.get("train", 0),
        "n_val": counts.get("val", 0),
        "n_test": counts.get("test", 0),
        "source": str(src),
        "filter": str(fdir),
        "note": "simplifying templates only",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    # carry the label maps alongside the featurized data
    for fn in ("label_to_template_id.json", "label_to_template_smarts.json",
               "old_to_new_label.json", "kept_old_labels.json", "filter_meta.json"):
        shutil.copyfile(fdir / fn, out / fn)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

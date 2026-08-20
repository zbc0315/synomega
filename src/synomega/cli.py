"""synomega command line.

    synomega plan  --target "CC(=O)Nc1ccccc1" --model runs/uspto50k_r0_min10 \
                 --stock emolecules.smi --max-steps 5
    synomega score --targets targets.smi --model runs/... --stock ... --out report.json
    synomega build-stock --catalogue emolecules.smi.gz --out emolecules.keys.gz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_stock(path: str, keys: bool):
    from .stock import InMemoryStock

    if keys:
        return InMemoryStock.from_keys_file(path)
    return InMemoryStock.from_file(path)


def _load_model(args):
    from .singlestep import TemplateGNN

    # No --model given: download and use a default pretrained model. `score`
    # defaults to the simplification-constrained ("breaking") model; `plan`
    # defaults to the unconstrained ("original") one (see `simplify` default set
    # per-subcommand in `build_parser`).
    if not getattr(args, "model", None):
        if getattr(args, "simplify", False):
            return TemplateGNN.simplify(
                device=args.device, topk_templates=args.expansion_width
            )
        return TemplateGNN.default(
            device=args.device, topk_templates=args.expansion_width
        )
    return TemplateGNN.from_pretrained(
        args.model,
        templates_path=args.templates,
        device=args.device,
        topk_templates=args.expansion_width,
    )


def _build_planner(args):
    from .planner import Planner
    from .stock import InMemoryStock

    model = _load_model(args)
    # No --stock given: download and use the default building-block stock.
    if not getattr(args, "stock", None):
        stock = InMemoryStock.default()
    else:
        stock = _load_stock(args.stock, args.stock_is_keys)
    return Planner(
        model,
        stock,
        algorithm=args.algorithm,
        expansion_width=args.expansion_width,
        max_depth=args.max_steps,
        time_limit=args.time_limit,
        max_expansions=args.max_expansions,
        cache_path=args.cache,
    )


def cmd_plan(args) -> int:
    planner = _build_planner(args)
    result = planner.plan(args.target, exclude_target=args.exclude_target)
    print(f"solved: {result.solved}   {result.stats.as_dict()}")
    if result.best_route is not None:
        print()
        print(result.best_route.describe())
        if args.out:
            Path(args.out).write_text(result.best_route.to_json())
            print(f"\nsaved -> {args.out}")
    else:
        print("no route found")
    return 0 if result.solved else 1


def cmd_score(args) -> int:
    from .synthesizability import SynthesizabilityScorer

    planner = _build_planner(args)
    scorer = SynthesizabilityScorer(planner)

    targets = [
        line.split()[0]
        for line in Path(args.targets).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    report = scorer.score_batch(
        targets, max_steps=args.max_steps, exclude_target=args.exclude_target
    )

    print(report.describe())
    if args.out:
        Path(args.out).write_text(report.to_json())
        print(f"\nsaved -> {args.out}")
    return 0


def cmd_download(args) -> int:
    """Pre-fetch the default model + stock into the local cache."""
    from .data import cache_dir, ensure_default_assets

    run_dir, stock_path = ensure_default_assets()
    print(f"model: {run_dir}")
    print(f"stock: {stock_path}")
    print(f"cache: {cache_dir()}")
    return 0


def cmd_forward(args) -> int:
    """Predict product(s) from reactants with the forward template model."""
    from .forward import ForwardTemplateGNN

    if args.model:
        model = ForwardTemplateGNN.from_pretrained(
            args.model, templates_path=args.templates,
            device=args.device, topk_templates=args.topk_templates,
        )
    else:
        model = ForwardTemplateGNN.default(
            device=args.device, topk_templates=args.topk_templates,
        )
    preds = model.predict(args.reactants, top_k=args.top_k)
    if not preds:
        print("no product predicted")
        return 1
    for i, p in enumerate(preds, 1):
        print(f"{i:>2}. {p.product}\tscore={p.score:.4f}\ttemplate={p.template_id}")
    return 0


def cmd_evolve(args) -> int:
    """Grow a forward synthesis network from starting reactants."""
    from .forward import ForwardTemplateGNN, MultiComponentEvolution

    if args.reactants_file:
        reactants = [
            line.split()[0]
            for line in Path(args.reactants_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        reactants = [r for r in args.reactants.split(".") if r]
    if not reactants:
        print("no starting reactants given")
        return 1

    if args.model:
        model = ForwardTemplateGNN.from_pretrained(
            args.model, templates_path=args.templates,
            device=args.device, topk_templates=args.topk_templates,
        )
    else:
        model = ForwardTemplateGNN.default(
            device=args.device, topk_templates=args.topk_templates,
        )

    evolver = MultiComponentEvolution(
        model,
        max_depth=args.max_depth,
        score_threshold=args.score_threshold,
        mode=args.mode,
        work_dir=args.work_dir,
        forward_top_k=args.forward_top_k,
        allow_self_pair=args.allow_self_pair,
        frontier_width=args.frontier_width,
    )
    result = evolver.evolve(reactants)
    print(result.describe())
    if args.out:
        Path(args.out).write_text(result.to_json())
        print(f"\nsaved -> {args.out}")
    result.close()
    return 0


def cmd_build_stock(args) -> int:
    """Precompute InChIKeys once so later loads are fast."""
    from .stock import InMemoryStock

    print(f"reading {args.catalogue} ...", file=sys.stderr)
    stock = InMemoryStock.from_file(args.catalogue, smiles_column=args.smiles_column)
    print(f"  {len(stock):,} unique InChIKeys", file=sys.stderr)
    stock.save_keys(args.out)
    print(f"saved -> {args.out}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="synomega", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--model", default=None,
                        help="run dir containing best.pt "
                             "(default: download the pretrained model)")
        sp.add_argument("--templates", default=None,
                        help="label_to_template_smarts.json / templates TSV")
        sp.add_argument("--stock", default=None,
                        help="building-block file "
                             "(default: download the ZINC in-stock set)")
        sp.add_argument("--stock-is-keys", action="store_true",
                        help="stock file holds precomputed InChIKeys")
        sp.add_argument("--algorithm", default="retrostar",
                        choices=["retrostar", "bfs", "mcts"])
        sp.add_argument("--max-steps", type=int, default=5)
        sp.add_argument("--exclude-target", action="store_true",
                        help="treat the target as not purchasable even if it is "
                             "in the stock (so a catalogue molecule is not "
                             "trivially solved in zero steps)")
        sp.add_argument("--expansion-width", type=int, default=50)
        sp.add_argument("--time-limit", type=float, default=60.0)
        sp.add_argument("--max-expansions", type=int, default=500)
        sp.add_argument("--device", default=None)
        sp.add_argument("--cache", default=None, help="SQLite expansion cache path")
        sp.add_argument("--out", default=None)

    sp_plan = sub.add_parser("plan", help="find routes to one target")
    sp_plan.add_argument("--target", required=True)
    add_common(sp_plan)
    # `plan` defaults to the unconstrained ("original") model; --simplify opts in
    # to the fragmentation-only ("breaking") model (only used when no --model given).
    sp_plan.add_argument("--simplify", dest="simplify", action="store_true",
                         help="use the simplification-constrained (breaking) model")
    sp_plan.set_defaults(func=cmd_plan, simplify=False)

    sp_score = sub.add_parser("score", help="synthesizability over a target list")
    sp_score.add_argument("--targets", required=True, help="one SMILES per line")
    add_common(sp_score)
    # `score` defaults to the fragmentation-only ("breaking") model, the recommended
    # model for synthesizability scoring; --original reverts to the unconstrained one
    # (only used when no --model given).
    sp_score.add_argument("--original", dest="simplify", action="store_false",
                          help="score with the unconstrained (original) model instead")
    sp_score.set_defaults(func=cmd_score, simplify=True)

    sub.add_parser(
        "download",
        help="pre-fetch the default model + stock into the local cache",
    ).set_defaults(func=cmd_download)

    sp_fwd = sub.add_parser("forward", help="predict product(s) from reactants")
    sp_fwd.add_argument("reactants",
                        help="reactant SMILES (dot-separated for multiple)")
    sp_fwd.add_argument("--top-k", type=int, default=10,
                        help="number of ranked products to print")
    sp_fwd.add_argument("--topk-templates", type=int, default=10,
                        help="templates searched per reactant set")
    sp_fwd.add_argument("--model", default=None,
                        help="run dir containing best.pt "
                             "(default: download the forward model)")
    sp_fwd.add_argument("--templates", default=None,
                        help="label_to_template_smarts.json / templates TSV")
    sp_fwd.add_argument("--device", default=None)
    sp_fwd.set_defaults(func=cmd_forward)

    sp_evo = sub.add_parser(
        "evolve", help="grow a forward synthesis network from reactants")
    sp_evo.add_argument("--reactants", default="",
                        help="starting reactant SMILES; '.' separates individual "
                             "starting molecules (each seeds the pool separately)")
    sp_evo.add_argument("--reactants-file", default=None,
                        help="file with one starting reactant SMILES per line "
                             "(first whitespace-separated column)")
    sp_evo.add_argument("--max-depth", type=int, required=True,
                        help="max synthesis-tree depth (not step count)")
    sp_evo.add_argument("--score-threshold", type=float, required=True,
                        help="min total score for a molecule to be reactable")
    sp_evo.add_argument("--mode", default="memory",
                        choices=["memory", "disk", "auto"],
                        help="'disk' spills intermediates to SQLite under --work-dir")
    sp_evo.add_argument("--work-dir", default=None,
                        help="directory for the SQLite store (disk mode)")
    sp_evo.add_argument("--forward-top-k", type=int, default=5,
                        help="products taken per reaction pair")
    sp_evo.add_argument("--frontier-width", type=int, default=None,
                        help="max selectable molecules paired per round (top by "
                             "score); recommended for many starting reactants")
    sp_evo.add_argument("--no-self-pair", dest="allow_self_pair",
                        action="store_false",
                        help="disallow a molecule reacting with itself (A+A)")
    sp_evo.add_argument("--topk-templates", type=int, default=10)
    sp_evo.add_argument("--model", default=None,
                        help="run dir containing best.pt "
                             "(default: download the forward model)")
    sp_evo.add_argument("--templates", default=None,
                        help="label_to_template_smarts.json / templates TSV")
    sp_evo.add_argument("--device", default=None)
    sp_evo.add_argument("--out", default=None)
    sp_evo.set_defaults(func=cmd_evolve, allow_self_pair=True)

    sp_stock = sub.add_parser("build-stock", help="catalogue -> InChIKey file")
    sp_stock.add_argument("--catalogue", required=True)
    sp_stock.add_argument("--smiles-column", type=int, default=0)
    sp_stock.add_argument("--out", required=True)
    sp_stock.set_defaults(func=cmd_build_stock)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Download-on-demand for the default pretrained model and building-block stock.

A pip wheel cannot carry hundreds of megabytes of weights and data (PyPI's file
limits, and it would force every install to pull them). So the default assets are
hosted as **GitHub Release assets** and fetched into a local cache the first time
they are needed — the same pattern spaCy, nltk and HuggingFace use.

Cache location: ``~/.cache/synomega`` (override with the ``SYNOMEGA_CACHE`` env
var). Nothing is downloaded until you call one of the ``default()`` helpers,
``synomega download`` on the CLI, or :func:`load_default_planner`.

The default model is a D-MPNN template classifier; the default stock is the ZINC
in-stock building-block set.
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Data assets are pinned to a dedicated release tag, decoupled from the code
# version so the (large) files are not re-uploaded on every code release.
#
# Two mirrors host the same assets: GitHub Releases (international) and a USTC
# GitLab generic-package registry (fast within China). By default the base is
# auto-detected by latency, domestic-first; override with:
#   * SYNOMEGA_ASSETS_BASE=<url>          use this base verbatim, or
#   * SYNOMEGA_MIRROR=ustc | github | cn  force a specific mirror.
_MIRRORS = {
    "github": "https://github.com/zbc0315/synomega/releases/download/assets-v1",
    "ustc": (
        "https://git.ustc.edu.cn/api/v4/projects/"
        "zbc1%2Fsynomega-assets/packages/generic/synomega/assets-v1"
    ),
}
_MIRRORS["cn"] = _MIRRORS["ustc"]

# Probe order (domestic first) and the small-ish file used to measure latency.
_AUTO_ORDER = ("ustc", "github")
_PROBE_FILE = "label_to_template_smarts_r20.json"

_resolved_base: str | None = None


def _probe(base: str, timeout: float = 3.0) -> float | None:
    """Round-trip time to fetch one byte from ``base``; None if unreachable."""
    import time

    req = urllib.request.Request(
        f"{base}/{_PROBE_FILE}", headers={"Range": "bytes=0-0"}
    )
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(1)
        return time.monotonic() - t0
    except Exception:  # noqa: BLE001 - any failure just means "not this one"
        return None


def _auto_base() -> str:
    best, best_t = None, float("inf")
    for name in _AUTO_ORDER:
        t = _probe(_MIRRORS[name])
        if t is None:
            continue
        if t < 1.0:  # fast enough — don't bother probing the rest
            return _MIRRORS[name]
        if t < best_t:
            best, best_t = name, t
    return _MIRRORS[best] if best else _MIRRORS["github"]


def _base() -> str:
    global _resolved_base
    override = os.environ.get("SYNOMEGA_ASSETS_BASE", "").strip()
    if override:
        return override.rstrip("/")
    mirror = os.environ.get("SYNOMEGA_MIRROR", "").strip().lower()
    if mirror in _MIRRORS:
        return _MIRRORS[mirror]
    if _resolved_base is None:
        _resolved_base = _auto_base()
    return _resolved_base

# Local filename -> remote asset name.
_MODEL_FILES = {
    "best.pt": "r20_center-best.pt",
    # Renamed so TemplateGNN.from_pretrained auto-discovers it in the run dir.
    "label_to_template_smarts.json": "label_to_template_smarts_r20.json",
}
# Simplification-constrained single-step model (fragmentation-only templates):
# proposes only disconnections that split the target, giving cheaper multi-step
# search at matched solvability. Same file layout as the default model.
_SIMPLIFY_MODEL_FILES = {
    "best.pt": "r20_center_simplify-best.pt",
    "label_to_template_smarts.json": "label_to_template_smarts_r20_simplify.json",
}
_STOCK_FILE = "zinc_stock_keys.txt.gz"
_PLAUSIBILITY_FILE = "plaus_dual-best.pt"
# Forward reaction-prediction model (reactants -> product). Shares the retro r20
# template inventory, so it reuses the already-hosted label->SMARTS map and only
# ships its own checkpoint.
_FORWARD_MODEL_FILES = {
    "best.pt": "r20_forward-best.pt",
    "label_to_template_smarts.json": "label_to_template_smarts_r20.json",
}


def cache_dir() -> Path:
    """Directory where downloaded assets live (created if missing)."""
    d = Path(
        os.environ.get("SYNOMEGA_CACHE", Path.home() / ".cache" / "synomega")
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` atomically, with a progress line on stderr."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"synomega: downloading {url}", file=sys.stderr)
    try:
        with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https host
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done / total
                        print(
                            f"\r  {done >> 20:,} / {total >> 20:,} MiB "
                            f"({pct:4.1f}%)",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
            print("", file=sys.stderr)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"failed to download {url} (HTTP {exc.code}). The default assets may "
            f"not be published yet; pass an explicit model/stock instead."
        ) from exc
    tmp.replace(dest)


def _ensure(local_name: str, remote_name: str, subdir: Path) -> Path:
    dest = subdir / local_name
    if not dest.exists():
        subdir.mkdir(parents=True, exist_ok=True)
        _download(f"{_base()}/{remote_name}", dest)
    return dest


def ensure_default_model() -> Path:
    """Download the default model files; return the run directory to load from."""
    run = cache_dir() / "r20_center"
    for local, remote in _MODEL_FILES.items():
        _ensure(local, remote, run)
    return run


def ensure_simplify_model() -> Path:
    """Download the simplification-constrained model files; return the run dir.

    ``SYNOMEGA_SIMPLIFY_MODEL=/path/to/run_dir`` overrides with a local run dir.
    """
    override = os.environ.get("SYNOMEGA_SIMPLIFY_MODEL", "").strip()
    if override:
        return Path(override)
    run = cache_dir() / "r20_center_simplify"
    for local, remote in _SIMPLIFY_MODEL_FILES.items():
        _ensure(local, remote, run)
    return run


def ensure_forward_model() -> Path:
    """Download the forward reaction-prediction model files; return the run dir.

    ``SYNOMEGA_FORWARD_MODEL=/path/to/run_dir`` overrides with a local run dir.
    The label->SMARTS map is the retro r20 map (identical label space).
    """
    override = os.environ.get("SYNOMEGA_FORWARD_MODEL", "").strip()
    if override:
        return Path(override)
    run = cache_dir() / "r20_forward"
    for local, remote in _FORWARD_MODEL_FILES.items():
        _ensure(local, remote, run)
    return run


def ensure_default_stock() -> Path:
    """Download the default building-block stock; return its path."""
    return _ensure(_STOCK_FILE, _STOCK_FILE, cache_dir())


def ensure_default_plausibility_model() -> Path:
    """Download the default dual-tower plausibility checkpoint; return its path.

    ``SYNOMEGA_PLAUSIBILITY_MODEL=/path/to/best.pt`` overrides with a local file.
    """
    override = os.environ.get("SYNOMEGA_PLAUSIBILITY_MODEL", "").strip()
    if override:
        return Path(override)
    return _ensure(_PLAUSIBILITY_FILE, _PLAUSIBILITY_FILE, cache_dir())


def ensure_default_assets() -> tuple[Path, Path]:
    """Download both the default model and stock. Returns (run_dir, stock_path)."""
    return ensure_default_model(), ensure_default_stock()


def clear_cache() -> None:
    """Delete all downloaded assets."""
    shutil.rmtree(cache_dir(), ignore_errors=True)


__all__ = [
    "cache_dir",
    "ensure_default_model",
    "ensure_simplify_model",
    "ensure_forward_model",
    "ensure_default_stock",
    "ensure_default_plausibility_model",
    "ensure_default_assets",
    "clear_cache",
]

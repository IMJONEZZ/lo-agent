"""The Jacobian lens — access-ladder Rung 6 (activations) over HTTP.

Reads and writes the residual stream of a GGUF model served by llama.cpp,
through a small activation sidecar (``jlens-server``). Lens math is pure numpy;
requires the ``lens`` extra (``uv sync --extra lens`` — plain sync drops it).

    from local_harness.jlens import JacobianLensGGUF, LensReadout, ReadoutWeights
    from local_harness.jlens import NativeClient   # sidecar client

See docs/jlens-integration.md and
thoughts/shared/plans/2026-07-12-jlens-rung6-activations.md.
"""

from __future__ import annotations

_EXTRA_MSG = (
    "the Jacobian lens needs the 'lens' extra (numpy + gguf). Install it:\n"
    "    uv sync --extra lens        # from a source checkout\n"
    "    uv tool install 'lo-agent[lens]'\n"
    "    brew upgrade lo-agent       # Homebrew ships it since 0.2.2_1\n"
)


def missing_lens_deps() -> list[str]:
    """Names from the lens extra that aren't importable (empty = all good)."""
    import importlib.util

    def _have(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is not None
        except ModuleNotFoundError:
            return False

    return [m for m in ("numpy", "gguf") if not _have(m)]


def __getattr__(name: str):
    # Lazy so importing the package (e.g. for capability flags) never forces
    # numpy/gguf; the clear error only fires if you actually use the math.
    _map = {
        "JacobianLensGGUF": ("lens", "JacobianLensGGUF"),
        "LensReadout": ("readout", "LensReadout"),
        "compute_grid": ("readout", "compute_grid"),
        "SliceGrid": ("readout", "SliceGrid"),
        "ReadoutWeights": ("model_reader", "ReadoutWeights"),
        "NativeClient": ("native_client", "NativeClient"),
        "ForwardResult": ("native_client", "ForwardResult"),
        "fit_regression": ("fitting", "fit_regression"),
        "load_corpus": ("fitting", "load_corpus"),
    }
    if name not in _map:
        raise AttributeError(name)
    module, attr = _map[name]
    try:
        import importlib

        mod = importlib.import_module(f"local_harness.jlens.{module}")
    except ImportError as e:  # numpy/gguf missing
        raise ImportError(_EXTRA_MSG) from e
    return getattr(mod, attr)


__all__ = [
    "JacobianLensGGUF", "LensReadout", "compute_grid", "SliceGrid",
    "ReadoutWeights", "NativeClient", "ForwardResult",
    "fit_regression", "load_corpus",
]

"""Optional DSPy integration (install with: uv sync --extra dspy).

The native optimizers in bootstrap.py cover BootstrapFewShot/mini-MIPRO
without DSPy's dependency tree; this adapter exists for users who want the
full DSPy optimizer zoo (MIPROv2, GEPA) against their local endpoint.
"""

from __future__ import annotations


def configure_dspy(base_url: str, model: str, api_key: str = "local"):
    """Point DSPy at a local OpenAI-compatible endpoint and return the LM."""
    try:
        import dspy
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "DSPy is not installed. Use the native optimizers in "
            "local_harness.optimize, or install with: uv sync --extra dspy"
        ) from e
    lm = dspy.LM(f"openai/{model}", api_base=f"{base_url.rstrip('/')}/v1", api_key=api_key)
    dspy.configure(lm=lm)
    return lm

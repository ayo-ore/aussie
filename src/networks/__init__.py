# Lazy loading to avoid importing heavy dependencies unless needed
# This allows users to run experiments without installing xformers, lloca, etc.

__all__ = ["MLP", "TransformerEncoder", "LGATr"]


def __getattr__(name):
    """Lazy import networks only when accessed"""
    match name:
        case "MLP":
            from .mlp import MLP
            return MLP
        case "TransformerEncoder":
            from .transformer import TransformerEncoder
            return TransformerEncoder
        case "LGATr":
            from .lgatr import LGATr
            return LGATr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

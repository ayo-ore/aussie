# Lazy loading to avoid importing heavy dependencies unless needed

__all__ = [
    "Classifier",
    "PartClassifier",
    "AutoDiffUnfolder",
    "OmniFolder",
    "KernelUnfolder",
]


def __getattr__(name):
    """Lazy import models only when accessed"""
    match name:
        case "Classifier":
            from .classifier import Classifier
            return Classifier
        case "PartClassifier":
            from .classifier import PartClassifier
            return PartClassifier
        case "AutoDiffUnfolder":
            from .unfolder import AutoDiffUnfolder
            return AutoDiffUnfolder
        case "OmniFolder":
            from .unfolder import OmniFolder
            return OmniFolder
        case "KernelUnfolder":
            from .unfolder import KernelUnfolder
            return KernelUnfolder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

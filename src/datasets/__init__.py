# Lazy loading to avoid importing heavy dependencies unless needed
# This allows users to run experiments without installing all dependencies

__all__ = [
    "UnfoldingData",
    "GaussianToyData",
    "GaussianToyProcess",
    "ZJetData",
    "ZJetProcess",
    "ZJetParticleData",
    "ZJetParticleProcess",
    "YukawaData",
    "YukawaProcess",
]


def __getattr__(name):
    """Lazy import datasets only when accessed"""
    match name:
        case "UnfoldingData":
            from .base_dataset import UnfoldingData
            return UnfoldingData
        case "GaussianToyData":
            from .gaussian import GaussianToyData
            return GaussianToyData
        case "GaussianToyProcess":
            from .gaussian import GaussianToyProcess
            return GaussianToyProcess
        case "ZJetData":
            from .zjet import ZJetData
            return ZJetData
        case "ZJetProcess":
            from .zjet import ZJetProcess
            return ZJetProcess
        case "ZJetParticleData":
            from .zjet_particle import ZJetParticleData
            return ZJetParticleData
        case "ZJetParticleProcess":
            from .zjet_particle import ZJetParticleProcess
            return ZJetParticleProcess
        case "YukawaData":
            from .yukawa import YukawaData
            return YukawaData
        case "YukawaProcess":
            from .yukawa import YukawaProcess
            return YukawaProcess
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

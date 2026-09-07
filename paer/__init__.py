"""PAER-AES: perturbation-aware evidence routing for robust AES."""

def __getattr__(name):
    # CPU-only preparation/CLI validation must not eagerly load CUDA libraries.
    if name in ("PAERForEssayScoring", "PAEROutput"):
        from paer import modeling_paer
        return getattr(modeling_paer, name)
    if name in ("PAERV3ForEssayScoring", "PAERV3Output"):
        from paer import modeling_paer_v3
        return getattr(modeling_paer_v3, name)
    raise AttributeError(name)

__all__ = [
    "PAERForEssayScoring",
    "PAEROutput",
    "PAERV3ForEssayScoring",
    "PAERV3Output",
]

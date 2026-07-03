"""Abstract inference: fuzzy diffusion, typed compositional, operator algebra."""
from src.reasoning.abstract_inference import (
    FuzzyInferenceEngine,
    HallucinationGuard,
    TypedInferenceEngine,
)
from src.reasoning.operator_algebra import OperatorRelationAlgebra
from src.reasoning.relation_spectrum import (
    cycle_members,
    is_acyclic,
    katz_resolvent,
    spectral_radius,
)

__all__ = [
    "FuzzyInferenceEngine",
    "TypedInferenceEngine",
    "HallucinationGuard",
    "OperatorRelationAlgebra",
    "spectral_radius",
    "is_acyclic",
    "cycle_members",
    "katz_resolvent",
]

"""API pública para la segmentación de clientes."""

from .data import (
    DATA_PATH,
    PreparedData,
    load_customer_data,
    prepare_data,
    transform_new_data,
)
from .model import (
    ALGORITHMS,
    Algorithm,
    ExperimentResult,
    project_to_2d,
    select_best_model,
)

__all__ = [
    "ALGORITHMS",
    "DATA_PATH",
    "Algorithm",
    "ExperimentResult",
    "PreparedData",
    "load_customer_data",
    "prepare_data",
    "project_to_2d",
    "select_best_model",
    "transform_new_data",
]

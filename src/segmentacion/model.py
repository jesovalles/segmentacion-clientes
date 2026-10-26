"""Tipos y utilidades para crear, comparar y visualizar modelos de clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

Algorithm = Literal["K-Means", "MiniBatchKMeans", "Gaussian Mixture"]
# Algoritmos que puede crear `make_model` y seleccionar la interfaz.
ALGORITHMS: tuple[Algorithm, ...] = (
    "K-Means",
    "MiniBatchKMeans",
    "Gaussian Mixture",
)
# Tipos de estimador cuyo método `predict` conserva los clusters entrenados.
ClusteringModel = KMeans | MiniBatchKMeans | GaussianMixture


@dataclass
class ExperimentResult:
    """Métricas, etiquetas y estimador ajustado de un candidato de clustering.

    Las métricas pueden ser `None` si la muestra utilizada para evaluarlas no
    contiene todas las etiquetas producidas por el modelo.

    Atributos:
        algorithm: Nombre del algoritmo empleado.
        n_clusters: Cantidad solicitada de grupos o componentes.
        labels: Cluster asignado a cada fila de entrenamiento.
        model: Estimador ajustado, reutilizable para predecir clientes nuevos.
        silhouette: Cohesión/separación, entre -1 y 1; mayor suele ser mejor.
        davies_bouldin: Índice de compactación y separación; menor suele ser mejor.
        calinski_harabasz: Separación entre grupos frente a dispersión interna.
        fit_seconds: Segundos empleados en el ajuste.
        inertia: Inercia del modelo, si el estimador la calcula.
        cluster_sizes: Número de filas asignadas a cada etiqueta.
        error: Motivo por el cual no se pudieron calcular algunas métricas.
    """

    algorithm: Algorithm
    n_clusters: int
    labels: np.ndarray
    model: ClusteringModel
    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None
    fit_seconds: float
    inertia: float | None
    cluster_sizes: dict[int, int]
    error: str | None = None


def make_model(
    algorithm: Algorithm, n_clusters: int, random_state: int
) -> ClusteringModel:
    """Crea un estimador reproducible para el algoritmo solicitado.

    Args:
        algorithm: Nombre de uno de los algoritmos soportados.
        n_clusters: Cantidad de grupos o componentes que debe ajustar el modelo.
        random_state: Semilla para reproducir las inicializaciones aleatorias.

    Returns:
        Estimador de scikit-learn todavía no entrenado.

    Raises:
        ValueError: Si el algoritmo no es compatible con la unión de tipos.
    """
    if algorithm == "K-Means":
        return KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=random_state,
        )
    if algorithm == "MiniBatchKMeans":
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1024,
            n_init=10,
            random_state=random_state,
        )
    if algorithm == "Gaussian Mixture":
        return GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            n_init=2,
            reg_covar=1e-6,
            random_state=random_state,
        )
    raise ValueError(f"Algoritmo no soportado: {algorithm}")


def select_best_model(results: list[ExperimentResult]) -> ExperimentResult:
    """Selecciona el candidato mejor puntuado entre resultados evaluables.

    Prioriza Silhouette descendente; desempata por Davies-Bouldin ascendente y
    luego Calinski-Harabasz descendente. Estos indicadores no sustituyen la
    validación de utilidad e interpretabilidad con las áreas de negocio.

    Args:
        results: Resultados de experimentos que se desean comparar.

    Returns:
        Resultado seleccionado con métricas calculadas.

    Raises:
        ValueError: Si la lista está vacía o ningún modelo tiene métricas válidas.
    """
    eligible = [
        result
        for result in results
        if result.silhouette is not None
        and result.davies_bouldin is not None
        and result.calinski_harabasz is not None
    ]
    if not eligible:
        raise ValueError("Ningún modelo produjo métricas válidas para seleccionarlo.")
    return max(
        eligible,
        key=lambda result: (
            result.silhouette,
            -result.davies_bouldin,
            result.calinski_harabasz,
        ),
    )


def project_to_2d(scaled_data: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Proyecta características escaladas a dos dimensiones para visualizar.

    PCA se utiliza solamente para dibujar los clientes; la selección y ajuste
    del clustering se realizan con las características escaladas originales.
    Si existe una sola característica, la segunda coordenada será cero.

    Args:
        scaled_data: Matriz de clientes por características ya escalada.
        random_state: Semilla conservada para una interfaz reproducible.

    Returns:
        Matriz con dos coordenadas por cliente.

    Raises:
        ValueError: Si la matriz no tiene forma válida o carece de características.
    """
    values = np.asarray(scaled_data, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("La visualización requiere al menos una característica.")
    if values.shape[1] == 1:
        return np.column_stack((values[:, 0], np.zeros(len(values))))
    return PCA(n_components=2, random_state=random_state).fit_transform(values)

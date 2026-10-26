"""Entrenamiento, evaluación y asignación de segmentos a clientes."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

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
    make_model,
    select_best_model,
)


def run_experiments(
    scaled_data: np.ndarray,
    cluster_counts: range | list[int] | tuple[int, ...] = range(2, 9),
    algorithms: tuple[Algorithm, ...] | list[Algorithm] = ALGORITHMS,
    random_state: int = 42,
    sample_size: int = 2000,
) -> list[ExperimentResult]:
    """Ajusta combinaciones de algoritmos y `k`, y calcula métricas internas.

    Cada estimador se entrena con todas las filas. Para limitar el coste de
    Silhouette, Davies-Bouldin y Calinski-Harabasz, las métricas se calculan
    sobre una muestra aleatoria reproducible. Los candidatos cuya muestra no
    contenga todos sus segmentos se devuelven con métricas nulas y un motivo.

    Args:
        scaled_data: Matriz de entrenamiento numérica y ya estandarizada.
        cluster_counts: Valores de `k` o cantidad de componentes a comparar.
        algorithms: Algoritmos que se ajustarán para cada valor de `k`.
        random_state: Semilla común para reproducir modelos y muestra.
        sample_size: Máximo de filas usadas para calcular las métricas.

    Returns:
        Un resultado con modelo, etiquetas, medidas y tamaños por cada candidato.

    Raises:
        ValueError: Si los datos, algoritmos, cantidades de clusters o tamaño de
            muestra no cumplen las condiciones necesarias.
    """
    values = np.asarray(scaled_data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("Se necesitan al menos 3 filas y una característica.")
    if not np.isfinite(values).all():
        raise ValueError("Los datos escalados contienen valores no finitos.")
    if sample_size < 2:
        raise ValueError("El tamaño de muestra para las métricas debe ser al menos 2.")

    counts = tuple(dict.fromkeys(int(count) for count in cluster_counts))
    if not counts:
        raise ValueError("Selecciona al menos un número de clusters.")
    if any(count < 2 or count >= len(values) for count in counts):
        raise ValueError("Cada número de clusters debe estar entre 2 y filas - 1.")
    if not algorithms:
        raise ValueError("Selecciona al menos un algoritmo.")
    if any(algorithm not in ALGORITHMS for algorithm in algorithms):
        raise ValueError("Se solicitó un algoritmo que no está soportado.")

    sample_count = min(max(sample_size, max(counts) + 1), len(values))
    rng = np.random.default_rng(random_state)
    sample_indices = np.sort(rng.choice(len(values), size=sample_count, replace=False))
    metric_data = values[sample_indices]
    results: list[ExperimentResult] = []

    for algorithm in algorithms:
        for n_clusters in counts:
            model = make_model(algorithm, n_clusters, random_state)
            started = perf_counter()
            labels = model.fit_predict(values)
            fit_seconds = perf_counter() - started

            sampled_labels = labels[sample_indices]
            sizes = pd.Series(labels).value_counts().sort_index()
            full_label_set = set(np.unique(labels))
            sampled_label_set = set(np.unique(sampled_labels))
            if sampled_label_set != full_label_set:
                missing_labels = sorted(full_label_set - sampled_label_set)
                results.append(
                    ExperimentResult(
                        algorithm=algorithm,
                        n_clusters=n_clusters,
                        labels=labels,
                        model=model,
                        silhouette=None,
                        davies_bouldin=None,
                        calinski_harabasz=None,
                        fit_seconds=fit_seconds,
                        inertia=(
                            float(model.inertia_)
                            if hasattr(model, "inertia_")
                            else None
                        ),
                        cluster_sizes={
                            int(label): int(size) for label, size in sizes.items()
                        },
                        error=(
                            "La muestra no contiene todos los segmentos; "
                            f"faltan las etiquetas {missing_labels}."
                        ),
                    )
                )
                continue

            results.append(
                ExperimentResult(
                    algorithm=algorithm,
                    n_clusters=n_clusters,
                    labels=labels,
                    model=model,
                    silhouette=float(silhouette_score(metric_data, sampled_labels)),
                    davies_bouldin=float(
                        davies_bouldin_score(metric_data, sampled_labels)
                    ),
                    calinski_harabasz=float(
                        calinski_harabasz_score(metric_data, sampled_labels)
                    ),
                    fit_seconds=fit_seconds,
                    inertia=(
                        float(model.inertia_)
                        if hasattr(model, "inertia_")
                        else None
                    ),
                    cluster_sizes={
                        int(label): int(size) for label, size in sizes.items()
                    },
                )
            )
    return results


def results_to_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    """Convierte los resultados de experimentación en una tabla ordenada.

    Args:
        results: Resultados de modelos, incluidos los candidatos no evaluables.

    Returns:
        Tabla ordenada por Silhouette, Davies-Bouldin y Calinski-Harabasz,
        lista para mostrar en Streamlit o exportar a CSV.
    """
    return pd.DataFrame(
        [
            {
                "Algoritmo": result.algorithm,
                "Clusters": result.n_clusters,
                "Silhouette": result.silhouette,
                "Davies-Bouldin": result.davies_bouldin,
                "Calinski-Harabasz": result.calinski_harabasz,
                "Tiempo de ajuste (s)": result.fit_seconds,
                "Inercia": result.inertia,
                "Tamaño mínimo": min(result.cluster_sizes.values()),
                "Tamaño máximo": max(result.cluster_sizes.values()),
                "Estado": result.error or "Evaluado",
            }
            for result in results
        ]
    ).sort_values(
        ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
        ascending=[False, True, False],
        ignore_index=True,
        na_position="last",
    )


def name_segments(
    features: pd.DataFrame, labels: np.ndarray
) -> dict[int, str]:
    """Asigna etiquetas descriptivas usando los promedios de cada grupo.

    Cuando hay tres grupos y están disponibles `PURCHASES` y `CASH_ADVANCE`,
    distingue compradores frecuentes/de alto gasto, usuarios intensivos de
    avances de efectivo y compradores ocasionales. En cualquier otro caso
    conserva etiquetas genéricas para no inferir perfiles sin evidencia.

    Args:
        features: Características limpias, en las unidades originales.
        labels: Etiqueta de cluster correspondiente a cada fila de `features`.

    Returns:
        Diccionario de etiqueta numérica a nombre descriptivo.
    """
    cluster_ids = sorted(int(cluster_id) for cluster_id in np.unique(labels))
    names = {cluster_id: f"Segmento {cluster_id}" for cluster_id in cluster_ids}
    if len(cluster_ids) != 3:
        return names

    feature_names = {str(column).strip().upper(): column for column in features.columns}
    required = {"PURCHASES", "CASH_ADVANCE"}
    if not required.issubset(feature_names):
        return names

    profile = features.assign(Cluster=labels).groupby("Cluster").mean()
    purchases_column = feature_names["PURCHASES"]
    cash_column = feature_names["CASH_ADVANCE"]
    purchase_frequency_column = feature_names.get("PURCHASES_FREQUENCY")

    purchase_scores = profile[purchases_column]
    if purchase_frequency_column is not None:
        purchase_scores = purchase_scores.rank(pct=True) + profile[
            purchase_frequency_column
        ].rank(pct=True)
    high_purchase_id = int(purchase_scores.idxmax())

    remaining_ids = [cluster_id for cluster_id in cluster_ids if cluster_id != high_purchase_id]
    cash_advance_id = int(profile.loc[remaining_ids, cash_column].idxmax())
    occasional_id = next(
        cluster_id
        for cluster_id in cluster_ids
        if cluster_id not in {high_purchase_id, cash_advance_id}
    )

    names[high_purchase_id] = "Compradores frecuentes de alto gasto"
    names[cash_advance_id] = "Usuarios intensivos de avances de efectivo"
    names[occasional_id] = "Compradores ocasionales"
    return names


def assign_clusters(
    new_data: pd.DataFrame,
    training_data: PreparedData,
    experiment: ExperimentResult,
) -> pd.DataFrame:
    """Asigna clientes nuevos a los grupos de un modelo ya ajustado.

    Reutiliza las medianas y el escalador aprendidos en el entrenamiento y
    llama a `predict`; no vuelve a ajustar el modelo ni desplaza los segmentos.
    Conserva las columnas de entrada y añade la columna `Cluster`.

    Args:
        new_data: Registros por categorizar con las características requeridas.
        training_data: Esquema, medianas y escalador aprendidos al entrenar.
        experiment: Resultado cuyo estimador ya está ajustado.

    Returns:
        Copia de los registros nuevos con su etiqueta numérica de segmento.

    Raises:
        ValueError: Si los datos nuevos no contienen las características
            requeridas o no pueden transformarse.
    """
    _, scaled, _ = transform_new_data(new_data, training_data)
    assigned = new_data.copy()
    assigned["Cluster"] = experiment.model.predict(scaled)
    return assigned


def segment_customers(
    path: str | Path = DATA_PATH, n_clusters: int = 4
) -> pd.DataFrame:
    """Entrena K-Means desde un CSV y devuelve cada cliente con su cluster.

    Esta función es una utilidad de línea de comandos; para clasificar nuevos
    clientes con el mismo modelo, utiliza `assign_clusters` conservando el
    resultado del entrenamiento.

    Args:
        path: Ruta del CSV de entrenamiento.
        n_clusters: Número de segmentos que se ajustarán.

    Returns:
        Características limpias en unidades originales y columna `Cluster`.
    """
    data = load_customer_data(path)
    prepared = prepare_data(data)
    experiment = run_experiments(
        prepared.scaled,
        cluster_counts=[n_clusters],
        algorithms=["K-Means"],
    )[0]
    result = prepared.features.copy()
    result["Cluster"] = experiment.labels
    return result


def main() -> None:
    """Ejecuta una línea base de K-Means con cuatro grupos e imprime métricas."""
    data = load_customer_data()
    prepared = prepare_data(data)
    results = run_experiments(
        prepared.scaled,
        cluster_counts=[4],
        algorithms=["K-Means"],
    )
    best = select_best_model(results)
    print(results_to_frame([best]).to_string(index=False))


if __name__ == "__main__":
    main()

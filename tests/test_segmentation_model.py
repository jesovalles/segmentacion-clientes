"""Pruebas unitarias del pipeline, los modelos y la asignación de clientes."""

import numpy as np
import pandas as pd
import pytest

from segmentacion import (
    prepare_data,
    project_to_2d,
    select_best_model,
    transform_new_data,
)
from segmentacion.train import (
    assign_clusters,
    name_segments,
    results_to_frame,
    run_experiments,
    segment_customers,
)


def make_customer_data() -> pd.DataFrame:
    """Construye dos grupos sintéticos separables para pruebas reproducibles."""
    rng = np.random.default_rng(123)
    first = rng.normal(loc=(-3, -3), scale=0.35, size=(30, 2))
    second = rng.normal(loc=(3, 3), scale=0.35, size=(30, 2))
    return pd.DataFrame(
        {
            "CUST_ID": [f"C{i:03}" for i in range(60)],
            "BALANCE": np.concatenate([first[:, 0], second[:, 0]]),
            "PURCHASES": np.concatenate([first[:, 1], second[:, 1]]),
        }
    )


def test_prepare_data_excludes_id_imputes_and_does_not_mutate_input():
    data = make_customer_data()
    data.loc[0, "BALANCE"] = np.nan
    original = data.copy(deep=True)

    prepared = prepare_data(data)

    assert "CUST_ID" not in prepared.features.columns
    assert prepared.missing_filled == {"BALANCE": 1}
    assert prepared.features.isna().sum().sum() == 0
    assert prepared.scaled.shape == (60, 2)
    pd.testing.assert_frame_equal(data, original)


def test_prepare_data_drops_empty_feature_and_reports_it():
    data = make_customer_data()
    data["EMPTY"] = np.nan

    prepared = prepare_data(data)

    assert prepared.dropped_empty_columns == ("EMPTY",)
    assert "EMPTY" not in prepared.features


def test_transform_new_data_uses_training_columns_medians_and_scaler():
    training_data = make_customer_data()
    training_data.loc[0, "BALANCE"] = np.nan
    prepared = prepare_data(training_data)
    new_data = pd.DataFrame(
        {
            "CUST_ID": ["NEW-1"],
            "PURCHASES": [4.0],
            "BALANCE": [np.nan],
            "UNUSED": ["ignored"],
        }
    )

    features, scaled, missing_filled = transform_new_data(new_data, prepared)

    assert list(features.columns) == ["BALANCE", "PURCHASES"]
    assert features.loc[0, "BALANCE"] == prepared.medians["BALANCE"]
    assert scaled.shape == (1, 2)
    assert missing_filled == {"BALANCE": 1}


def test_transform_new_data_rejects_missing_training_features():
    prepared = prepare_data(make_customer_data())
    new_data = pd.DataFrame({"BALANCE": [1.0]})

    with pytest.raises(ValueError, match="Faltan características requeridas"):
        transform_new_data(new_data, prepared)


def test_assign_clusters_predicts_new_customers_with_fitted_model():
    training_data = make_customer_data()
    prepared = prepare_data(training_data)
    experiment = run_experiments(
        prepared.scaled,
        cluster_counts=[2],
        algorithms=["K-Means"],
        sample_size=60,
    )[0]
    new_customers = training_data.iloc[[0, -1]].copy()

    assigned = assign_clusters(new_customers, prepared, experiment)

    assert assigned["Cluster"].tolist() == experiment.model.predict(
        prepared.scaler.transform(new_customers[prepared.features.columns])
    ).tolist()
    assert assigned["CUST_ID"].tolist() == new_customers["CUST_ID"].tolist()
    assert assigned["Cluster"].nunique() == 2


def test_name_segments_maps_three_customer_profiles_to_behavioral_names():
    features = pd.DataFrame(
        {
            "PURCHASES": [4000, 500, 300],
            "PURCHASES_FREQUENCY": [0.95, 0.45, 0.23],
            "CASH_ADVANCE": [450, 339, 3900],
        }
    )
    labels = np.array([0, 1, 2])

    names = name_segments(features, labels)

    assert names == {
        0: "Compradores frecuentes de alto gasto",
        1: "Compradores ocasionales",
        2: "Usuarios intensivos de avances de efectivo",
    }


def test_name_segments_uses_generic_names_for_other_cluster_counts():
    features = pd.DataFrame({"PURCHASES": [10, 20], "CASH_ADVANCE": [1, 2]})

    names = name_segments(features, np.array([3, 7]))

    assert names == {3: "Segmento 3", 7: "Segmento 7"}


def test_segment_customers_returns_clustered_features_without_id(tmp_path):
    path = tmp_path / "customers.csv"
    make_customer_data().to_csv(path, index=False)

    clustered = segment_customers(path, n_clusters=2)

    assert "CUST_ID" not in clustered.columns
    assert "Cluster" in clustered.columns
    assert clustered["Cluster"].nunique() == 2


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame(),
        pd.DataFrame({"CUST_ID": ["A", "B"]}),
        pd.DataFrame({"VALUE": [np.inf, -np.inf]}),
    ],
)
def test_prepare_data_rejects_empty_or_unusable_data(data):
    with pytest.raises(ValueError):
        prepare_data(data)


def test_experiments_compare_algorithms_and_select_well_separated_clusters():
    prepared = prepare_data(make_customer_data())

    results = run_experiments(
        prepared.scaled,
        cluster_counts=[2, 3],
        algorithms=["K-Means", "MiniBatchKMeans", "Gaussian Mixture"],
        sample_size=60,
    )
    best = select_best_model(results)

    assert len(results) == 6
    assert best.n_clusters == 2
    assert best.silhouette > 0.8
    assert set(best.cluster_sizes.values()) == {30}
    assert results_to_frame(results).shape[0] == 6
    assert best.fit_seconds >= 0


def test_experiment_validation_rejects_invalid_cluster_count():
    prepared = prepare_data(make_customer_data())

    with pytest.raises(ValueError, match="entre 2 y filas"):
        run_experiments(prepared.scaled, cluster_counts=[60], algorithms=["K-Means"])


def test_selection_requires_results():
    with pytest.raises(ValueError, match="Ningún modelo produjo"):
        select_best_model([])


def test_experiment_keeps_candidate_when_metric_sample_misses_a_cluster():
    data = np.vstack((np.zeros((30, 1)), np.full((1, 1), 10.0)))

    results = run_experiments(
        data,
        cluster_counts=[2],
        algorithms=["K-Means"],
        random_state=42,
        sample_size=2,
    )

    assert len(results) == 1
    assert results[0].silhouette is None
    assert results[0].error is not None
    assert results_to_frame(results).loc[0, "Estado"] != "Evaluado"
    with pytest.raises(ValueError, match="Ningún modelo produjo"):
        select_best_model(results)


def test_pca_projection_returns_two_coordinates_per_row():
    prepared = prepare_data(make_customer_data())

    projection = project_to_2d(prepared.scaled)

    assert projection.shape == (60, 2)


def test_pca_projection_supports_a_single_feature():
    projection = project_to_2d(np.array([[0.0], [1.0], [2.0]]))

    assert projection.shape == (3, 2)
    assert np.array_equal(projection[:, 1], np.zeros(3))

"""Aplicación web para analizar y asignar segmentos de clientes.

El usuario puede cargar el conjunto base o usar el CSV local, comparar modelos
de clustering y clasificar clientes nuevos con el estimador seleccionado.
Streamlit conserva los resultados en la sesión; los archivos cargados no se
escriben en disco.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from segmentacion import (
    ALGORITHMS,
    DATA_PATH,
    prepare_data,
    project_to_2d,
    select_best_model,
)
from segmentacion.train import (
    assign_clusters,
    name_segments,
    results_to_frame,
    run_experiments,
)

DEFAULT_DATA_PATH = DATA_PATH

# Configuración general de la página y controles de entrada.
st.set_page_config(
    page_title="Segmentación de clientes",
    page_icon="👥",
    layout="wide",
)
st.title("Segmentación de clientes")
st.caption(
    "Explora el dataset, compara algoritmos de clustering y revisa las métricas "
    "antes de elegir una segmentación."
)

with st.sidebar:
    st.header("Datos")
    uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])
    st.caption("Los archivos se procesan en esta sesión y no se guardan en disco.")

if uploaded_file is not None:
    try:
        source_data = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        st.error(f"No se pudo leer el CSV: {error}")
        st.stop()
    data_label = uploaded_file.name
elif DEFAULT_DATA_PATH.exists():
    source_data = pd.read_csv(DEFAULT_DATA_PATH)
    data_label = DEFAULT_DATA_PATH.name
else:
    st.error(
        "No se encontró `Customer Data.csv`. Sube un CSV para comenzar o "
        "coloca el dataset en `data/Customer Data.csv`."
    )
    st.stop()

try:
    prepared = prepare_data(source_data)
except ValueError as error:
    st.error(str(error))
    st.stop()

if len(prepared.features) < 3:
    st.error("Se necesitan al menos 3 clientes para ejecutar una segmentación.")
    st.stop()

# Parámetros del experimento; las métricas se calculan sobre una muestra limitada.
with st.sidebar:
    st.header("Experimento")
    minimum_clusters, maximum_clusters = st.slider(
        "Rango de clusters (k)",
        min_value=2,
        max_value=min(12, len(prepared.features) - 1),
        value=(2, min(8, len(prepared.features) - 1)),
    )
    selected_algorithms = st.multiselect(
        "Algoritmos a comparar",
        options=list(ALGORITHMS),
        default=list(ALGORITHMS),
    )
    metric_sample_size = st.slider(
        "Filas para calcular métricas",
        min_value=min(500, len(prepared.features)),
        max_value=min(5000, len(prepared.features)),
        value=min(2000, len(prepared.features)),
        step=max(1, min(500, len(prepared.features))),
        help="Los modelos se ajustan con todas las filas; la puntuación Silhouette "
        "se estima en una muestra reproducible.",
    )
    run_clicked = st.button(
        "Comparar modelos",
        type="primary",
        use_container_width=True,
    )

signature = hashlib.sha256(
    pd.util.hash_pandas_object(source_data, index=True).values.tobytes()
    + repr(
        (
            minimum_clusters,
            maximum_clusters,
            selected_algorithms,
            metric_sample_size,
        )
    ).encode()
).hexdigest()

if run_clicked:
    st.session_state.pop("experiment_signature", None)
    st.session_state.pop("experiment_results", None)
    if not selected_algorithms:
        st.error("Selecciona al menos un algoritmo.")
    else:
        try:
            with st.spinner("Ajustando y evaluando modelos..."):
                experiments = run_experiments(
                    prepared.scaled,
                    cluster_counts=range(minimum_clusters, maximum_clusters + 1),
                    algorithms=selected_algorithms,
                    sample_size=metric_sample_size,
                )
                select_best_model(experiments)
            st.session_state["experiment_signature"] = signature
            st.session_state["experiment_results"] = experiments
        except ValueError as error:
            st.error(f"No se pudo completar la comparación: {error}")

# Resumen de calidad de datos y vista previa del conjunto cargado.
st.subheader(f"Dataset: {data_label}")
quality_left, quality_middle, quality_right = st.columns(3)
quality_left.metric("Clientes", f"{len(source_data):,}")
quality_middle.metric("Características usadas", len(prepared.features.columns))
quality_right.metric("Filas duplicadas", f"{prepared.duplicate_rows:,}")

if prepared.missing_filled:
    missing_table = pd.DataFrame(
        [
            {"Columna": column, "Valores imputados": count}
            for column, count in prepared.missing_filled.items()
        ]
    )
    st.info(
        f"Se imputaron valores faltantes con la mediana en "
        f"{len(prepared.missing_filled)} columna(s)."
    )
    with st.expander("Ver detalle de calidad de datos"):
        st.dataframe(missing_table, hide_index=True, use_container_width=True)
if prepared.dropped_empty_columns:
    st.warning(
        "Se excluyeron columnas sin valores utilizables: "
        + ", ".join(prepared.dropped_empty_columns)
    )

with st.expander("Vista previa de los datos"):
    st.dataframe(source_data.head(20), hide_index=True, use_container_width=True)

results = st.session_state.get("experiment_results")
if results and st.session_state.get("experiment_signature") == signature:
    # Presenta el mejor candidato, los perfiles y las herramientas de asignación.
    best = select_best_model(results)
    result_frame = results_to_frame(results)
    st.subheader("Comparación de modelos")
    st.caption(
        "El ranking prioriza Silhouette (más alto es mejor); en empate, "
        "Davies-Bouldin más bajo y Calinski-Harabasz más alto."
    )
    st.dataframe(
        result_frame.style.format(
            {
                "Silhouette": "{:.4f}",
                "Davies-Bouldin": "{:.4f}",
                "Calinski-Harabasz": "{:.2f}",
                "Tiempo de ajuste (s)": "{:.2f}",
                "Inercia": "{:.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    best_left, best_middle, best_right = st.columns(3)
    best_left.metric("Modelo seleccionado", best.algorithm)
    best_middle.metric("Número de clusters", best.n_clusters)
    best_right.metric("Silhouette", f"{best.silhouette:.4f}")
    st.caption(
        "Silhouette va de -1 a 1 y combina cohesión y separación. "
        "Es una guía para comparar, no una validación de negocio."
    )

    labeled_data = source_data.copy()
    labeled_data["Cluster"] = best.labels
    segment_names = name_segments(prepared.features, best.labels)
    labeled_data["Segmento"] = labeled_data["Cluster"].map(segment_names)
    profile_data = prepared.features.copy()
    profile_data["Cluster"] = best.labels
    pca_coordinates = project_to_2d(prepared.scaled)
    chart_data = pd.DataFrame(
        {
            "PCA1": pca_coordinates[:, 0],
            "PCA2": pca_coordinates[:, 1],
            "Segmento": pd.Series(best.labels).map(segment_names).to_numpy(),
        }
    )

    chart_tab, profiles_tab, export_tab, classification_tab = st.tabs(
        [
            "Visualización",
            "Perfil de segmentos",
            "Descargas",
            "Clasificar nuevos clientes",
        ]
    )
    with chart_tab:
        st.markdown(
            "Proyección PCA en dos dimensiones; se utiliza solo para visualizar, "
            "no para ajustar ni seleccionar el modelo."
        )
        st.scatter_chart(
            chart_data,
            x="PCA1",
            y="PCA2",
            color="Segmento",
            use_container_width=True,
        )
        segment_counts = (
            labeled_data.groupby(["Cluster", "Segmento"])
            .size()
            .rename("Clientes")
            .reset_index()
            .sort_values("Cluster")
            .set_index("Segmento")["Clientes"]
        )
        st.bar_chart(segment_counts)
    with profiles_tab:
        profile = profile_data.groupby("Cluster").mean()
        profile.insert(0, "Nombre del segmento", profile.index.map(segment_names))
        numeric_profile_columns = [
            column for column in profile.columns if column != "Nombre del segmento"
        ]
        st.dataframe(
            profile.style.format(
                {column: "{:,.2f}" for column in numeric_profile_columns}
            ),
            use_container_width=True,
        )
        st.caption(
            "Los promedios están expresados en las unidades originales. "
            "Los nombres se asignan según el perfil promedio de cada grupo."
        )
    with export_tab:
        st.download_button(
            "Descargar clientes con segmento (CSV)",
            data=labeled_data.to_csv(index=False).encode("utf-8"),
            file_name="clientes_segmentados.csv",
            mime="text/csv",
        )
        st.download_button(
            "Descargar métricas de modelos (CSV)",
            data=result_frame.to_csv(index=False).encode("utf-8"),
            file_name="metricas_modelos.csv",
            mime="text/csv",
        )
    with classification_tab:
        st.markdown(
            "El modelo seleccionado ya está entrenado. Carga clientes nuevos o "
            "completa un registro individual para asignarlos a uno de esos segmentos."
        )
        classification_file = st.file_uploader(
            "Subir CSV de clientes nuevos",
            type=["csv"],
            key="new_customers_csv",
            help=(
                "Incluye las mismas características usadas para entrenar. "
                "CUST_ID es opcional."
            ),
        )
        classify_file_clicked = st.button(
            "Categorizar archivo",
            key="classify_new_file",
            disabled=classification_file is None,
        )
        if classify_file_clicked and classification_file is not None:
            try:
                new_customers = pd.read_csv(classification_file)
                assigned_customers = assign_clusters(new_customers, prepared, best)
                assigned_customers["Segmento"] = assigned_customers["Cluster"].map(
                    segment_names
                )
                st.session_state["assigned_customers"] = assigned_customers
                st.session_state["assigned_customers_signature"] = (
                    signature,
                    best.algorithm,
                    best.n_clusters,
                    hashlib.sha256(classification_file.getvalue()).hexdigest(),
                )
                st.session_state.pop("individual_assignment", None)
                st.success(
                    f"Se categorizaron {len(assigned_customers):,} clientes con "
                    f"{best.algorithm} ({best.n_clusters} segmentos)."
                )
            except (
                UnicodeDecodeError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
                ValueError,
            ) as error:
                st.error(f"No se pudo categorizar el archivo: {error}")

        assignment_signature = (
            signature,
            best.algorithm,
            best.n_clusters,
        )
        saved_file_signature = st.session_state.get(
            "assigned_customers_signature", ()
        )[:3]
        current_file_hash = (
            hashlib.sha256(classification_file.getvalue()).hexdigest()
            if classification_file is not None
            else None
        )
        if (
            st.session_state.get("assigned_customers") is not None
            and saved_file_signature == assignment_signature
            and st.session_state["assigned_customers_signature"][3]
            == current_file_hash
        ):
            assigned_customers = st.session_state["assigned_customers"]
            st.dataframe(
                assigned_customers.head(100),
                hide_index=True,
                use_container_width=True,
            )
            counts = assigned_customers["Cluster"].value_counts().sort_index()
            st.write("Clientes por segmento")
            st.dataframe(
                counts.rename_axis("Cluster").reset_index(name="Clientes").assign(
                    Segmento=lambda frame: frame["Cluster"].map(segment_names)
                )[["Cluster", "Segmento", "Clientes"]],
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                "Descargar clientes categorizados",
                data=assigned_customers.to_csv(index=False).encode("utf-8"),
                file_name="clientes_categorizados.csv",
                mime="text/csv",
                key="download_assigned_customers",
            )

        st.divider()
        st.markdown("#### Categorizar un cliente individual")
        st.caption(
            "Los campos empiezan con la mediana del dataset de entrenamiento; "
            "edítalos con los valores reales del cliente."
        )
        with st.form("individual_customer_form"):
            customer_id = st.text_input("Identificador (opcional)")
            form_columns = st.columns(3)
            individual_values: dict[str, float] = {}
            for index, column in enumerate(prepared.features.columns):
                individual_values[column] = form_columns[index % 3].number_input(
                    str(column),
                    value=float(prepared.medians[column]),
                    format="%.6f",
                    key=f"individual_{column}",
                )
            categorize_individual = st.form_submit_button(
                "Categorizar cliente",
                type="primary",
            )

        if categorize_individual:
            individual_customer = pd.DataFrame([individual_values])
            if customer_id.strip():
                individual_customer.insert(0, "CUST_ID", customer_id.strip())
            try:
                assigned_individual = assign_clusters(
                    individual_customer,
                    prepared,
                    best,
                )
                assigned_individual["Segmento"] = assigned_individual[
                    "Cluster"
                ].map(segment_names)
                st.session_state["individual_assignment"] = (
                    assignment_signature,
                    assigned_individual,
                )
            except ValueError as error:
                st.error(f"No se pudo categorizar el cliente: {error}")

        individual_result = st.session_state.get("individual_assignment")
        if individual_result and individual_result[0] == assignment_signature:
            individual_customer = individual_result[1]
            cluster = int(individual_customer["Cluster"].iloc[0])
            segment_name = individual_customer["Segmento"].iloc[0]
            st.success(f"Cliente asignado al segmento {cluster}: {segment_name}.")
            st.dataframe(
                individual_customer,
                hide_index=True,
                use_container_width=True,
            )
elif not run_clicked:
    st.info("Ajusta los parámetros y pulsa «Comparar modelos» para iniciar el análisis.")

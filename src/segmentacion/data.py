"""Carga y preparación de datos de clientes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Customer Data.csv"
# El CSV de ejemplo es local y está excluido del repositorio por privacidad.
ID_COLUMN_NAMES = {"CUST_ID", "CUSTOMER_ID", "CUSTOMERID", "ID"}
# Nombres normalizados de columnas que no deben convertirse en características.


@dataclass
class PreparedData:
    """Datos preparados y estadísticas aprendidas exclusivamente del entrenamiento.

    Attributes:
        features: Características numéricas imputadas en sus unidades originales.
        scaled: Matriz de `features` transformada con `scaler`.
        scaler: Escalador ajustado al conjunto de entrenamiento.
        medians: Medianas de entrenamiento reutilizables para imputar nuevos clientes.
        missing_filled: Cantidad de valores ausentes o no finitos detectados por columna.
        dropped_empty_columns: Columnas sin valores utilizables que fueron excluidas.
        duplicate_rows: Cantidad de filas duplicadas en los datos de entrada.
    """

    features: pd.DataFrame
    scaled: np.ndarray
    scaler: StandardScaler
    medians: pd.Series
    missing_filled: dict[str, int]
    dropped_empty_columns: tuple[str, ...]
    duplicate_rows: int


def load_customer_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Carga un CSV de clientes.

    Args:
        path: Ruta del archivo CSV. Por defecto, usa el dataset local del proyecto.

    Returns:
        DataFrame con las columnas originales del archivo.

    Raises:
        FileNotFoundError: Si la ruta no existe.
        pandas.errors.ParserError: Si el contenido no tiene un formato CSV válido.
    """
    return pd.read_csv(path)


def prepare_data(data: pd.DataFrame) -> PreparedData:
    """Valida, imputa y escala las características del conjunto de entrenamiento.

    Las columnas identificadoras conocidas se excluyen del modelo. Los valores
    no numéricos y no finitos se consideran ausentes; las columnas completamente
    vacías se descartan y el resto se imputa con su mediana. Después se ajusta un
    `StandardScaler` que debe reutilizarse para clasificar clientes futuros.

    Args:
        data: Registros de entrenamiento, conservados sin modificaciones.

    Returns:
        Características originales limpias, matriz escalada y estadísticas de
        transformación necesarias para aplicar el mismo preprocesamiento luego.

    Raises:
        ValueError: Si no hay filas, características utilizables o columnas únicas.
    """
    if data.empty:
        raise ValueError("El archivo no contiene filas.")
    if data.columns.duplicated().any():
        raise ValueError("El archivo contiene nombres de columna duplicados.")

    feature_columns = [
        column
        for column in data.columns
        if str(column).strip().upper() not in ID_COLUMN_NAMES
    ]
    if not feature_columns:
        raise ValueError("No hay columnas de características aparte del identificador.")

    numeric = data.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    missing_filled = {
        str(column): int(count)
        for column, count in numeric.isna().sum().items()
        if count > 0
    }
    dropped_empty = tuple(
        str(column) for column in numeric.columns if numeric[column].isna().all()
    )
    numeric = numeric.drop(columns=list(dropped_empty))
    if numeric.empty:
        raise ValueError(
            "No se encontraron características numéricas con valores utilizables."
        )

    numeric = numeric.fillna(numeric.median())
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Los datos conservan valores no finitos tras la limpieza.")

    medians = numeric.median()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)
    return PreparedData(
        features=numeric,
        scaled=scaled,
        scaler=scaler,
        medians=medians,
        missing_filled=missing_filled,
        dropped_empty_columns=dropped_empty,
        duplicate_rows=int(data.duplicated().sum()),
    )


def transform_new_data(
    data: pd.DataFrame, training_data: PreparedData
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """Prepara clientes nuevos con el esquema y estadísticas del entrenamiento.

    No vuelve a calcular medianas ni ajusta el escalador, lo que evita fuga de
    información y mantiene comparables las distancias respecto a los clusters.
    Se permiten columnas adicionales (por ejemplo, un identificador), pero todas
    las características de entrenamiento deben estar presentes.

    Args:
        data: Filas nuevas que se desean clasificar.
        training_data: Resultado de `prepare_data` aplicado al entrenamiento.

    Returns:
        Tupla con características imputadas, matriz escalada y cantidad de
        valores que debieron imputarse en las filas nuevas.

    Raises:
        ValueError: Si el archivo está vacío, tiene columnas duplicadas, faltan
            características requeridas o no hay valores numéricos utilizables.
    """
    if data.empty:
        raise ValueError("El archivo de clientes nuevos no contiene filas.")
    if data.columns.duplicated().any():
        raise ValueError("El archivo contiene nombres de columna duplicados.")

    feature_columns = list(training_data.features.columns)
    missing_columns = [
        column for column in feature_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Faltan características requeridas: " + ", ".join(map(str, missing_columns))
        )

    numeric = data.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    missing_filled = {
        str(column): int(count)
        for column, count in numeric.isna().sum().items()
        if count > 0
    }
    numeric = numeric.fillna(training_data.medians)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Los nuevos datos contienen valores no numéricos utilizables.")

    return numeric, training_data.scaler.transform(numeric), missing_filled

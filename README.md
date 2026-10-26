# Segmentación de clientes

Aplicación de ciencia de datos para explorar perfiles de clientes, comparar
algoritmos de clustering y asignar nuevos registros a segmentos existentes. La
interfaz está construida con Streamlit y el pipeline de análisis se organiza
como un paquete Python instalable.

> **Privacidad:** el dataset contiene información financiera de clientes y se
> excluye del control de versiones. No publiques archivos reales ni datos
> personales. Para probar la app localmente, coloca el CSV autorizado en
> `data/Customer Data.csv` o cárgalo desde la interfaz.

## Contenido

- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos y ejecución local](#requisitos-y-ejecución-local)
- [Flujo de uso de la aplicación](#flujo-de-uso-de-la-aplicación)
- [Formato de los datos](#formato-de-los-datos)
- [Cómo se prepara y evalúa el modelo](#cómo-se-prepara-y-evalúa-el-modelo)
- [Módulos y funciones](#módulos-y-funciones)
- [Pruebas](#pruebas)
- [Despliegue en Streamlit Community Cloud](#despliegue-en-streamlit-community-cloud)
- [Limitaciones y consideraciones](#limitaciones-y-consideraciones)

## Estructura del repositorio

```text
.
├── .streamlit/
│   └── config.toml                  # Apariencia y opciones de Streamlit
├── data/
│   └── README.md                    # Instrucciones para datos locales (CSV ignorados)
├── eda/
│   └── Segmentación de clientes.ipynb # Análisis exploratorio original
├── src/
│   └── segmentacion/
│       ├── __init__.py              # API pública del paquete
│       ├── data.py                  # Carga, limpieza y transformación
│       ├── model.py                 # Estimadores y selección
│       └── train.py                 # Experimentos, perfiles y predicciones
├── tests/
│   └── test_segmentation_model.py   # Pruebas unitarias y de integración
├── app.py                           # Aplicación web
├── pyproject.toml                   # Configuración e instalación src-layout
├── requirements.txt                 # Dependencias de ejecución
└── requirements-dev.txt             # Dependencias de desarrollo y pruebas
```

## Requisitos y ejecución local

Se requiere Python 3.10 o posterior. Desde PowerShell en Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
streamlit run app.py
```

En macOS o Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
streamlit run app.py
```

Abre la dirección local que indique Streamlit, normalmente
<http://localhost:8501>. Para ejecutar el análisis base desde la terminal:

```bash
python -m segmentacion.train
```

El paquete se instala desde `src/` mediante `pyproject.toml`; `requirements.txt`
incluye esa instalación local para que el mismo comando funcione en despliegue.

## Flujo de uso de la aplicación

1. **Cargar datos:** la app usa `data/Customer Data.csv` si existe. También
   permite cargar un CSV desde la barra lateral.
2. **Configurar el experimento:** seleccionar rango de clusters, algoritmos y
   tamaño de muestra para las métricas.
3. **Comparar modelos:** pulsar **Comparar modelos**. La app ordena candidatos y
   muestra métricas, cantidad de clientes y perfiles por cluster.
4. **Interpretar los grupos:** revisar **Visualización** y **Perfil de
   segmentos**. Para tres clusters con las columnas necesarias, se muestran
   nombres como compradores frecuentes de alto gasto.
5. **Clasificar clientes nuevos:** en **Clasificar nuevos clientes**, cargar un
   CSV con las características de entrenamiento o completar el formulario de
   un cliente individual.
6. **Exportar:** descargar asignaciones y métricas en formato CSV.

La clasificación nueva reutiliza el modelo, las medianas y el escalador del
entrenamiento. No vuelve a ajustar el modelo, por lo que no cambia la definición
de los segmentos durante una asignación.

## Formato de los datos

El conjunto de entrenamiento necesita al menos tres filas y una o más
características numéricas. Se pueden subir datos numéricos o valores que pandas
pueda convertir a números. Los identificadores conocidos (`CUST_ID`,
`CUSTOMER_ID`, `CUSTOMERID` e `ID`) no se usan como características.

Para el dataset de ejemplo, las características son:

```text
BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX, CREDIT_LIMIT,
PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT, TENURE
```

El CSV de clasificación debe incluir **todas las características usadas al
entrenar**, aunque puede contener columnas adicionales e identificadores. La
app conserva dichas columnas de entrada y agrega:

- `Cluster`: etiqueta numérica del modelo.
- `Segmento`: nombre descriptivo, cuando el modelo y las columnas disponibles
  permiten inferirlo.

Los datos nuevos con valores ausentes se imputan usando las medianas aprendidas
del entrenamiento. No se usa la mediana del nuevo archivo para evitar cambiar
la referencia del modelo.

## Cómo se prepara y evalúa el modelo

### Preparación

- Excluye columnas de identificación conocidas.
- Convierte las características a números; valores inválidos e infinitos se
  tratan como ausentes.
- Descarta columnas completamente vacías y registra la calidad de los datos.
- Imputa los valores ausentes restantes con la mediana.
- Ajusta `StandardScaler` sobre el conjunto de entrenamiento.
- Al clasificar nuevos clientes, reutiliza las medianas y el escalador ya
  ajustados.

### Algoritmos disponibles

| Algoritmo | Descripción |
|---|---|
| K-Means | Agrupa por distancia a centroides; es la referencia principal. |
| MiniBatchKMeans | Variante por lotes pequeños, útil para conjuntos mayores. |
| Gaussian Mixture | Modelo probabilístico de mezcla con covarianza diagonal. |

### Métricas

- **Silhouette:** más alto indica mejor cohesión y separación; se prioriza para
  ordenar modelos.
- **Davies-Bouldin:** más bajo suele indicar grupos más compactos y separados.
- **Calinski-Harabasz:** más alto suele indicar mejor separación entre grupos.
- **Inercia:** suma de distancias cuadráticas a centroides; solo está disponible
  para modelos K-Means.
- **Tiempo de ajuste** y tamaños mínimo/máximo de cluster: ayudan a comparar
  coste y desbalance entre grupos.

Los modelos se ajustan con todas las filas. Para reducir el coste de las
métricas, la app calcula las puntuaciones de clustering en una muestra
reproducible cuyo tamaño se puede cambiar desde la barra lateral. La selección
automática ordena por Silhouette descendente, Davies-Bouldin ascendente y
Calinski-Harabasz descendente.

### Etiquetas descriptivas

Cuando hay exactamente tres grupos y se encuentran `PURCHASES` y
`CASH_ADVANCE`, se asignan nombres según sus promedios:

- **Compradores frecuentes de alto gasto:** mayor puntuación de gasto y
  frecuencia de compras.
- **Usuarios intensivos de avances de efectivo:** mayor promedio de adelantos
  entre los dos grupos restantes.
- **Compradores ocasionales:** grupo restante de menor actividad relativa.

Los nombres se muestran junto al índice numérico del cluster. Si el número de
grupos o las columnas no permiten justificar esos perfiles, se usan etiquetas
genéricas (`Segmento 0`, `Segmento 1`, etc.). Los nombres describen patrones,
no son categorías de riesgo ni una recomendación de crédito.

## Módulos y funciones

### `segmentacion.data`

| Función o tipo | Responsabilidad |
|---|---|
| `load_customer_data(path)` | Lee un CSV y devuelve sus filas en un DataFrame. |
| `prepare_data(data)` | Limpia los datos de entrenamiento, imputa y ajusta el escalador. |
| `transform_new_data(data, training_data)` | Alinea y escala datos nuevos con la referencia aprendida. |
| `PreparedData` | Conserva características, escalador, medianas y estadísticas de calidad. |

### `segmentacion.model`

| Función o tipo | Responsabilidad |
|---|---|
| `make_model(algorithm, n_clusters, random_state)` | Crea un estimador reproducible sin entrenarlo. |
| `select_best_model(results)` | Selecciona el resultado mejor puntuado que tenga métricas válidas. |
| `project_to_2d(scaled_data)` | Proyecta características a dos dimensiones para visualización con PCA. |
| `ExperimentResult` | Reúne modelo, predicciones, métricas, duración y tamaños de cluster. |

### `segmentacion.train`

| Función | Responsabilidad |
|---|---|
| `run_experiments(...)` | Ajusta combinaciones de algoritmo y número de grupos y calcula métricas. |
| `results_to_frame(results)` | Convierte resultados a una tabla ordenada para mostrar o exportar. |
| `name_segments(features, labels)` | Asigna nombres interpretables cuando los perfiles lo permiten. |
| `assign_clusters(new_data, training_data, experiment)` | Predice grupos nuevos sin volver a entrenar. |
| `segment_customers(path, n_clusters)` | Atajo de terminal para entrenar K-Means desde un CSV. |

Las funciones públicas y los módulos incluyen docstrings en español con
parámetros, resultados y errores esperados.

## Pruebas

Ejecuta desde la raíz:

```bash
pytest -q
```

Las pruebas cubren limpieza e imputación, validación del esquema, comparación
entre algoritmos, selección de modelos, asignación de nuevos clientes,
nomenclatura de perfiles y proyección PCA.

## Despliegue en Streamlit Community Cloud

1. Publica en GitHub el código y la configuración; el CSV local está excluido
   deliberadamente.
2. En Streamlit Community Cloud, crea una aplicación y selecciona `app.py`
   como archivo principal.
3. La plataforma instalará las dependencias indicadas en `requirements.txt`.
4. Una vez desplegada, carga el dataset de entrenamiento desde la app. Luego
   podrás clasificar otros clientes con ese modelo durante la sesión.

No se requieren secretos ni servicios externos. Los CSV cargados se procesan en
memoria de la sesión y no se guardan en el repositorio ni en disco por la app.

## Limitaciones y consideraciones

- Clustering no tiene etiquetas objetivo: las métricas son criterios internos,
  no demuestran que los grupos sean útiles para una decisión comercial.
- Silhouette puede variar según la muestra; contrasta modelos con varios
  tamaños de muestra y valida estabilidad antes de tomar decisiones.
- K-Means y la distancia euclidiana favorecen grupos aproximadamente compactos;
  pueden ser sensibles a valores extremos y distribuciones sesgadas.
- Los números de cluster son identificadores, no una escala ordinal; pueden
  cambiar si se reentrena el modelo.
- La proyección PCA solo representa visualmente los datos y puede ocultar
  diferencias en dimensiones que no aparecen en el gráfico.
- Los nombres sugeridos son descripciones estadísticas. No deben utilizarse
  como etiquetas de riesgo o para decisiones de crédito sin validación,
  revisión de sesgos y controles de cumplimiento.

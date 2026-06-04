# Databricks notebook source
# MAGIC %md
# MAGIC # Sesión 9 — Notebook Estudiante — Batch Inference, MLOps básico y monitoreo
# MAGIC ## Del modelo champion a predicciones batch monitoreables
# MAGIC
# MAGIC Este notebook está **blindado y parametrizado** para reutilizarlo con otros modelos, versiones, runs y tablas de scoring.
# MAGIC
# MAGIC La idea central de la sesión:
# MAGIC
# MAGIC ```text
# MAGIC Modelo entrenado → modelo registrado → inferencia batch → predicciones en Delta → monitoreo → alertas
# MAGIC ```
# MAGIC
# MAGIC > Nota: si `models:/` falla por problemas de descarga de artefactos del Model Registry, el notebook intentará cargar desde `version.source`, `runs:/` y runs activos del experimento configurado.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cómo reutilizar este notebook en otro caso
# MAGIC
# MAGIC Cambiar únicamente los parámetros del bloque 0:
# MAGIC
# MAGIC - `MODEL_NAME`: nombre completo del modelo registrado.
# MAGIC - `MODEL_VERSION_PREFERRED`: versión específica o vacío para última versión.
# MAGIC - `EXPERIMENT_NAME`: experimento activo para fallback de runs.
# MAGIC - `FEATURE_TABLE`: tabla Delta de scoring.
# MAGIC - `TARGET_CANDIDATES`: nombres posibles del valor real.
# MAGIC - `TRACE_COLUMNS`: columnas de trazabilidad.
# MAGIC - `CATEGORICAL_COLUMNS`: categóricas que requieren one-hot encoding.
# MAGIC - `MANUAL_EXPECTED_FEATURE_COLUMNS`: contrato manual si el modelo no tiene signature.
# MAGIC
# MAGIC No modifiques Gold ni reentrenes el modelo en esta sesión.

# COMMAND ----------

# ============================================================
# 0. Configuración general PARAMETRIZABLE
# ============================================================
# Nota para reutilizar este notebook en otros modelos:
# 1) Cambia MODEL_NAME por el modelo registrado que quieras usar.
# 2) Cambia FEATURE_TABLE por la tabla Delta que contiene los datos de scoring.
# 3) Cambia TARGET_CANDIDATES si tu variable real tiene otro nombre.
# 4) Cambia TRACE_COLUMNS para conservar las columnas de trazabilidad de tu negocio.
# 5) Cambia CATEGORICAL_COLUMNS y KNOWN_CATEGORIES si tienes variables categóricas.
# 6) Si el modelo no tiene signature, define MANUAL_EXPECTED_FEATURE_COLUMNS.
#
# El notebook intenta cargar modelos en este orden:
# - models:/<modelo>/<version>                         (Registry / gobierno)
# - source real de la versión registrada, normalmente dbfs:/.../artifacts/model
# - runs:/<run_id>/model                               (run activo)
# - artifact_uri/model de runs activos encontrados      (fallback académico)
#
# Esto blinda el caso visto en clase: el modelo existe en Catalog Explorer,
# pero models:/ puede fallar al descargar artefactos desde Unity Catalog storage.

import datetime as dt
import json
import math
import os
import re
import traceback
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from pyspark.sql import functions as F
from pyspark.sql import types as T
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from mlflow.tracking import MlflowClient
except Exception:
    MlflowClient = None

# ------------------------------------------------------------
# Widgets Databricks opcionales
# ------------------------------------------------------------
# En clase puedes dejarlos como vienen. Para reutilizar el notebook, cambia los widgets
# sin tocar el resto del código.

def _current_user_for_default():
    try:
        return spark.sql("SELECT current_user() AS user").collect()[0]["user"]
    except Exception:
        return "<usuario>"

_CURRENT_USER = _current_user_for_default()
DEFAULT_EXPERIMENT_NAME = f"/Users/{_CURRENT_USER}/sesion_08_bagazo_mlflow"

try:
    dbutils.widgets.text("CATALOG", "workspace", "01 Catalog")
    dbutils.widgets.text("FEATURE_TABLE", "workspace.ml_features.bagazo_features_training", "02 Feature/scoring table")
    dbutils.widgets.text("MODEL_NAME", "workspace.ml_models.bagazo_champion_sesion_08", "03 Registered model name")
    dbutils.widgets.text("MODEL_VERSION_PREFERRED", "", "04 Preferred model version (blank = latest)")
    dbutils.widgets.text("EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME, "05 Experiment name fallback")
    dbutils.widgets.text("MODEL_ARTIFACT_PATH", "model", "06 Model artifact path")
    dbutils.widgets.text("SCORING_LIMIT", "90", "07 Scoring rows")
    dbutils.widgets.text("ORDER_BY_COLUMN", "fecha", "08 Order by column")
    dbutils.widgets.text("SCENARIO_HINT", "B_forecast_sin_leakage__random_forest", "09 Scenario/run hint")
    dbutils.widgets.text("MODEL_SELECTION_METRIC", "r2", "10 Metric for active run fallback")
    dbutils.widgets.dropdown("ALLOW_ACTIVE_RUN_FALLBACK", "true", ["true", "false"], "11 Allow active run fallback")
except Exception:
    pass


def _widget_or_default(name, default):
    try:
        value = dbutils.widgets.get(name)
        return value if value is not None else default
    except Exception:
        return default

CATALOG = _widget_or_default("CATALOG", "workspace")
FEATURE_TABLE = _widget_or_default("FEATURE_TABLE", "workspace.ml_features.bagazo_features_training")
MODEL_NAME = _widget_or_default("MODEL_NAME", "workspace.ml_models.bagazo_champion_sesion_08")
MODEL_VERSION_PREFERRED = _widget_or_default("MODEL_VERSION_PREFERRED", "").strip() or None
EXPERIMENT_NAME = _widget_or_default("EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
MODEL_ARTIFACT_PATH = _widget_or_default("MODEL_ARTIFACT_PATH", "model")
SCORING_LIMIT = int(_widget_or_default("SCORING_LIMIT", "90"))
ORDER_BY_COLUMN = _widget_or_default("ORDER_BY_COLUMN", "fecha")
SCENARIO_HINT = _widget_or_default("SCENARIO_HINT", "B_forecast_sin_leakage__random_forest")
MODEL_SELECTION_METRIC = _widget_or_default("MODEL_SELECTION_METRIC", "r2").strip() or "r2"
ALLOW_ACTIVE_RUN_FALLBACK = _widget_or_default("ALLOW_ACTIVE_RUN_FALLBACK", "true").lower() == "true"

# Schemas y tablas de salida. Cambia el sufijo si vas a ejecutar varias pruebas.
SCHEMA_SERVING = "ml_serving"
SCHEMA_MONITORING = "ml_monitoring"
SCORING_INPUT_TABLE = f"{CATALOG}.{SCHEMA_SERVING}.bagazo_batch_scoring_input_sesion_09"
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA_SERVING}.bagazo_batch_predictions_sesion_09"
PERFORMANCE_TABLE = f"{CATALOG}.{SCHEMA_MONITORING}.bagazo_model_performance_sesion_09"
PERFORMANCE_BY_GROUP_TABLE = f"{CATALOG}.{SCHEMA_MONITORING}.bagazo_model_performance_by_group_sesion_09"
ALERTS_TABLE = f"{CATALOG}.{SCHEMA_MONITORING}.bagazo_alert_rules_sesion_09"
LINEAGE_TABLE = f"{CATALOG}.{SCHEMA_MONITORING}.bagazo_model_lineage_sesion_09"

# Columnas de negocio para conservar trazabilidad.
TRACE_COLUMNS = ["fecha", "ingenio"]
GROUP_COLUMN = "ingenio"

# Candidatas de variable real/target. Para otro caso, agrega aquí el nombre real.
TARGET_CANDIDATES = [
    "target_bagazo_entregado_ton",
    "bagazo_entregado_ton",
    "bagazo_entregado_toneladas",
    "bagazo_real",
    "target",
    "y",
]

# Variables categóricas que se codifican con one-hot.
# Para otro modelo: por ejemplo ["segmento", "region"].
CATEGORICAL_COLUMNS = ["ingenio"]
KNOWN_CATEGORIES = {
    "ingenio": ["ILC", "Incauca", "Providencia"]
}

# Si el modelo NO trae signature, puedes forzar aquí el contrato exacto de columnas.
# Recomendado para notebooks reutilizables cuando el modelo fue guardado sin input_example/signature.
MANUAL_EXPECTED_FEATURE_COLUMNS = []

# Control avanzado para predicción robusta.
# Mantén true en clase: si MLflow pyfunc falla por signature/dtypes, intenta el modelo sklearn directo.
ENABLE_SKLEARN_DIRECT_PREDICTION_FALLBACK = True

# Cuando el modelo fue entrenado con scikit-learn, feature_names_in_ suele ser
# el contrato más fiel porque corresponde exactamente a las columnas vistas por fit().
# Prioridad recomendada: manual > sklearn.feature_names_in_ > MLflow signature > auto.
PREFER_SKLEARN_FEATURE_NAMES_IN = True


# Si quieres forzar columnas raw antes del encoding, llena esta lista.
# Si está vacía, se calculan automáticamente excluyendo targets y trazabilidad.
MANUAL_RAW_FEATURE_COLUMNS = []

# Columnas que nunca deben entrar al modelo.
EXTRA_DROP_COLUMNS = [
    "batch_id", "fecha_inferencia", "modelo_nombre", "modelo_version", "run_id", "model_uri",
    "bagazo_predicho", "prediction", "prediccion", "error", "error_absoluto", "error_porcentual",
    "target_riesgo_bajo_bagazo"
]

# Umbrales académicos para monitoreo. En producción deben venir de negocio.
ERROR_ALTO_TON = 150.0
THRESHOLD_MAE = 120.0
THRESHOLD_RMSE = 180.0
THRESHOLD_SMAPE = 90.0
THRESHOLD_ERROR_ALTO_PCT = 25.0
MIN_EXPECTED_ROWS = max(10, min(30, SCORING_LIMIT))

BATCH_ID = f"batch_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

try:
    mlflow.set_tracking_uri("databricks")
except Exception as e:
    print("No fue posible fijar tracking URI databricks:", str(e)[:300])

# Para modelos en Unity Catalog suele ser necesario databricks-uc.
try:
    mlflow.set_registry_uri("databricks-uc")
except Exception as e:
    print("No fue posible fijar registry URI databricks-uc; se continuará con el default:", str(e)[:300])

print("Batch ID:", BATCH_ID)
print("Feature table:", FEATURE_TABLE)
print("Modelo registrado:", MODEL_NAME)
print("Versión preferida:", MODEL_VERSION_PREFERRED or "AUTO/latest")
print("Experimento fallback:", EXPERIMENT_NAME)
print("Artifact path del modelo:", MODEL_ARTIFACT_PATH)
print("Métrica para fallback de runs:", MODEL_SELECTION_METRIC)
print("Tracking URI:", mlflow.get_tracking_uri())
try:
    print("Registry URI:", mlflow.get_registry_uri())
except Exception:
    pass

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

# ============================================================
# 1. Validación de activos y creación de schemas
# ============================================================

def table_exists(full_name: str) -> bool:
    try:
        return spark.catalog.tableExists(full_name)
    except Exception:
        try:
            spark.table(full_name).limit(1).count()
            return True
        except Exception:
            return False

required_tables = [FEATURE_TABLE]
validation_rows = []
for tbl in required_tables:
    exists = table_exists(tbl)
    validation_rows.append((tbl, exists, "OK" if exists else "NO ENCONTRADA"))

validation_sdf = spark.createDataFrame(validation_rows, ["asset", "exists", "status"])
display(validation_sdf)

missing_assets = [x[0] for x in validation_rows if not x[1]]
if missing_assets:
    raise ValueError("Faltan activos requeridos: " + ", ".join(missing_assets))

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA_SERVING}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA_MONITORING}")

print("✅ Activos mínimos validados y schemas listos.")

# COMMAND ----------

# ============================================================
# 2. Descubrimiento robusto de modelo, versiones, sources y runs activos
# ============================================================
# Este bloque evita depender de una versión quemada o de un run viejo.
# En el caso observado, models:/ fallaba por artefactos en UC storage,
# pero las versiones del modelo sí tenían source dbfs:/ válido.

client = MlflowClient() if MlflowClient is not None else None


def _safe_int(x, default=-1):
    try:
        return int(x)
    except Exception:
        return default


def _uri_join(base_uri: str, path: str) -> str:
    base_uri = (base_uri or "").rstrip("/")
    path = (path or "").strip("/")
    return f"{base_uri}/{path}" if path else base_uri


def _get_run_lifecycle(run_id: str):
    """Devuelve metadata de run y experimento. Si el experimento está deleted, lo reporta."""
    if not client or not run_id:
        return {"run_exists": False, "experiment_stage": None, "error": "No client/run_id"}
    try:
        run = client.get_run(run_id)
        exp = client.get_experiment(run.info.experiment_id)
        return {
            "run_exists": True,
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "experiment_name": exp.name if exp else None,
            "experiment_stage": exp.lifecycle_stage if exp else None,
            "artifact_uri": run.info.artifact_uri,
            "run_name": run.data.tags.get("mlflow.runName"),
        }
    except Exception as e:
        return {"run_exists": False, "run_id": run_id, "experiment_stage": None, "error": str(e)[:500]}


def discover_registered_model_candidates(model_name: str, preferred_version: str = None):
    """Construye candidatos desde Model Registry: models:/, source y runs:/ de cada versión."""
    candidates = []
    version_records = []
    if not client:
        return version_records, candidates

    try:
        versions = list(client.search_model_versions(f"name = '{model_name}'"))
    except Exception as e:
        print("⚠️ No se pudieron consultar versiones del modelo registrado:", str(e)[:700])
        return version_records, candidates

    versions = sorted(versions, key=lambda v: _safe_int(v.version), reverse=True)
    if preferred_version:
        versions = [v for v in versions if str(v.version) == str(preferred_version)] + [v for v in versions if str(v.version) != str(preferred_version)]

    seen_uri = set()
    for v in versions:
        lifecycle = _get_run_lifecycle(getattr(v, "run_id", None))
        rec = {
            "version": str(v.version),
            "status": getattr(v, "status", None),
            "run_id": getattr(v, "run_id", None),
            "source": getattr(v, "source", None),
            "current_stage": getattr(v, "current_stage", None),
            "aliases": list(getattr(v, "aliases", []) or []),
            "run_experiment_stage": lifecycle.get("experiment_stage"),
            "run_experiment_id": lifecycle.get("experiment_id"),
            "run_name": lifecycle.get("run_name"),
        }
        version_records.append(rec)

        # 1) Camino gobernado: Registry.
        registry_uri = f"models:/{model_name}/{v.version}"
        if registry_uri not in seen_uri:
            candidates.append({
                "uri": registry_uri,
                "source_type": "models:/ registry",
                "version": str(v.version),
                "run_id": getattr(v, "run_id", None),
                "priority": 10,
                "note": "Preferido para gobierno; puede fallar si UC storage no descarga artefactos."
            })
            seen_uri.add(registry_uri)

        # 2) Source real de la versión. En Databricks suele ser dbfs:/.../artifacts/model.
        source_uri = getattr(v, "source", None)
        if source_uri and source_uri not in seen_uri:
            candidates.append({
                "uri": source_uri,
                "source_type": "version.source",
                "version": str(v.version),
                "run_id": getattr(v, "run_id", None),
                "priority": 20,
                "note": "Fallback fuerte: usa el path real de artefactos de la versión registrada."
            })
            seen_uri.add(source_uri)

        # 3) Run activo de esa versión.
        run_id = getattr(v, "run_id", None)
        if run_id:
            run_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
            if run_uri not in seen_uri:
                candidates.append({
                    "uri": run_uri,
                    "source_type": "runs:/ from registry version",
                    "version": str(v.version),
                    "run_id": run_id,
                    "priority": 30,
                    "note": "Útil si el experimento del run sigue activo."
                })
                seen_uri.add(run_uri)

    return version_records, candidates


def discover_active_run_candidates(experiment_name: str, scenario_hint: str = None, max_results: int = 10):
    """Busca runs activos del experimento para fallback reutilizable."""
    if not client or not experiment_name or not ALLOW_ACTIVE_RUN_FALLBACK:
        return []

    try:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            print(f"⚠️ No existe el experimento fallback: {experiment_name}")
            return []
        if exp.lifecycle_stage != "active":
            print(f"⚠️ El experimento fallback existe pero no está activo: {exp.lifecycle_stage}")
            return []

        runs_pdf = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=max_results,
            order_by=[f"metrics.{MODEL_SELECTION_METRIC} DESC"],
        )
        if runs_pdf.empty:
            return []

        # Filtrado suave por nombre/escenario. Si no encuentra, usa los mejores por R2.
        filtered = runs_pdf.copy()
        if scenario_hint:
            text_cols = [c for c in ["tags.mlflow.runName", "params.scenario", "params.modelo", "params.model_name"] if c in filtered.columns]
            if text_cols:
                mask = False
                for c in text_cols:
                    mask = mask | filtered[c].astype(str).str.contains(scenario_hint, case=False, regex=False, na=False)
                if mask.any():
                    filtered = filtered[mask]

        candidates = []
        seen = set()
        for _, row in filtered.head(max_results).iterrows():
            run_id = row.get("run_id")
            artifact_uri = row.get("artifact_uri")
            run_name = row.get("tags.mlflow.runName", None)
            if not run_id:
                continue
            for uri, stype, prio in [
                (f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}", "runs:/ active experiment", 40),
                (_uri_join(artifact_uri, MODEL_ARTIFACT_PATH), "artifact_uri active experiment", 50),
            ]:
                if uri and uri not in seen:
                    candidates.append({
                        "uri": uri,
                        "source_type": stype,
                        "version": "active_run",
                        "run_id": run_id,
                        "priority": prio,
                        "note": f"Run activo encontrado en {experiment_name}; run_name={run_name}"
                    })
                    seen.add(uri)
        return candidates
    except Exception as e:
        print("⚠️ No se pudieron buscar runs activos de fallback:", str(e)[:700])
        return []

version_records, registry_candidates = discover_registered_model_candidates(MODEL_NAME, MODEL_VERSION_PREFERRED)
active_run_candidates = discover_active_run_candidates(EXPERIMENT_NAME, SCENARIO_HINT, max_results=10)

model_uri_candidates = registry_candidates + active_run_candidates
# Deduplicar preservando orden.
_seen = set()
model_uri_candidates = [c for c in model_uri_candidates if not (c["uri"] in _seen or _seen.add(c["uri"]))]

print("Versiones encontradas en Model Registry:")
if version_records:
    display(pd.DataFrame(version_records))
else:
    print("No se encontraron versiones vía API. Revisa MODEL_NAME o permisos.")

print("\nCandidatos de carga, en orden:")
for i, c in enumerate(model_uri_candidates, 1):
    print(f"{i:02d}. {c['uri']} | {c['source_type']} | version={c.get('version')} | run_id={c.get('run_id')}")

if not model_uri_candidates:
    raise RuntimeError("No se generó ningún candidato de carga. Revisa MODEL_NAME, EXPERIMENT_NAME o permisos de MLflow.")

# COMMAND ----------

# ============================================================
# 3. Preparar dataset de scoring batch
# ============================================================
source_sdf = spark.table(FEATURE_TABLE)

if ORDER_BY_COLUMN and ORDER_BY_COLUMN in source_sdf.columns:
    scoring_sdf = source_sdf.orderBy(F.col(ORDER_BY_COLUMN).desc()).limit(SCORING_LIMIT)
else:
    print(f"⚠️ ORDER_BY_COLUMN={ORDER_BY_COLUMN} no existe. Se tomará una muestra limitada sin ordenar.")
    scoring_sdf = source_sdf.limit(SCORING_LIMIT)

# Guardamos el input del lote para trazabilidad. Esto ayuda a auditar qué datos se usaron para predecir.
scoring_with_batch_sdf = scoring_sdf.withColumn("batch_id", F.lit(BATCH_ID)).withColumn("fecha_inferencia", F.current_timestamp())
(
    scoring_with_batch_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SCORING_INPUT_TABLE)
)

print("✅ Scoring input guardado en:", SCORING_INPUT_TABLE)
print("Filas del lote:", scoring_sdf.count())
display(scoring_sdf.limit(10))

# COMMAND ----------

# ============================================================
# 4. Cargar modelo con fallback robusto
# ============================================================
# Punto clave de blindaje:
# Si models:/ falla por descarga de artefactos desde UC storage, el notebook intenta version.source.
# Si runs:/ falla porque el experimento viejo fue eliminado, el notebook busca runs activos del experimento configurado.
# Además, se intenta cargar también el modelo nativo sklearn para obtener feature_names_in_
# y para tener un fallback cuando MLflow pyfunc falla por validación estricta de signature/dtypes.

model = None                       # Modelo MLflow pyfunc
sklearn_model = None               # Modelo sklearn nativo, si el artifact fue guardado con mlflow.sklearn
model_uri_used = None
model_version_used = None
model_source_type_used = None
run_id_used = None
model_load_errors = []

for candidate in model_uri_candidates:
    uri = candidate["uri"]
    try:
        print(f"Intentando cargar pyfunc: {uri}  [{candidate.get('source_type')}]")
        model = mlflow.pyfunc.load_model(uri)
        model_uri_used = uri
        model_version_used = candidate.get("version")
        model_source_type_used = candidate.get("source_type")
        run_id_used = candidate.get("run_id")
        print("✅ Modelo pyfunc cargado correctamente")
        print("   URI usado:", model_uri_used)
        print("   Fuente:", model_source_type_used)
        print("   Versión:", model_version_used)
        print("   Run ID:", run_id_used)
        break
    except Exception as e:
        msg = {
            "uri": uri,
            "source_type": candidate.get("source_type"),
            "version": candidate.get("version"),
            "run_id": candidate.get("run_id"),
            "error_type": type(e).__name__,
            "error": str(e)[:1200],
        }
        model_load_errors.append(msg)
        print(f"⚠️ Falló pyfunc: {uri} -> {type(e).__name__}: {str(e)[:500]}")

if model is None:
    errors_pdf = pd.DataFrame(model_load_errors)
    display(errors_pdf)
    raise RuntimeError(
        "No fue posible cargar el modelo con ningún candidato. "
        "Revisa si existe un run activo con artefacto 'model' o vuelve a registrar el modelo desde un run activo."
    )

# Intentar cargar el modelo sklearn directo desde el mismo URI que funcionó con pyfunc.
# Esto es especialmente útil cuando el modelo fue guardado con mlflow.sklearn.log_model().
try:
    sklearn_model = mlflow.sklearn.load_model(model_uri_used)
    print("✅ Modelo sklearn nativo cargado correctamente")
    if hasattr(sklearn_model, "feature_names_in_"):
        print("✅ feature_names_in_ detectado en sklearn_model:", list(sklearn_model.feature_names_in_))
except Exception as e:
    sklearn_model = None
    print("⚠️ No se pudo cargar sklearn_model directo. Se usará solo pyfunc.")
    print(f"Detalle: {type(e).__name__}: {str(e)[:500]}")

# Intentar extraer signature MLflow.
# Esta ayuda a documentar el contrato, pero en algunos entornos puede ser más estricta con dtypes.
expected_cols_from_signature = None
model_input_schema = None
schema_type_by_col = {}
try:
    model_input_schema = model.metadata.get_input_schema()
    if model_input_schema is not None:
        expected_cols_from_signature = [x.name for x in model_input_schema.inputs if getattr(x, "name", None)]
        for x in model_input_schema.inputs:
            if getattr(x, "name", None):
                schema_type_by_col[x.name] = str(getattr(x, "type", "")).lower()
except Exception as e:
    print("No se pudo leer la signature desde metadata del modelo:", str(e)[:500])

expected_cols_from_sklearn = None
if sklearn_model is not None and hasattr(sklearn_model, "feature_names_in_"):
    expected_cols_from_sklearn = [str(c) for c in list(sklearn_model.feature_names_in_)]

# Prioridad de contrato:
# 1) MANUAL_EXPECTED_FEATURE_COLUMNS: cuando el instructor quiere forzar un contrato.
# 2) sklearn.feature_names_in_: contrato exacto visto por fit().
# 3) signature de MLflow: útil si no hay sklearn directo.
# 4) automático: usar columnas codificadas resultantes.
if MANUAL_EXPECTED_FEATURE_COLUMNS:
    expected_feature_columns = MANUAL_EXPECTED_FEATURE_COLUMNS
    expected_contract_source = "manual"
elif PREFER_SKLEARN_FEATURE_NAMES_IN and expected_cols_from_sklearn:
    expected_feature_columns = expected_cols_from_sklearn
    expected_contract_source = "sklearn_feature_names_in"
elif expected_cols_from_signature:
    expected_feature_columns = expected_cols_from_signature
    expected_contract_source = "mlflow_signature"
else:
    expected_feature_columns = None
    expected_contract_source = "auto_no_signature"

print("\nResumen de carga:")
print("model_uri_used:", model_uri_used)
print("model_source_type_used:", model_source_type_used)
print("model_version_used:", model_version_used)
print("run_id_used:", run_id_used)
print("expected_contract_source:", expected_contract_source)
print("expected_feature_columns:", expected_feature_columns if expected_feature_columns else "No disponible; se usará contrato automático")
print("schema_type_by_col:", schema_type_by_col if schema_type_by_col else "No disponible")


# COMMAND ----------

# ============================================================
# 5. Construir contrato de entrada y alinear columnas
# ============================================================
# Este bloque es el más reutilizable del notebook.
# Hace 5 cosas:
# 1) identifica target real si está disponible;
# 2) conserva columnas de trazabilidad;
# 3) transforma categóricas con one-hot encoding cuando el modelo espera dummies;
# 4) alinea exactamente con el contrato detectado;
# 5) prepara variantes con dtypes más seguros para pyfunc y sklearn.

pdf = scoring_sdf.toPandas()

if pdf.empty:
    raise ValueError("El lote de scoring está vacío. Revisa FEATURE_TABLE o SCORING_LIMIT.")

# Normalizar fecha si existe.
for c in pdf.columns:
    if c.lower() == "fecha":
        pdf[c] = pd.to_datetime(pdf[c], errors="coerce")

# Target real: en producción futura puede no existir. Para clase sí existe y permite monitorear error.
target_col = next((c for c in TARGET_CANDIDATES if c in pdf.columns), None)
if target_col is None:
    print("⚠️ No se encontró target real. Se generarán predicciones sin métricas de error.")
else:
    print("Target real detectado:", target_col)

trace_cols_available = [c for c in TRACE_COLUMNS if c in pdf.columns]
trace_pdf = pdf[trace_cols_available].copy() if trace_cols_available else pd.DataFrame(index=pdf.index)
if target_col:
    trace_pdf["valor_real"] = pd.to_numeric(pdf[target_col], errors="coerce")
else:
    trace_pdf["valor_real"] = np.nan

# Definir features raw.
if MANUAL_RAW_FEATURE_COLUMNS:
    raw_feature_cols = [c for c in MANUAL_RAW_FEATURE_COLUMNS if c in pdf.columns]
    missing_manual = [c for c in MANUAL_RAW_FEATURE_COLUMNS if c not in pdf.columns]
    if missing_manual:
        print("⚠️ Features manuales no encontradas y se omiten:", missing_manual)
else:
    excluded = set(TRACE_COLUMNS + TARGET_CANDIDATES + EXTRA_DROP_COLUMNS)
    raw_feature_cols = [c for c in pdf.columns if c not in excluded]

if not raw_feature_cols:
    raise ValueError("No se detectaron features raw para inferencia. Define MANUAL_RAW_FEATURE_COLUMNS.")

feature_pdf_raw = pdf[raw_feature_cols].copy()

# One-hot encoding para categóricas configuradas.
# Nota reutilizable:
# - Si el contrato esperado contiene la categórica cruda, por ejemplo "ingenio", NO se codifica.
# - Si el contrato esperado contiene dummies, por ejemplo "ingenio_ILC", sí se codifica.
expected_set = set(expected_feature_columns or [])
raw_categoricals_expected_by_model = [c for c in CATEGORICAL_COLUMNS if c in expected_set]
categoricals_to_encode = [c for c in CATEGORICAL_COLUMNS if c in feature_pdf_raw.columns and c not in raw_categoricals_expected_by_model]

if categoricals_to_encode:
    feature_pdf_encoded = pd.get_dummies(feature_pdf_raw, columns=categoricals_to_encode, drop_first=False, dummy_na=False)
else:
    feature_pdf_encoded = feature_pdf_raw.copy()

# Agregar dummies conocidas aunque no aparezcan en este lote.
for cat_col, values in KNOWN_CATEGORIES.items():
    if cat_col in categoricals_to_encode:
        for value in values:
            dummy_col = f"{cat_col}_{value}"
            if dummy_col not in feature_pdf_encoded.columns:
                feature_pdf_encoded[dummy_col] = 0

# Convertir booleanos a enteros.
bool_cols = feature_pdf_encoded.select_dtypes(include=["bool"]).columns.tolist()
if bool_cols:
    feature_pdf_encoded[bool_cols] = feature_pdf_encoded[bool_cols].astype(int)

# Convertir a numérico las columnas que no sean categóricas crudas esperadas por el modelo.
for c in feature_pdf_encoded.columns:
    if c in raw_categoricals_expected_by_model:
        feature_pdf_encoded[c] = feature_pdf_encoded[c].astype(str).fillna("")
    else:
        feature_pdf_encoded[c] = pd.to_numeric(feature_pdf_encoded[c], errors="coerce")

feature_pdf_encoded = feature_pdf_encoded.replace([np.inf, -np.inf], np.nan)
for c in feature_pdf_encoded.columns:
    if c in raw_categoricals_expected_by_model:
        feature_pdf_encoded[c] = feature_pdf_encoded[c].fillna("")
    else:
        feature_pdf_encoded[c] = feature_pdf_encoded[c].fillna(0)

# Alinear contrato final.
contract_strategy = "encoded_columns_auto"
missing_expected_cols = []
extra_cols_removed = []

if expected_feature_columns:
    expected_feature_columns = [c for c in expected_feature_columns if c and str(c).lower() not in ["none", "null"]]
    missing_expected_cols = [c for c in expected_feature_columns if c not in feature_pdf_encoded.columns]
    for c in missing_expected_cols:
        feature_pdf_encoded[c] = 0

    extra_cols_removed = [c for c in feature_pdf_encoded.columns if c not in expected_feature_columns]
    X_scoring = feature_pdf_encoded[expected_feature_columns].copy()
    contract_strategy = f"contract_from_{expected_contract_source}"
else:
    X_scoring = feature_pdf_encoded.copy()

if X_scoring.empty or X_scoring.shape[1] == 0:
    raise ValueError("El contrato final no tiene columnas. Revisa signature o MANUAL_EXPECTED_FEATURE_COLUMNS.")

# ------------------------------------------------------------
# Coerción de dtypes para evitar errores de MLflow signature.
# ------------------------------------------------------------
def coerce_for_pyfunc_signature(X, schema_type_by_col=None):
    """
    Ajusta tipos para MLflow pyfunc.
    Si hay signature, respeta integer/long/double/string.
    Si no hay signature, usa numérico estable.
    """
    schema_type_by_col = schema_type_by_col or {}
    X2 = X.copy()
    for c in X2.columns:
        expected_type = schema_type_by_col.get(c, "")
        if expected_type in ["integer", "int"]:
            X2[c] = pd.to_numeric(X2[c], errors="coerce").fillna(0).astype("int32")
        elif expected_type in ["long"]:
            X2[c] = pd.to_numeric(X2[c], errors="coerce").fillna(0).astype("int64")
        elif expected_type in ["float"]:
            X2[c] = pd.to_numeric(X2[c], errors="coerce").fillna(0).astype("float32")
        elif expected_type in ["double"]:
            X2[c] = pd.to_numeric(X2[c], errors="coerce").fillna(0).astype("float64")
        elif expected_type in ["boolean", "bool"]:
            X2[c] = X2[c].astype(bool)
        elif expected_type in ["string", "str"]:
            X2[c] = X2[c].astype(str).fillna("")
        else:
            # Si no hay tipo declarado, dejamos numéricas como float64 para estabilidad de sklearn.
            if pd.api.types.is_numeric_dtype(X2[c]):
                X2[c] = pd.to_numeric(X2[c], errors="coerce").fillna(0).astype("float64")
            else:
                X2[c] = X2[c].astype(str).fillna("")
    return X2

X_scoring_pyfunc = coerce_for_pyfunc_signature(X_scoring, schema_type_by_col)
X_scoring_sklearn = X_scoring.copy()
for c in X_scoring_sklearn.columns:
    if c not in raw_categoricals_expected_by_model:
        X_scoring_sklearn[c] = pd.to_numeric(X_scoring_sklearn[c], errors="coerce").fillna(0).astype("float64")

# Para compatibilidad con bloques posteriores, X_scoring queda como contrato final base.
contract_summary = {
    "contract_strategy": contract_strategy,
    "expected_contract_source": expected_contract_source,
    "raw_feature_cols": raw_feature_cols,
    "categorical_present": categoricals_to_encode,
    "raw_categoricals_expected_by_model": raw_categoricals_expected_by_model,
    "final_feature_count": int(X_scoring.shape[1]),
    "final_features": list(X_scoring.columns),
    "missing_expected_cols_added_as_zero": missing_expected_cols,
    "extra_cols_removed": extra_cols_removed,
    "pyfunc_dtypes": {c: str(t) for c, t in X_scoring_pyfunc.dtypes.items()},
    "sklearn_dtypes": {c: str(t) for c, t in X_scoring_sklearn.dtypes.items()},
}

print("Estrategia de contrato:", contract_strategy)
print("Fuente del contrato esperado:", expected_contract_source)
print("Número de features finales:", X_scoring.shape[1])
print("Categóricas codificadas:", categoricals_to_encode)
print("Categóricas crudas esperadas por el modelo:", raw_categoricals_expected_by_model)
print("Columnas faltantes agregadas en cero:", missing_expected_cols)
print("Columnas extra removidas:", extra_cols_removed[:30], "..." if len(extra_cols_removed) > 30 else "")

contract_display_pdf = pd.DataFrame({
    "feature_final": list(X_scoring.columns),
    "dtype_base": [str(X_scoring[c].dtype) for c in X_scoring.columns],
    "dtype_pyfunc": [str(X_scoring_pyfunc[c].dtype) for c in X_scoring.columns],
    "dtype_sklearn": [str(X_scoring_sklearn[c].dtype) for c in X_scoring.columns],
    "schema_type_mlflow": [schema_type_by_col.get(c, "") for c in X_scoring.columns],
})
display(contract_display_pdf)


# COMMAND ----------

# ============================================================
# 6. Ejecutar inferencia batch
# ============================================================
# Este bloque queda blindado con varios intentos:
# 1) pyfunc + dtypes alineados a signature;
# 2) sklearn nativo + feature_names_in_;
# 3) pyfunc con contrato base, como último intento.
#
# Para explicar a estudiantes:
# - pyfunc es el camino estándar/reutilizable de MLflow.
# - sklearn directo es un fallback académico útil cuando pyfunc falla por validación estricta
#   de schema/dtypes, pero el modelo realmente sí puede predecir.

prediction_attempts = []
predictions = None
prediction_engine_used = None
prediction_error_details = []

candidate_prediction_engines = []

# Intento 1: MLflow pyfunc con dtypes ajustados a la signature.
candidate_prediction_engines.append(("mlflow_pyfunc_signature_coerced", model, X_scoring_pyfunc))

# Intento 2: sklearn directo, si está disponible.
if ENABLE_SKLEARN_DIRECT_PREDICTION_FALLBACK and sklearn_model is not None:
    candidate_prediction_engines.append(("sklearn_direct_feature_names_in", sklearn_model, X_scoring_sklearn))

# Intento 3: pyfunc con contrato base.
candidate_prediction_engines.append(("mlflow_pyfunc_base_contract", model, X_scoring))

for engine_name, engine_model, X_candidate in candidate_prediction_engines:
    try:
        print(f"Intentando predicción con: {engine_name}")
        predictions = engine_model.predict(X_candidate)
        prediction_engine_used = engine_name
        X_scoring_used_for_prediction = X_candidate
        print(f"✅ Predicción exitosa con: {engine_name}")
        break
    except Exception as e:
        detail = {
            "engine": engine_name,
            "error_type": type(e).__name__,
            "error": str(e)[:2000],
            "columns": list(X_candidate.columns),
            "dtypes": {c: str(t) for c, t in X_candidate.dtypes.items()},
        }
        prediction_error_details.append(detail)
        print(f"⚠️ Falló predicción con {engine_name}: {type(e).__name__}: {str(e)[:800]}")

if predictions is None:
    print("❌ Falló la predicción con todos los motores disponibles.")
    print("Columnas contrato base:", list(X_scoring.columns))
    print("Tipos contrato base:")
    print(X_scoring.dtypes)
    display(pd.DataFrame(prediction_error_details))
    raise RuntimeError(
        "No fue posible generar predicciones. Revisa la tabla de errores: probablemente el modelo espera "
        "otras columnas, otro orden, o fue entrenado con un preprocesamiento distinto."
    )

pred_array = np.asarray(predictions).reshape(-1)
if len(pred_array) != len(X_scoring):
    raise ValueError(f"El modelo devolvió {len(pred_array)} predicciones para {len(X_scoring)} filas.")

print("✅ Predicciones generadas:", len(pred_array))
print("Motor usado:", prediction_engine_used)
display(pd.DataFrame({"prediccion_preview": pred_array[:10]}))


# COMMAND ----------

# ============================================================
# 7. Guardar predicciones en Delta
# ============================================================
result_pdf = trace_pdf.copy()
result_pdf["bagazo_predicho"] = pd.to_numeric(pred_array, errors="coerce")
result_pdf["bagazo_real"] = pd.to_numeric(result_pdf["valor_real"], errors="coerce")
result_pdf = result_pdf.drop(columns=["valor_real"], errors="ignore")

result_pdf["error"] = result_pdf["bagazo_real"] - result_pdf["bagazo_predicho"]
result_pdf["error_absoluto"] = result_pdf["error"].abs()
result_pdf["error_porcentual"] = np.where(
    result_pdf["bagazo_real"].abs() > 0,
    result_pdf["error_absoluto"] / result_pdf["bagazo_real"].abs(),
    np.nan,
)

result_pdf["modelo_nombre"] = MODEL_NAME
result_pdf["modelo_version"] = str(model_version_used)
result_pdf["run_id"] = run_id_used
result_pdf["model_uri"] = model_uri_used
result_pdf["model_source_type"] = model_source_type_used
result_pdf["escenario"] = SCENARIO_HINT
result_pdf["fecha_inferencia"] = dt.datetime.now()
result_pdf["tipo_inferencia"] = "batch"
result_pdf["batch_id"] = BATCH_ID
result_pdf["contract_strategy"] = contract_strategy
result_pdf["feature_count"] = int(X_scoring.shape[1])

predictions_sdf = spark.createDataFrame(result_pdf)
if "fecha" in predictions_sdf.columns:
    predictions_sdf = predictions_sdf.withColumn("fecha", F.to_date("fecha"))

(
    predictions_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PREDICTIONS_TABLE)
)

print("✅ Predicciones guardadas en:", PREDICTIONS_TABLE)
display(predictions_sdf.orderBy(F.desc("error_absoluto")).limit(10))

# COMMAND ----------

# ============================================================
# 8. Monitoreo básico del modelo
# ============================================================
valid_metrics_pdf = result_pdf.dropna(subset=["bagazo_real", "bagazo_predicho"]).copy()

if valid_metrics_pdf.empty:
    print("⚠️ No hay valores reales disponibles. Se omiten métricas de performance y solo quedan predicciones.")
    mae = rmse = r2 = smape = error_promedio = pct_error_alto = np.nan
    estado_modelo = "NO_REAL_TARGET"
    ingenio_mayor_error = None
else:
    y_true = valid_metrics_pdf["bagazo_real"].astype(float)
    y_pred = valid_metrics_pdf["bagazo_predicho"].astype(float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred)) if len(valid_metrics_pdf) > 1 else np.nan
    except Exception:
        r2 = np.nan
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_values = np.where(denominator > 0, np.abs(y_true - y_pred) / denominator, np.nan)
    smape = float(np.nanmean(smape_values) * 100)
    error_promedio = float((y_true - y_pred).mean())
    pct_error_alto = float((valid_metrics_pdf["error_absoluto"] > ERROR_ALTO_TON).mean() * 100)

    if GROUP_COLUMN in valid_metrics_pdf.columns:
        by_group = (
            valid_metrics_pdf.groupby(GROUP_COLUMN, dropna=False)
            .agg(
                n_predicciones=("bagazo_predicho", "size"),
                mae=("error_absoluto", "mean"),
                error_promedio=("error", "mean"),
                max_error_abs=("error_absoluto", "max"),
            )
            .reset_index()
            .sort_values("mae", ascending=False)
        )
        ingenio_mayor_error = str(by_group.iloc[0][GROUP_COLUMN]) if not by_group.empty else None
        by_group["batch_id"] = BATCH_ID
        by_group["fecha_ejecucion"] = dt.datetime.now()
        by_group["modelo_nombre"] = MODEL_NAME
        by_group["modelo_version"] = str(model_version_used)
        spark.createDataFrame(by_group).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(PERFORMANCE_BY_GROUP_TABLE)
        print("✅ Monitoreo por grupo guardado en:", PERFORMANCE_BY_GROUP_TABLE)
        display(by_group)
    else:
        ingenio_mayor_error = None

    estado_modelo = "OK"
    if mae > THRESHOLD_MAE or rmse > THRESHOLD_RMSE or pct_error_alto > THRESHOLD_ERROR_ALTO_PCT:
        estado_modelo = "REVISAR"

performance_pdf = pd.DataFrame([{
    "batch_id": BATCH_ID,
    "fecha_ejecucion": dt.datetime.now(),
    "modelo_nombre": MODEL_NAME,
    "modelo_version": str(model_version_used),
    "run_id": run_id_used,
    "model_uri_usado": model_uri_used,
    "model_source_type": model_source_type_used,
    "n_predicciones": int(len(result_pdf)),
    "n_con_real": int(len(valid_metrics_pdf)),
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "smape": smape,
    "error_promedio": error_promedio,
    "porcentaje_error_alto": pct_error_alto,
    "ingenio_mayor_error": ingenio_mayor_error,
    "estado_modelo": estado_modelo,
    "threshold_mae": THRESHOLD_MAE,
    "threshold_rmse": THRESHOLD_RMSE,
    "threshold_smape": THRESHOLD_SMAPE,
    "threshold_error_alto_pct": THRESHOLD_ERROR_ALTO_PCT,
}])

spark.createDataFrame(performance_pdf).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(PERFORMANCE_TABLE)

print("✅ Performance guardada en:", PERFORMANCE_TABLE)
display(performance_pdf)

# COMMAND ----------

# ============================================================
# 9. Reglas de alerta
# ============================================================

def alert_row(regla, metrica, valor, umbral, condicion, severidad, accion):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        estado = "NO_EVALUABLE"
    else:
        estado = "ALERTA" if condicion else "OK"
    return {
        "batch_id": BATCH_ID,
        "fecha_ejecucion": dt.datetime.now(),
        "regla": regla,
        "metrica": metrica,
        "valor_observado": None if valor is None or (isinstance(valor, float) and np.isnan(valor)) else float(valor),
        "umbral": float(umbral) if umbral is not None else None,
        "estado": estado,
        "severidad": severidad if estado == "ALERTA" else "INFO",
        "accion_recomendada": accion if estado == "ALERTA" else "Sin acción inmediata.",
        "modelo_nombre": MODEL_NAME,
        "modelo_version": str(model_version_used),
    }

alerts = [
    alert_row("MAE mayor al umbral", "mae", mae, THRESHOLD_MAE, mae > THRESHOLD_MAE if not np.isnan(mae) else False, "WARNING", "Revisar error global del lote y comparar contra histórico."),
    alert_row("RMSE mayor al umbral", "rmse", rmse, THRESHOLD_RMSE, rmse > THRESHOLD_RMSE if not np.isnan(rmse) else False, "WARNING", "Buscar días con errores extremos."),
    alert_row("SMAPE mayor al umbral", "smape", smape, THRESHOLD_SMAPE, smape > THRESHOLD_SMAPE if not np.isnan(smape) else False, "WARNING", "Revisar valores bajos y variabilidad relativa."),
    alert_row("Porcentaje de error alto", "porcentaje_error_alto", pct_error_alto, THRESHOLD_ERROR_ALTO_PCT, pct_error_alto > THRESHOLD_ERROR_ALTO_PCT if not np.isnan(pct_error_alto) else False, "WARNING", "Revisar features, calidad de datos y cambios operativos."),
    alert_row("Volumen de scoring incompleto", "n_predicciones", len(result_pdf), MIN_EXPECTED_ROWS, len(result_pdf) < MIN_EXPECTED_ROWS, "CRITICAL", "Validar llegada completa del lote de scoring."),
]

alerts_pdf = pd.DataFrame(alerts)
spark.createDataFrame(alerts_pdf).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(ALERTS_TABLE)

print("✅ Alertas guardadas en:", ALERTS_TABLE)
display(alerts_pdf)

# COMMAND ----------

# ============================================================
# 10. Lineage y resumen de ejecución
# ============================================================
lineage_pdf = pd.DataFrame([{
    "batch_id": BATCH_ID,
    "fecha_ejecucion": dt.datetime.now(),
    "feature_table": FEATURE_TABLE,
    "scoring_input_table": SCORING_INPUT_TABLE,
    "predictions_table": PREDICTIONS_TABLE,
    "performance_table": PERFORMANCE_TABLE,
    "alerts_table": ALERTS_TABLE,
    "model_name": MODEL_NAME,
    "model_version": str(model_version_used),
    "model_uri_used": model_uri_used,
    "model_source_type": model_source_type_used,
    "run_id_used": run_id_used,
    "contract_strategy": contract_strategy,
    "feature_count": int(X_scoring.shape[1]),
    "contract_summary_json": json.dumps(contract_summary, ensure_ascii=False),
    "load_errors_json": json.dumps(model_load_errors, ensure_ascii=False),
}])

spark.createDataFrame(lineage_pdf).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(LINEAGE_TABLE)

print("✅ Lineage guardado en:", LINEAGE_TABLE)
display(lineage_pdf)

# COMMAND ----------

# ============================================================
# 11. Consultas rápidas para mini dashboard
# ============================================================
print("Tablas generadas:")
for t in [SCORING_INPUT_TABLE, PREDICTIONS_TABLE, PERFORMANCE_TABLE, PERFORMANCE_BY_GROUP_TABLE, ALERTS_TABLE, LINEAGE_TABLE]:
    print("-", t)

print("\nConsulta 1 — Real vs predicho:")
print(f"""
SELECT fecha, ingenio, bagazo_real, bagazo_predicho, error_absoluto
FROM {PREDICTIONS_TABLE}
ORDER BY fecha DESC
""")

print("Consulta 2 — Ranking de mayores errores:")
print(f"""
SELECT fecha, ingenio, bagazo_real, bagazo_predicho, error, error_absoluto
FROM {PREDICTIONS_TABLE}
ORDER BY error_absoluto DESC
LIMIT 20
""")

print("Consulta 3 — Estado de alertas:")
print(f"""
SELECT regla, metrica, valor_observado, umbral, estado, severidad, accion_recomendada
FROM {ALERTS_TABLE}
ORDER BY estado DESC, severidad DESC
""")

# COMMAND ----------

# ============================================================
# 12. Retos del estudiante
# ============================================================
# TODO 1: Interpreta el MAE en lenguaje de negocio.
# Ejemplo esperado: "El modelo se equivoca en promedio X toneladas por predicción..."

# TODO 2: Identifica el ingenio con mayor error promedio.
# Pista: usa la tabla PERFORMANCE_BY_GROUP_TABLE.

# TODO 3: Propón una regla de alerta adicional.
# Pista: puede estar basada en error_promedio, R2, volumen del lote o error por ingenio.

# TODO 4: Explica cuándo usarías models:/ y cuándo usarías runs:/.
# Pista: models:/ para operación gobernada; runs:/ para depuración o fallback académico.

# TODO 5: Propón una mejora para producción empresarial.
# Pista: Lakeflow Jobs, aliases Champion/Challenger, reentrenamiento, dashboard o CI/CD.

print("Retos listos. Responde en celdas Markdown debajo de esta celda.")

# COMMAND ----------

# MAGIC %md
# MAGIC TODO 1: Interpretar el MAE en lenguaje de negocioBasándonos en la corrida real observada en el lote de inferencia batch batch_20260603_223033, el modelo arrojó un Error Absoluto Medio (MAE) de 97.18 toneladas. 
# MAGIC
# MAGIC  En lenguaje de negocio, esto significa que el modelo Champion se equivoca, en promedio, por aproximadamente 97 toneladas en cada estimación diaria de entrega de bagazo por ingenio. Desde la perspectiva de MLOps y suministro, esta métrica es excelente, ya que es sumamente cercana y consistente con el MAE base de 97.01 toneladas calculado originalmente en el holdout de laboratorio de la Sesión 8. Esto demuestra que el modelo mantiene una alta estabilidad y no está experimentando degradación inmediata (data drift) al enfrentarse al nuevo bloque de datos en producción. 
# MAGIC
# MAGIC  TODO 2: Identificar el ingenio con mayor error promedioAl revisar la tabla de monitoreo desagregado workspace.ml_monitoring.bagazo_model_performance_by_group_sesion_09, el comportamiento por proveedor es el siguiente: 
# MAGIC  
# MAGIC   Incauca: Registra el mayor error del lote, con un MAE de 149.25 toneladas y el error absoluto máximo del lote completo, alcanzando las 570.91 toneladas en un solo día.  Providencia: Muestra un error intermedio, con un MAE de 103.14 toneladas.  ILC: Registra el desempeño más preciso, con un MAE sumamente bajo de apenas 39.13 toneladas.  Conclusión: El ingenio con mayor error promedio es Incauca. Operativamente, esto nos indica que el comportamiento de Incauca está experimentando anomalías puntuales o una mayor volatilidad en este período, por lo que las alertas logísticas de inventario deben enfocarse prioritariamente en este proveedor.  
# MAGIC  
# MAGIC  TODO 3: Proponer una regla de alerta adicionalComo regla de monitoreo y gobernanza complementaria, propongo la regla de Desviación de Sesgo Sistemático (Over/Under Prediction Alert) basada en la métrica error_promedio (bias).  
# MAGIC  
# MAGIC  Lógica técnica: Disparar una alerta de severidad WARNING si el error_promedio global del lote es menor a -50 toneladas o mayor a 50 toneladas.Justificación de negocio: El MAE mide la magnitud del error, pero es ciego al sentido del mismo. Si el error promedio se desvía drásticamente hacia los negativos (como el -19.18 observado en este lote), significa que el modelo está sobreestimando de forma sistemática el bagazo que llegará. Una sobreestimación persistente en producción es peligrosa porque puede inducir a la planta papelera a creer que tiene biomasa garantizada, cuando en la práctica real llegará menos combustible de lo esperado. 
# MAGIC  
# MAGIC   TODO 4: Explicar cuándo usarías models:/ y cuándo usarías runs:
# MAGIC   
# MAGIC   /Cuándo usar models:/ (Model Registry): Se debe utilizar obligatoriamente en entornos productivos, pipelines de inferencia batch automatizados y aplicaciones de consumo final. Apuntar a rutas de gobernanza como models:/ garantiza que la organización consuma una versión explícitamente aprobada por el gobierno de MLOps (ej. la versión con el alias de Champion), independientemente de la infraestructura técnica interna que la generó. 
# MAGIC   
# MAGIC    Cuándo usar runs:/ (Experiment Tracking): Se utiliza exclusivamente durante las fases de investigación, desarrollo (R&D), depuración de código, análisis de importancia de variables o como un mecanismo de respaldo (fallback) académico y de emergencia en entornos locales (como la Free Edition de Databricks). Acceder vía runs:/ te vincula de forma directa a un experimento específico, permitiendo rastrear los gráficos o artefactos exactos generados en esa ejecución, pero carece del control de ciclo de vida necesario para producción.  
# MAGIC   
# MAGIC   TODO 5: Proponer una mejora para producción empresarialPara robustecer este sistema a nivel corporativo, propongo la implementación de un Pipeline de Reentrenamiento Automatizado y Evaluación Sombra (Champion/Challenger) orquestado por Databricks Workflows (o Lakeflow Jobs)
# MAGIC   
# MAGIC    Mecanismo sugerido: Cuando la tabla de monitoreo diario detecte que la alerta de MAE entra en estado ALERTA de forma sostenida por más de 7 días, se debe disparar un trigger automático que ejecute un job de reentrenamiento incorporando los datos más recientes de la capa Gold. El nuevo modelo candidato (Challenger) no reemplazará al Champion de inmediato; el pipeline de MLOps lo evaluará de forma silenciosa ("sombra") contra los lotes diarios. Si el Challenger demuestra una reducción del error absoluto superior al 10% frente al Champion, un webhook de Unity Catalog cambiará el alias productivo de manera transparente y sin interrumpir el servicio, garantizando resiliencia operativa continua.  

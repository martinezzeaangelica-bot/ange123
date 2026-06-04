# Databricks notebook source
# MAGIC %md
# MAGIC # Sesión 8 — Machine Learning tradicional con MLflow
# MAGIC
# MAGIC ## Del dashboard al laboratorio de Machine Learning
# MAGIC
# MAGIC En la Sesión 7 publicamos la capa Gold, KPIs y dashboards. En esta sesión convertimos ese producto analítico en un laboratorio reproducible de modelos con MLflow.
# MAGIC
# MAGIC **Pregunta guía:** ¿podemos estimar el bagazo entregado por ingenio usando lluvia, caña molida, estacionalidad e histórico reciente?
# MAGIC
# MAGIC **Caso principal:** Bagazo.  
# MAGIC **Caso complementario:** Lumi, solo como reto opcional de experiencia/delivery.
# MAGIC
# MAGIC > MLflow convierte el entrenamiento de modelos en un proceso trazable, comparable, auditable y gobernable.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Reglas de seguridad
# MAGIC
# MAGIC - No modificar `workspace.bagazo_gold.*`.
# MAGIC - No modificar `workspace.lumi_gold.*`.
# MAGIC - No usar `workspace.delta_lab.*` como fuente de ML.
# MAGIC - Crear únicamente schemas y tablas nuevas bajo `workspace.ml_*`.
# MAGIC

# COMMAND ----------

# ==========================================================
# Configuración general de la Sesión 8
# ==========================================================
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import *

import pandas as pd
import numpy as np
import os
import json
import tempfile
import warnings
warnings.filterwarnings("ignore")

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
try:
    from mlflow.models.signature import infer_signature
except Exception:
    infer_signature = None

CATALOG = "workspace"
SOURCE_BAGAZO = f"{CATALOG}.bagazo_gold.fact_operacion_ingenios"
CONTROL_GOLD = f"{CATALOG}.control.gold_publication_summary_sesion_07"

SCHEMA_FEATURES = f"{CATALOG}.ml_features"
SCHEMA_MODELS = f"{CATALOG}.ml_models"
SCHEMA_MONITORING = f"{CATALOG}.ml_monitoring"

TABLE_FEATURES = f"{SCHEMA_FEATURES}.bagazo_features_training"
TABLE_SUMMARY = f"{SCHEMA_MODELS}.bagazo_experiment_summary_sesion_08"
TABLE_HOLDOUT = f"{SCHEMA_MONITORING}.bagazo_holdout_predictions_sesion_08"
TABLE_REGISTRY_LOG = f"{SCHEMA_MODELS}.model_registry_attempt_log_sesion_08"

RANDOM_STATE = 42
TEST_SIZE = 0.20
ENABLE_OPTIONAL_GRADIENT_BOOSTING = False
ENABLE_UC_MODEL_REGISTRY = False

print("✅ Configuración cargada")
print(f"Fuente principal: {SOURCE_BAGAZO}")
print(f"Tabla de features: {TABLE_FEATURES}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Validar fuentes Gold
# MAGIC
# MAGIC El modelo parte de Gold, no de Bronze. Esta es una decisión de trazabilidad y gobierno.
# MAGIC

# COMMAND ----------

# ==========================================================
# 1. Validar fuentes Gold y control
# ==========================================================
def table_exists(full_name: str) -> bool:
    try:
        return spark.catalog.tableExists(full_name)
    except Exception:
        try:
            spark.sql(f"DESCRIBE TABLE {full_name}")
            return True
        except Exception:
            return False

required_tables = [SOURCE_BAGAZO]
for tbl in required_tables:
    if not table_exists(tbl):
        raise ValueError(f"❌ No existe la tabla requerida: {tbl}. Ejecuta primero la Sesión 7.")
    print(f"✅ Existe: {tbl}")

bagazo_raw = spark.table(SOURCE_BAGAZO)
print("Columnas disponibles en Gold Bagazo:")
print(bagazo_raw.columns)

conteo = bagazo_raw.agg(
    F.count("*").alias("filas")
)
display(conteo)

# Validación de granularidad fecha + ingenio. Se resuelven nombres de columna de forma robusta.
def resolve_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"No encontré ninguna columna candidata: {candidates}. Columnas actuales: {df.columns}")
    return None

COL_FECHA = resolve_col(bagazo_raw, ["fecha", "date", "fecha_operacion"])
COL_INGENIO = resolve_col(bagazo_raw, ["ingenio", "planta", "nombre_ingenio"])
COL_LLUVIA = resolve_col(bagazo_raw, ["lluvia_mm", "promedio_lluvias_mm", "promedio_lluvia_mm", "lluvia", "precipitacion_mm"])
COL_CANA = resolve_col(bagazo_raw, ["cana_molida_ton", "caña_molida_ton", "cana_molida_toneladas", "caña_molida_toneladas", "cana_molida"])
COL_BAGAZO = resolve_col(bagazo_raw, ["bagazo_entregado_ton", "bagazo_entregado_toneladas", "bagazo_entregado", "bagazo"])
COL_COMENTARIOS = resolve_col(bagazo_raw, ["comentarios", "comentario", "comentarios_operativos", "comentarios_operacion"], required=False)
COL_RIESGO = resolve_col(bagazo_raw, ["riesgo_bajo_bagazo", "target_riesgo_bajo_bagazo", "flag_riesgo_bajo_bagazo"], required=False)

print("\nColumnas resueltas:")
for k, v in {
    "fecha": COL_FECHA,
    "ingenio": COL_INGENIO,
    "lluvia": COL_LLUVIA,
    "caña": COL_CANA,
    "bagazo": COL_BAGAZO,
    "comentarios": COL_COMENTARIOS,
    "riesgo": COL_RIESGO
}.items():
    print(f"- {k}: {v}")

granularidad = bagazo_raw.select(
    F.col(COL_FECHA).alias("fecha"),
    F.col(COL_INGENIO).alias("ingenio")
).agg(
    F.count("*").alias("filas"),
    F.countDistinct("fecha", "ingenio").alias("combinaciones_fecha_ingenio")
)
display(granularidad)

if table_exists(CONTROL_GOLD):
    print("✅ Tabla de control Gold disponible")
    display(spark.table(CONTROL_GOLD).limit(20))
else:
    print("⚠️ No se encontró la tabla de control Gold. El flujo puede continuar, pero revisa la Sesión 7.")


# COMMAND ----------

# ==========================================================
# 2. Crear schemas ML sin tocar Gold
# ==========================================================
for schema_name in [SCHEMA_FEATURES, SCHEMA_MODELS, SCHEMA_MONITORING]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
    print(f"✅ Schema listo: {schema_name}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature engineering Bagazo
# MAGIC
# MAGIC Construiremos variables temporales, operativas e históricas por `ingenio`. Las ventanas móviles usan únicamente días anteriores.
# MAGIC

# COMMAND ----------

# ==========================================================
# 3. Feature engineering Bagazo con PySpark
# ==========================================================
comentarios_expr = F.lit("") if COL_COMENTARIOS is None else F.coalesce(F.col(COL_COMENTARIOS).cast("string"), F.lit(""))

base = (
    bagazo_raw
    .select(
        F.to_date(F.col(COL_FECHA)).alias("fecha"),
        F.col(COL_INGENIO).cast("string").alias("ingenio"),
        F.col(COL_LLUVIA).cast("double").alias("lluvia_mm"),
        F.col(COL_CANA).cast("double").alias("cana_molida_ton"),
        F.col(COL_BAGAZO).cast("double").alias("bagazo_entregado_ton"),
        comentarios_expr.alias("comentarios_operativos")
    )
    .filter(F.col("fecha").isNotNull())
    .filter(F.col("ingenio").isNotNull())
)

# Umbral de riesgo bajo por ingenio: si no viene desde Gold, se estima con percentil 25 por ingenio.
if COL_RIESGO is not None:
    riesgo_df = bagazo_raw.select(
        F.to_date(F.col(COL_FECHA)).alias("fecha"),
        F.col(COL_INGENIO).cast("string").alias("ingenio"),
        F.col(COL_RIESGO).cast("int").alias("target_riesgo_bajo_bagazo")
    )
    base = base.join(riesgo_df, on=["fecha", "ingenio"], how="left")
else:
    umbrales = base.groupBy("ingenio").agg(
        F.expr("percentile_approx(bagazo_entregado_ton, 0.25)").alias("umbral_riesgo_bajo_bagazo")
    )
    base = (
        base.join(umbrales, on="ingenio", how="left")
        .withColumn(
            "target_riesgo_bajo_bagazo",
            F.when(F.col("bagazo_entregado_ton") <= F.col("umbral_riesgo_bajo_bagazo"), F.lit(1)).otherwise(F.lit(0))
        )
        .drop("umbral_riesgo_bajo_bagazo")
    )

w = Window.partitionBy("ingenio").orderBy("fecha")
w_prev_7 = w.rowsBetween(-7, -1)
w_prev_14 = w.rowsBetween(-14, -1)

features = (
    base
    .withColumn("anio", F.year("fecha"))
    .withColumn("mes", F.month("fecha"))
    .withColumn("dia_semana", F.dayofweek("fecha"))
    .withColumn("dia_mes", F.dayofmonth("fecha"))
    .withColumn("semana_anio", F.weekofyear("fecha"))
    .withColumn("trimestre", F.quarter("fecha"))
    .withColumn("es_fin_de_semana", F.when(F.col("dia_semana").isin(1, 7), 1).otherwise(0))
    .withColumn("lluvia_alta", F.when(F.col("lluvia_mm") >= 10, 1).otherwise(0))
    .withColumn("temporada_lluviosa", F.when(F.col("mes").isin(4,5,9,10,11), 1).otherwise(0))
    .withColumn("tiene_comentario_operativo", F.when(F.length(F.trim("comentarios_operativos")) > 0, 1).otherwise(0))
    .withColumn("lluvia_lag_1", F.lag("lluvia_mm", 1).over(w))
    .withColumn("lluvia_lag_7", F.lag("lluvia_mm", 7).over(w))
    .withColumn("lluvia_promedio_7d", F.avg("lluvia_mm").over(w_prev_7))
    .withColumn("lluvia_promedio_14d", F.avg("lluvia_mm").over(w_prev_14))
    .withColumn("bagazo_lag_1", F.lag("bagazo_entregado_ton", 1).over(w))
    .withColumn("bagazo_lag_7", F.lag("bagazo_entregado_ton", 7).over(w))
    .withColumn("bagazo_promedio_7d", F.avg("bagazo_entregado_ton").over(w_prev_7))
    .withColumn("cana_lag_1", F.lag("cana_molida_ton", 1).over(w))
    .withColumn("cana_promedio_7d", F.avg("cana_molida_ton").over(w_prev_7))
    .withColumn("target_bagazo_entregado_ton", F.col("bagazo_entregado_ton"))
)

# Filtrar primeras filas sin histórico suficiente para evitar nulos en lags relevantes.
required_feature_cols = [
    "fecha", "ingenio", "anio", "mes", "dia_semana", "dia_mes", "semana_anio",
    "lluvia_mm", "cana_molida_ton", "lluvia_alta", "temporada_lluviosa", "tiene_comentario_operativo",
    "lluvia_lag_1", "lluvia_lag_7", "lluvia_promedio_7d", "lluvia_promedio_14d",
    "bagazo_lag_1", "bagazo_lag_7", "bagazo_promedio_7d", "cana_lag_1", "cana_promedio_7d",
    "target_bagazo_entregado_ton", "target_riesgo_bajo_bagazo"
]

features_training = features.select(*required_feature_cols).dropna(subset=[
    "lluvia_lag_7", "lluvia_promedio_14d", "bagazo_lag_7", "bagazo_promedio_7d", "cana_lag_1", "cana_promedio_7d", "target_bagazo_entregado_ton"
])

features_training.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable(TABLE_FEATURES)
print(f"✅ Tabla creada: {TABLE_FEATURES}")

display(spark.table(TABLE_FEATURES).orderBy("ingenio", "fecha").limit(10))


# COMMAND ----------

# ==========================================================
# 4. Validaciones del dataset de features
# ==========================================================
features_df = spark.table(TABLE_FEATURES)

resumen_features = features_df.agg(
    F.count("*").alias("filas_features"),
    F.countDistinct("fecha", "ingenio").alias("granularidad_fecha_ingenio"),
    F.countDistinct("ingenio").alias("ingenios"),
    F.min("fecha").alias("fecha_min"),
    F.max("fecha").alias("fecha_max"),
    F.avg("target_bagazo_entregado_ton").alias("target_promedio")
)
display(resumen_features)

nulos = features_df.select([
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in features_df.columns
])
display(nulos)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Escenarios predictivos
# MAGIC
# MAGIC Escenario A usa caña del mismo día. Escenario B excluye caña del mismo día para reducir riesgo de leakage operacional.
# MAGIC

# COMMAND ----------

# ==========================================================
# 5. Escenarios predictivos: Nowcasting vs Forecast sin leakage
# ==========================================================
TARGET = "target_bagazo_entregado_ton"
ID_COLS = ["fecha", "ingenio"]

FEATURES_NOWCASTING = [
    "ingenio", "anio", "mes", "dia_semana", "dia_mes", "semana_anio", "trimestre", "es_fin_de_semana",
    "lluvia_mm", "cana_molida_ton", "lluvia_alta", "temporada_lluviosa", "tiene_comentario_operativo",
    "lluvia_lag_1", "lluvia_lag_7", "lluvia_promedio_7d", "lluvia_promedio_14d",
    "bagazo_lag_1", "bagazo_lag_7", "bagazo_promedio_7d", "cana_lag_1", "cana_promedio_7d"
]

FEATURES_FORECAST = [
    "ingenio", "anio", "mes", "dia_semana", "dia_mes", "semana_anio", "trimestre", "es_fin_de_semana",
    "lluvia_mm", "lluvia_alta", "temporada_lluviosa", "tiene_comentario_operativo",
    "lluvia_lag_1", "lluvia_lag_7", "lluvia_promedio_7d", "lluvia_promedio_14d",
    "bagazo_lag_1", "bagazo_lag_7", "bagazo_promedio_7d", "cana_lag_1", "cana_promedio_7d"
]

SCENARIOS = {
    "A_nowcasting_operativo": {
        "features": FEATURES_NOWCASTING,
        "description": "Incluye caña molida del mismo día. Útil para estimación operativa si la variable está disponible al momento de estimar.",
        "leakage_risk": "medio"
    },
    "B_forecast_sin_leakage": {
        "features": FEATURES_FORECAST,
        "description": "Excluye caña molida del mismo día y usa histórico reciente. Más realista para anticipación.",
        "leakage_risk": "bajo"
    }
}

for name, meta in SCENARIOS.items():
    print(f"\n{name}")
    print(meta["description"])
    print(f"Features: {len(meta['features'])}")
    print(f"Riesgo leakage: {meta['leakage_risk']}")


# COMMAND ----------

from pyspark.sql import functions as F

TABLE_FEATURES = "workspace.ml_features.bagazo_features_training"

features_fixed = (
    spark.table(TABLE_FEATURES)
    .withColumn(
        "trimestre",
        F.quarter(F.col("fecha")).cast("int")
    )
    .withColumn(
        "es_fin_de_semana",
        F.when(F.col("dia_semana").isin(1, 7), F.lit(1)).otherwise(F.lit(0)).cast("int")
    )
)

(
    features_fixed
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_FEATURES)
)

print("✅ Tabla de features corregida con trimestre y es_fin_de_semana")

spark.table(TABLE_FEATURES).printSchema()

# COMMAND ----------

# ==========================================================
# 6. Preparar datos para scikit-learn con split temporal
# ==========================================================
def prepare_sklearn_dataset(spark_df, feature_cols, target_col=TARGET, test_size=TEST_SIZE):
    """
    Prepara un dataset para scikit-learn conservando la trazabilidad operacional.

    La tabla de features trae columnas de identificación como fecha e ingenio.
    Ingenio también se usa como feature categórica del modelo, por eso la selección
    inicial debe deduplicar columnas antes de convertir a pandas.
    """

    id_cols = list(dict.fromkeys(ID_COLS))
    feature_cols_clean = list(dict.fromkeys(feature_cols))
    required_cols = list(dict.fromkeys(id_cols + feature_cols_clean + [target_col]))

    available_cols = spark_df.columns
    missing_cols = [c for c in required_cols if c not in available_cols]
    if missing_cols:
        raise ValueError(
            "Faltan columnas en la tabla de features: "
            + ", ".join(missing_cols)
            + "Columnas disponibles: "
            + ", ".join(available_cols)
        )

    subset_cols = list(dict.fromkeys(feature_cols_clean + [target_col]))

    pdf = (
        spark_df
        .select(*required_cols)
        .dropna(subset=subset_cols)
        .toPandas()
    )

    pdf["fecha"] = pd.to_datetime(pdf["fecha"])
    pdf = pdf.sort_values(["fecha", "ingenio"]).reset_index(drop=True)

    trace = pdf[id_cols].copy()
    y = pdf[target_col].astype(float).copy()
    X_raw = pdf[feature_cols_clean].copy()

    categorical_cols = [c for c in ["ingenio"] if c in X_raw.columns]
    X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)

    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()
    trace_train = trace.iloc[:split_index].copy()
    trace_test = trace.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test, trace_train, trace_test

# Prueba rápida con escenario principal sin leakage
X_train_demo, X_test_demo, y_train_demo, y_test_demo, trace_train_demo, trace_test_demo = prepare_sklearn_dataset(
    spark.table(TABLE_FEATURES), SCENARIOS["B_forecast_sin_leakage"]["features"]
)
print("✅ Dataset preparado")
print("Train:", X_train_demo.shape, "Test:", X_test_demo.shape)
print("Rango train:", trace_train_demo["fecha"].min(), "→", trace_train_demo["fecha"].max())
print("Rango test:", trace_test_demo["fecha"].min(), "→", trace_test_demo["fecha"].max())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MLflow Tracking
# MAGIC
# MAGIC Cada modelo será una carrera. Cada carrera guardará parámetros, métricas, artefactos y el modelo.
# MAGIC

# COMMAND ----------

# ==========================================================
# 7. Funciones de evaluación y logging MLflow
# ==========================================================
def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


def evaluate_regression(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred))
    }


def create_residual_plot(y_true, y_pred, output_path):
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.65)
    plt.axhline(0, linestyle="--")
    plt.title("Residual plot — Bagazo")
    plt.xlabel("Predicción")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def create_actual_vs_predicted_plot(y_true, y_pred, output_path):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.65)
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.title("Actual vs Predicted — Bagazo")
    plt.xlabel("Bagazo real")
    plt.ylabel("Bagazo predicho")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def create_feature_importance(model, X_test, y_test, feature_names, output_csv, output_png=None):
    importance_df = None
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
    elif hasattr(model, "coef_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(model.coef_)
        }).sort_values("importance", ascending=False)
    else:
        try:
            result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": result.importances_mean
            }).sort_values("importance", ascending=False)
        except Exception:
            importance_df = pd.DataFrame({"feature": [], "importance": []})

    importance_df.to_csv(output_csv, index=False)

    if output_png and len(importance_df) > 0:
        top = importance_df.head(12).sort_values("importance", ascending=True)
        plt.figure(figsize=(8, 5))
        plt.barh(top["feature"], top["importance"])
        plt.title("Top feature importance")
        plt.xlabel("Importancia")
        plt.tight_layout()
        plt.savefig(output_png, dpi=140)
        plt.close()

    return importance_df


def build_model_card_text(run_name, scenario_name, model_name, metrics, feature_cols, limitations=None):
    limitations = limitations or []
    return f"""# Model Card — Bagazo MLflow Sesión 8

## Nombre del modelo
{run_name}

## Problema de negocio
Estimar el bagazo entregado por ingenio usando lluvia, caña molida, estacionalidad e histórico reciente.

## Variable objetivo
`target_bagazo_entregado_ton`

## Fuente
`{SOURCE_BAGAZO}` → `{TABLE_FEATURES}`

## Escenario
{scenario_name}

## Modelo
{model_name}

## Métricas holdout
- MAE: {metrics['mae']:.4f}
- RMSE: {metrics['rmse']:.4f}
- R2: {metrics['r2']:.4f}
- SMAPE: {metrics['smape']:.4f}%

## Features principales
{', '.join(feature_cols)}

## Limitaciones
""" + "\n".join([f"- {x}" for x in limitations]) + "\n\n## Siguiente paso MLOps\nConvertir este champion en inferencia batch en la Sesión 9, guardando predicciones y monitoreando error.\n"


def train_and_log_model(scenario_name, scenario_meta, model_name, model, experiment_path):
    feature_cols = scenario_meta["features"]
    X_train, X_test, y_train, y_test, trace_train, trace_test = prepare_sklearn_dataset(
        spark.table(TABLE_FEATURES), feature_cols
    )

    run_name = f"{scenario_name}__{model_name}"
    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)

        params = {
            "modelo": model_name,
            "escenario": scenario_name,
            "target": TARGET,
            "split_strategy": "temporal_80_20",
            "features_count": len(feature_cols),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "random_state": RANDOM_STATE,
            "leakage_risk": scenario_meta["leakage_risk"]
        }
        # Agregar hiperparámetros si el modelo los expone
        if hasattr(model, "get_params"):
            for k, v in model.get_params().items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    params[f"hp_{k}"] = v

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "sesion": "08",
            "caso": "bagazo",
            "tipo": "regresion",
            "fuente_gold": SOURCE_BAGAZO
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = os.path.abspath(tmpdir)
            metrics_path = os.path.join(tmpdir, "metrics_summary.csv")
            pred_path = os.path.join(tmpdir, "predictions_holdout_sample.csv")
            residuals_path = os.path.join(tmpdir, "residuals_plot.png")
            actual_vs_pred_path = os.path.join(tmpdir, "actual_vs_predicted_plot.png")
            importance_path = os.path.join(tmpdir, "feature_importance.csv")
            importance_png_path = os.path.join(tmpdir, "feature_importance_plot.png")
            model_card_path = os.path.join(tmpdir, "model_card_bagazo.md")
            config_path = os.path.join(tmpdir, "training_config.json")

            pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
            pred_df = trace_test.copy()
            pred_df["bagazo_real"] = y_test.values
            pred_df["bagazo_predicho"] = y_pred
            pred_df["error"] = pred_df["bagazo_real"] - pred_df["bagazo_predicho"]
            pred_df["error_absoluto"] = pred_df["error"].abs()
            pred_df["modelo"] = model_name
            pred_df["run_id"] = run.info.run_id
            pred_df["escenario"] = scenario_name
            pred_df["fecha_ejecucion"] = pd.Timestamp.now()
            pred_df.to_csv(pred_path, index=False)

            create_residual_plot(y_test, y_pred, residuals_path)
            create_actual_vs_predicted_plot(y_test, y_pred, actual_vs_pred_path)
            create_feature_importance(model, X_test, y_test, X_train.columns.tolist(), importance_path, importance_png_path)

            model_card = build_model_card_text(
                run_name=run_name,
                scenario_name=scenario_name,
                model_name=model_name,
                metrics=metrics,
                feature_cols=feature_cols,
                limitations=[
                    "Dataset educativo de tamaño reducido en Databricks Free Edition.",
                    "No incorpora variables de humedad, logística, mantenimiento, inventario ni transporte.",
                    "Las ventanas históricas dependen de la calidad del registro por ingenio.",
                    "El escenario nowcasting puede no ser válido si caña molida no está disponible al momento de predecir."
                ]
            )
            open(model_card_path, "w", encoding="utf-8").write(model_card)
            open(config_path, "w", encoding="utf-8").write(json.dumps(params, ensure_ascii=False, indent=2))

            for artifact in [metrics_path, pred_path, residuals_path, actual_vs_pred_path, importance_path, model_card_path, config_path]:
                mlflow.log_artifact(artifact)
            if os.path.exists(importance_png_path):
                mlflow.log_artifact(importance_png_path)

        # Log del modelo. La firma ayuda a documentar entradas/salidas si el entorno lo permite.
        try:
            if infer_signature is not None:
                signature = infer_signature(X_train.head(5), model.predict(X_train.head(5)))
                mlflow.sklearn.log_model(model, artifact_path="model", signature=signature, input_example=X_train.head(5))
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception as e:
            print(f"⚠️ No se pudo inferir firma; se loggea el modelo sin signature. Detalle: {e}")
            mlflow.sklearn.log_model(model, artifact_path="model")

        result = {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "escenario": scenario_name,
            "modelo": model_name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "smape": metrics["smape"],
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features_count": int(len(feature_cols)),
            "leakage_risk": scenario_meta["leakage_risk"],
            "model_uri": f"runs:/{run.info.run_id}/model",
            "fecha_ejecucion": pd.Timestamp.now()
        }
        return result, pred_df, model


# COMMAND ----------

# ==========================================================
# 8. Ejecutar experimento MLflow: baseline vs modelos candidatos
# ==========================================================
def configure_mlflow_experiment(experiment_name="sesion_08_bagazo_mlflow"):
    """
    Configura MLflow Tracking para Databricks.

    En algunos entornos Free Edition / Spark Connect, MLflow puede intentar
    resolver configuraciones internas de Model Registry que no están disponibles.
    Por eso se define explícitamente el tracking URI y el registry URI antes
    de crear o seleccionar el experimento.
    """

    try:
        username = spark.sql("SELECT current_user() AS user").collect()[0]["user"]
        experiment_path = f"/Users/{username}/{experiment_name}"
    except Exception:
        experiment_path = f"/Shared/{experiment_name}"

    try:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks")
        mlflow.set_experiment(experiment_path)

        print(f"✅ Experimento MLflow configurado: {experiment_path}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Registry URI: {mlflow.get_registry_uri()}")

        return experiment_path, "databricks"

    except Exception as e:
        print("⚠️ No fue posible configurar MLflow contra el tracking nativo de Databricks.")
        print("Se usará un tracking local como alternativa para no detener la sesión.")
        print(f"Detalle técnico: {type(e).__name__}: {str(e)[:500]}")

        local_tracking_dir = "/tmp/mlruns_sesion_08_bagazo"
        mlflow.set_tracking_uri(f"file:{local_tracking_dir}")
        mlflow.set_registry_uri(f"file:{local_tracking_dir}")
        mlflow.set_experiment(experiment_name)

        print(f"✅ Experimento MLflow configurado en modo local: {local_tracking_dir}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")

        return experiment_name, "local"


EXPERIMENT_PATH, MLFLOW_MODE = configure_mlflow_experiment()

models_to_train = {
    "baseline_dummy_mean": DummyRegressor(strategy="mean"),
    "ridge_regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "random_forest": RandomForestRegressor(
        n_estimators=120,
        max_depth=7,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
}

if ENABLE_OPTIONAL_GRADIENT_BOOSTING:
    models_to_train["gradient_boosting"] = GradientBoostingRegressor(random_state=RANDOM_STATE)

results = []
predictions_by_run = {}
models_by_run = {}

for scenario_name, scenario_meta in SCENARIOS.items():
    print(f"\n🚀 Escenario: {scenario_name}")

    for model_name, model in models_to_train.items():
        print(f"Entrenando: {model_name}")

        result, pred_df, fitted_model = train_and_log_model(
            scenario_name=scenario_name,
            scenario_meta=scenario_meta,
            model_name=model_name,
            model=model,
            experiment_path=EXPERIMENT_PATH
        )

        result["mlflow_mode"] = MLFLOW_MODE

        results.append(result)
        predictions_by_run[result["run_id"]] = pred_df
        models_by_run[result["run_id"]] = fitted_model

        print(
            f"✅ {result['run_name']} | "
            f"MAE={result['mae']:.2f} | "
            f"RMSE={result['rmse']:.2f} | "
            f"R2={result['r2']:.3f}"
        )

summary_pdf = (
    pd.DataFrame(results)
    .sort_values(["mae", "rmse"])
    .reset_index(drop=True)
)

display(spark.createDataFrame(summary_pdf))


# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO 1 — Interpretar MAE
# MAGIC
# MAGIC Escribe una frase de negocio: ¿qué significa un MAE de X toneladas para un ingenio?
# MAGIC

# COMMAND ----------

# ==========================================================
# 9. Comparación de experimentos y selección de champion
# ==========================================================
summary_pdf = pd.DataFrame(results).copy()
summary_pdf["rank_mae"] = summary_pdf["mae"].rank(method="dense", ascending=True).astype(int)
summary_pdf["rank_rmse"] = summary_pdf["rmse"].rank(method="dense", ascending=True).astype(int)
summary_pdf["criterio_operativo"] = np.where(summary_pdf["escenario"].str.startswith("B_"), "anticipacion", "estimacion_operativa")
summary_pdf["recomendacion"] = np.where(
    summary_pdf["escenario"].str.startswith("B_"),
    "preferible si el objetivo es anticipar decisiones sin depender de caña del mismo día",
    "útil si caña molida del día está disponible al momento de estimar"
)

# Criterio sugerido: priorizar escenario B si su MAE está dentro del 20% del mejor MAE global; de lo contrario, discutir trade-off.
best_global_mae = summary_pdf["mae"].min()
scenario_b = summary_pdf[summary_pdf["escenario"].str.startswith("B_")].sort_values(["mae", "rmse"])
if len(scenario_b) > 0 and scenario_b.iloc[0]["mae"] <= best_global_mae * 1.20:
    champion = scenario_b.iloc[0].to_dict()
else:
    champion = summary_pdf.sort_values(["mae", "rmse"]).iloc[0].to_dict()

summary_pdf["is_champion"] = summary_pdf["run_id"] == champion["run_id"]
summary_spark = spark.createDataFrame(summary_pdf)
summary_spark.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable(TABLE_SUMMARY)

print(f"✅ Tabla resumen creada: {TABLE_SUMMARY}")
print("🏆 Champion sugerido")
print(json.dumps({k: str(v) for k, v in champion.items()}, ensure_ascii=False, indent=2))

display(spark.table(TABLE_SUMMARY).orderBy(F.desc("is_champion"), F.asc("mae")))


# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO 2 — Elegir champion
# MAGIC
# MAGIC Revisa la tabla comparativa y justifica si estás de acuerdo con el champion sugerido. Considera MAE, RMSE, interpretabilidad y leakage.
# MAGIC

# COMMAND ----------

# ==========================================================
# 10. Modelo champion: registro empresarial y fallback académico
# ==========================================================
"""
En esta sesión el modelo champion ya quedó loggeado dentro del run de MLflow.
Ese artifact es suficiente para la práctica de Free Edition: permite revisar el modelo,
sus métricas, sus parámetros y sus artefactos desde MLflow Tracking.

El registro formal en Model Registry / Models in Unity Catalog es un paso de gobierno
empresarial. En un entorno productivo de Azure Databricks normalmente requiere:

- Unity Catalog correctamente configurado.
- Permisos sobre el catálogo y el schema.
- Permiso para crear modelos.
- Permisos de escritura sobre el storage administrado del catálogo.
- Política de gobierno para versionamiento, aliases y promoción de modelos.

Por eso, en Free Edition dejamos el modelo como artifact de MLflow y registramos
la decisión en una tabla de auditoría. Si el instructor está en un workspace empresarial,
puede activar ENABLE_UC_MODEL_REGISTRY = True.
"""

registry_rows = []

model_uri = champion["model_uri"]
registered_model_name_uc = f"{SCHEMA_MODELS}.bagazo_champion_sesion_08"

status = "artifact_only_free_edition"
message = (
    "El modelo champion se conserva como artifact del run de MLflow. "
    "El registro formal en Unity Catalog se deja como paso empresarial, "
    "porque en Free Edition o workspaces sin permisos de storage administrado "
    "puede fallar por permisos de catálogo, schema o ubicación administrada."
)
registered_name_used = registered_model_name_uc

if ENABLE_UC_MODEL_REGISTRY:
    try:
        print(f"Intentando registrar modelo en Unity Catalog como: {registered_model_name_uc}")

        previous_registry_uri = mlflow.get_registry_uri()
        mlflow.set_registry_uri("databricks-uc")

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name_uc
        )

        status = "registered_uc"
        message = (
            f"Modelo registrado correctamente en Unity Catalog: "
            f"{registered_model.name}, versión {registered_model.version}."
        )

        print("✅", message)

        try:
            mlflow.set_registry_uri(previous_registry_uri)
        except Exception:
            pass

    except Exception as e:
        raw_message = str(e)

        if "AccessDenied" in raw_message or "PutObject" in raw_message:
            status = "fallback_artifact_only_storage_permission"
            message = (
                "No fue posible registrar el modelo en Unity Catalog porque el entorno "
                "no tiene permisos suficientes para escribir los artefactos del modelo "
                "en el storage administrado del catálogo. El modelo se conserva como "
                "artifact dentro del run de MLflow."
            )
        elif "PERMISSION_DENIED" in raw_message:
            status = "fallback_artifact_only_permission_denied"
            message = (
                "No fue posible registrar el modelo por permisos insuficientes sobre "
                "Model Registry / Unity Catalog. El modelo se conserva como artifact "
                "dentro del run de MLflow."
            )
        else:
            status = "fallback_artifact_only_registry_error"
            message = (
                "No fue posible registrar el modelo en el registry del workspace. "
                "El modelo se conserva como artifact dentro del run de MLflow. "
                f"Detalle técnico resumido: {raw_message[:500]}"
            )

        print("⚠️ Registro formal no disponible en este entorno.")
        print(message)

        try:
            mlflow.set_registry_uri("databricks")
        except Exception:
            pass

else:
    print("✅ Modelo champion conservado como artifact en MLflow.")
    print("ℹ️ Registro formal omitido para mantener compatibilidad con Free Edition.")
    print("ℹ️ En Azure Databricks empresarial, este paso se activaría con Unity Catalog y permisos adecuados.")

registry_rows.append({
    "run_id": champion["run_id"],
    "model_uri": model_uri,
    "registered_name_attempted": registered_name_used,
    "status": status,
    "message": message,
    "fecha_ejecucion": pd.Timestamp.now()
})

registry_log_pdf = pd.DataFrame(registry_rows)

(
    spark.createDataFrame(registry_log_pdf)
    .write
    .mode("overwrite")
    .format("delta")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_REGISTRY_LOG)
)

print(f"✅ Log de registro creado: {TABLE_REGISTRY_LOG}")
display(spark.table(TABLE_REGISTRY_LOG))


# COMMAND ----------

# ==========================================================
# 11. Guardar predicciones holdout del champion
# ==========================================================
champion_predictions = predictions_by_run[champion["run_id"]].copy()
champion_predictions["fecha"] = pd.to_datetime(champion_predictions["fecha"]).dt.date
champion_predictions["fecha_ejecucion"] = pd.to_datetime(champion_predictions["fecha_ejecucion"])

spark.createDataFrame(champion_predictions).write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable(TABLE_HOLDOUT)
print(f"✅ Predicciones holdout creadas: {TABLE_HOLDOUT}")
display(spark.table(TABLE_HOLDOUT).orderBy("fecha", "ingenio").limit(20))


# COMMAND ----------

# ==========================================================
# 12. Guía para explorar la interfaz MLflow y leer resultados
# ==========================================================
registry_status = spark.table(TABLE_REGISTRY_LOG).select("status").collect()[0]["status"]

print(f"""
🔎 Demo sugerida en la UI de MLflow

1. Abre el panel izquierdo de Databricks.
2. Entra a Experiments.
3. Busca el experimento de la sesión:
   {EXPERIMENT_PATH}
4. Compara los runs por MAE, RMSE, R2 y SMAPE.
5. Abre el run champion.
6. Revisa:
   - Parameters: configuración del entrenamiento.
   - Metrics: desempeño del modelo.
   - Artifacts: plots, CSVs, model card y configuración.
   - Model: modelo sklearn loggeado como artifact reutilizable.

📌 Lectura del champion observado

- MAE: {champion['mae']:.2f} toneladas.
  Significa que el error absoluto promedio del modelo es cercano a ese valor.

- RMSE: {champion['rmse']:.2f} toneladas.
  Penaliza más los errores grandes; si es bastante mayor que MAE, hay días difíciles o extremos.

- R2: {champion['r2']:.3f}.
  Indica qué tanto de la variabilidad del holdout logra capturar el modelo.

- SMAPE: {champion['smape']:.2f}%.
  Debe interpretarse con cuidado cuando existen días con valores reales bajos.

📌 Estado del registro del modelo

Estado actual: {registry_status}

En Free Edition es suficiente que el modelo quede loggeado como artifact del run de MLflow.
En un entorno empresarial de Azure Databricks, el siguiente paso sería registrarlo en Unity Catalog,
asignarle un alias como Champion y usarlo como entrada para inferencia batch o serving.

Mensaje clave: MLflow no es solo guardar números. Es dejar evidencia reproducible del experimento.
""")


# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO 3 — Conclusión sobre leakage
# MAGIC
# MAGIC ¿Por qué un modelo con mejor MAE puede ser menos adecuado si usa variables que no están disponibles al momento de predecir?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mini caso Lumi opcional: mala experiencia del cliente
# MAGIC
# MAGIC Este bloque queda como conversación o reto opcional. No desplaza el caso principal de Bagazo.
# MAGIC
# MAGIC **Pregunta:** ¿podríamos predecir una mala experiencia del cliente?
# MAGIC
# MAGIC **Fuente sugerida:** `workspace.lumi_gold.fact_delivery_experience`  
# MAGIC **Target sugerido:** `mala_experiencia = review_score <= 2`
# MAGIC
# MAGIC No se usa pagos como caso principal porque en la Sesión 7 se observó `valor_pagado_total = 0.0` en `kpi_payment_methods`.
# MAGIC

# COMMAND ----------

# ==========================================================
# 13. Mini caso Lumi opcional (no ejecutar si el tiempo es corto)
# ==========================================================
LUMI_DELIVERY = f"{CATALOG}.lumi_gold.fact_delivery_experience"
if table_exists(LUMI_DELIVERY):
    lumi = spark.table(LUMI_DELIVERY)
    print("✅ Fuente Lumi disponible")
    print(lumi.columns)
    # Ejemplo conceptual: ajustar nombres si difieren.
    # target sugerido: mala_experiencia = review_score <= 2
else:
    print("⚠️ No se encontró fact_delivery_experience. Saltar mini caso Lumi.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO 4 — Variable adicional de negocio
# MAGIC
# MAGIC Propón una variable que mejoraría el modelo: humedad, mantenimiento, transporte, inventario, demanda energética u otra.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO 5 — Regla de monitoreo para Sesión 9
# MAGIC
# MAGIC Diseña una regla simple: por ejemplo, alertar si el MAE semanal supera X toneladas o si el error absoluto promedio crece 20%.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Retos de cierre
# MAGIC
# MAGIC ## Reto 1 — Interpretar un MLflow Run
# MAGIC Abre el run champion y explica en tus palabras qué significan parámetros, métricas, artefactos y modelo.
# MAGIC
# MAGIC ## Reto 2 — Mejorar features
# MAGIC Agrega una feature temporal adicional, reentrena un modelo y compara contra el champion.
# MAGIC
# MAGIC ## Reto 3 — Clasificación de riesgo bajo de bagazo
# MAGIC Entrena un `RandomForestClassifier` para `target_riesgo_bajo_bagazo` y analiza precision vs recall.
# MAGIC
# MAGIC ## Reto consultor — Model Card ejecutivo
# MAGIC Completa una recomendación ejecutiva para el modelo champion.
# MAGIC

# COMMAND ----------

# ==============================================================================
# RETO 2: Agregar variable cíclica estacional y comparar en MLflow
# ==============================================================================
X_train_r2 = X_train_demo.copy()
X_test_r2 = X_test_demo.copy()

# Crear transformaciones cíclicas basadas en el número de mes
X_train_r2["mes_sin"] = np.sin(2 * np.pi * X_train_r2["mes"] / 12.0)
X_test_r2["mes_sin"] = np.sin(2 * np.pi * X_test_r2["mes"] / 12.0)

run_name_r2 = "B_forecast_sin_leakage__random_forest_features_reto2"

with mlflow.start_run(run_name=run_name_r2) as run_r2:
    # Instanciar el mismo RandomForest del champion
    rf_reto2 = RandomForestRegressor(
        n_estimators=120, max_depth=7, min_samples_leaf=3, random_state=42, n_jobs=1
    )
    rf_reto2.fit(X_train_r2, y_train_demo)
    preds_r2 = rf_reto2.predict(X_test_r2)
    metrics_r2 = evaluate_regression(y_test_demo, preds_r2)
    
    # Loggear parámetros y nuevas métricas
    mlflow.log_param("modelo", "random_forest_reto2")
    mlflow.log_param("escenario", "B_forecast_sin_leakage")
    mlflow.log_metrics(metrics_r2)
    mlflow.sklearn.log_model(rf_reto2, artifact_path="model")

print(f"✅ Reto 2 completado de forma exitosa.")
print(f"Métricas del nuevo Run vs Champion: MAE={metrics_r2['mae']:.2f} | RMSE={metrics_r2['rmse']:.2f} | R2={metrics_r2['r2']:.3f}")

# COMMAND ----------

# ==============================================================================
# RETO 3: Clasificación con RandomForestClassifier para Target Riesgo Bajo
# ==============================================================================
# Preparar datasets de clasificación usando el split temporal de holdout previo
y_train_cls = spark.table(TABLE_FEATURES).toPandas()["target_riesgo_bajo_bagazo"].iloc[:len(X_train_demo)].copy()
y_test_cls = spark.table(TABLE_FEATURES).toPandas()["target_riesgo_bajo_bagazo"].iloc[len(X_train_demo):].copy()

run_name_cls = "B_forecast_sin_leakage__rf_classifier_reto3"

with mlflow.start_run(run_name=run_name_cls) as run_cls:
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
    clf.fit(X_train_demo, y_train_cls)
    preds_cls = clf.predict(X_demo_test := X_test_demo)
    
    # Calcular métricas de clasificación
    acc = accuracy_score(y_test_cls, preds_cls)
    prec = precision_score(y_test_cls, preds_cls, zero_division=0)
    rec = recall_score(y_test_cls, preds_cls, zero_division=0)
    f1 = f1_score(y_test_cls, preds_cls, zero_division=0)
    
    # Enviar a MLflow Tracking
    mlflow.log_param("tipo_modelo", "random_forest_classifier")
    mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
    mlflow.sklearn.log_model(clf, artifact_path="model")

print(f"✅ Reto 3 de clasificación finalizado.")
print(f"Resultados en test: Precision={prec:.2f} (Evita falsas alarmas) | Recall={rec:.2f} (Efectividad capturando alertas)")

# COMMAND ----------

# MAGIC %md
# MAGIC Nombre del modelo asignado: B_forecast_sin_leakage__random_forest.Propósito analítico verificado: Estimar el volumen diario de entrega de bagazo para predecir la disponibilidad de biomasa energética con un horizonte de anticipación, mitigando riesgos de desabastecimiento.
# MAGIC
# MAGIC Métricas de validación en Holdout: Error Absoluto Medio (MAE) de 97.01 toneladas, Raíz del Error Cuadrático Medio (RMSE) de 132.89 toneladas y un Coeficiente de Determinación ($R^2$) de 0.733. El modelo es capaz de explicar el 73.3% de la variabilidad del comportamiento de las entregas históricas en escenarios futuros no vistos.
# MAGIC
# MAGIC Justificación de selección como Champion: A pesar de que el modelo operativo (Nowcasting) arrojó un error menor (MAE de 84.71), se descarta formalmente debido al alto riesgo de interrupción operacional que provoca el data leakage. El Random Forest de Forecast es seleccionado como el verdadero Champion institucional porque basa sus decisiones exclusivamente en variables exógenas climáticas del día actual y en el comportamiento rezagado (lags) de los 7 días anteriores, permitiendo una planificación logística real y anticipada sin depender de registros del presente inmediato.
# MAGIC
# MAGIC Recomendación estratégica de MLOps: El modelo demuestra la madurez matemática suficiente para avanzar a la fase de industrialización en la Sesión 9. Se recomienda integrarlo como un proceso de inferencia batch semanal automatizado, cuyos resultados alimenten una tabla de monitoreo continuo. Esto permitirá comparar las desviaciones contra el MAE base de 97 toneladas para detectar tempranamente fenómenos de degradación del modelo (data drift) provocados por cambios imprevistos en la estacionalidad climática o alteraciones en los patrones de molienda de los ingenios.

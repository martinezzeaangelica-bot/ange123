# Databricks notebook source
# MAGIC %md
# MAGIC # Sesión 4 — Limpieza, calidad y capa Silver
# MAGIC
# MAGIC **Capacitación práctica Databricks — iData Academy**  
# MAGIC Proyecto principal: **Lumi Commerce Lakehouse**  
# MAGIC Caso espejo empresarial: **Lluvia, Caña y Bagazo**
# MAGIC
# MAGIC ## Propósito de la sesión
# MAGIC
# MAGIC En la Sesión 3 exploramos la capa **Bronze** con PySpark y detectamos problemas reales de calidad: fechas como texto, pagos con múltiples registros por pedido, categorías sin estandarizar, números regionales, ceros operativos y comentarios que deben convertirse en señales analíticas.
# MAGIC
# MAGIC En esta sesión vamos a construir la capa **Silver**: datos limpios, tipados, estandarizados, auditables y listos para alimentar SQL de negocio, dashboards, KPIs y futuros modelos de machine learning.
# MAGIC
# MAGIC > Bronze es la bodega donde llega todo. Silver es el laboratorio de calidad donde revisamos, corregimos, tipamos, etiquetamos y dejamos evidencia antes de entregar datos a negocio.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Continuidad con la Sesión 3
# MAGIC
# MAGIC Este notebook asume que ya existen las tablas Bronze:
# MAGIC
# MAGIC ```text
# MAGIC workspace.lumi_bronze.*
# MAGIC workspace.bagazo_bronze.molienda_bagazo_y_lluvias_II
# MAGIC ```
# MAGIC
# MAGIC También retoma los hallazgos reales de la Sesión 3:
# MAGIC
# MAGIC - Lumi tiene tablas transaccionales de clientes, pedidos, pagos, productos, reseñas, vendedores y geografía.
# MAGIC - Bagazo tiene 798 filas originales y debe convertirse a formato largo: 798 días × 3 ingenios = 2,394 filas.
# MAGIC - Se detectaron columnas con error de codificación como `caa_molida...`.
# MAGIC - Se validó la necesidad de convertir números regionales como `1.486,2`.
# MAGIC - Los comentarios operativos se transformarán en banderas analíticas.

# COMMAND ----------

# =========================================
# 1. Parámetros de trabajo
# =========================================

CATALOG = "workspace"

LUMI_BRONZE_SCHEMA = "lumi_bronze"
BAGAZO_BRONZE_SCHEMA = "bagazo_bronze"

LUMI_SILVER_SCHEMA = "lumi_silver"
BAGAZO_SILVER_SCHEMA = "bagazo_silver"
CONTROL_SCHEMA = "control"

BAGAZO_TABLE = "molienda_bagazo_y_lluvias_II"
QUALITY_TABLE = "quality_summary_sesion_04"

# TODO pedagógico 1: discute con el grupo si 20 mm es un buen umbral para lluvia alta.
LLUVIA_ALTA_UMBRAL = 20.0
RIESGO_BAJO_BAGAZO_RATIO = 0.05

print("✅ Parámetros cargados")
print(f"Catálogo: {CATALOG}")
print(f"Silver Lumi: {CATALOG}.{LUMI_SILVER_SCHEMA}")
print(f"Silver Bagazo: {CATALOG}.{BAGAZO_SILVER_SCHEMA}")

# COMMAND ----------

# =========================================
# 2. Imports y funciones auxiliares
# =========================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, TimestampType
from datetime import datetime


def table_name(schema: str, table: str) -> str:
    return f"`{CATALOG}`.`{schema}`.`{table}`"


def table_exists(schema: str, table: str) -> bool:
    try:
        return spark.catalog.tableExists(f"{CATALOG}.{schema}.{table}")
    except Exception:
        return False


def show_df(df, n=10):
    try:
        display(df.limit(n))
    except NameError:
        df.show(n, truncate=False)


def print_shape(name, df):
    print(f"{name}: {df.count():,} filas x {len(df.columns)} columnas")


def write_managed_delta(df, schema: str, table: str):
    full_name = table_name(schema, table)
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_name)
    )
    print(f"✅ Tabla escrita: {full_name}")


def safe_double_es(col_name: str):
    # Convierte "1.486,2" -> 1486.2, "1486,2" -> 1486.2 y conserva formato decimal estándar.
    texto = F.trim(F.col(col_name).cast("string"))
    limpio = F.regexp_replace(texto, r"[^0-9,.\-]", "")

    normalizado = (
        F.when(limpio.rlike(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$"), F.regexp_replace(F.regexp_replace(limpio, r"\.", ""), ",", "."))
         .when(limpio.rlike(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$"), F.regexp_replace(limpio, ",", ""))
         .when(limpio.rlike(r"^-?\d+,\d+$"), F.regexp_replace(limpio, ",", "."))
         .otherwise(limpio)
    )

    numero_seguro = (
        F.when(texto.isNull() | (texto == "") | limpio.isin("", "-", ",", ".", "-,", "-."), F.lit(None))
         .when(normalizado.rlike(r"^-?\d+(\.\d+)?$"), normalizado)
         .otherwise(F.lit(None))
    )
    return numero_seguro.cast("double")


def clean_text_basic(col_name: str):
    c = F.trim(F.col(col_name).cast("string"))
    c = F.regexp_replace(c, "ca�a", "caña")
    c = F.regexp_replace(c, "a�o", "año")
    c = F.regexp_replace(c, "recepci�n", "recepción")
    return c

print("✅ Funciones auxiliares cargadas")

# COMMAND ----------

# =========================================
# 3. Crear schemas Silver y Control
# =========================================

for schema in [LUMI_SILVER_SCHEMA, BAGAZO_SILVER_SCHEMA, CONTROL_SCHEMA]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{schema}`")
    print(f"✅ Schema disponible: {CATALOG}.{schema}")

# COMMAND ----------

# =========================================
# 4. Validar y cargar tablas Bronze
# =========================================

required_lumi_tables = [
    "customers", "geolocation", "order_items", "order_payments", "order_reviews",
    "orders", "products", "sellers", "product_category_translation"
]

missing = []
for table in required_lumi_tables:
    if not table_exists(LUMI_BRONZE_SCHEMA, table):
        missing.append(table_name(LUMI_BRONZE_SCHEMA, table))

if not table_exists(BAGAZO_BRONZE_SCHEMA, BAGAZO_TABLE):
    missing.append(table_name(BAGAZO_BRONZE_SCHEMA, BAGAZO_TABLE))

if missing:
    raise ValueError("Faltan tablas Bronze. Ejecuta primero la sesión de ingesta Bronze: " + ", ".join(missing))

customers_bz = spark.table(table_name(LUMI_BRONZE_SCHEMA, "customers"))
geolocation_bz = spark.table(table_name(LUMI_BRONZE_SCHEMA, "geolocation"))
order_items_bz = spark.table(table_name(LUMI_BRONZE_SCHEMA, "order_items"))
order_payments_bz = spark.table(table_name(LUMI_BRONZE_SCHEMA, "order_payments"))
order_reviews_bz = spark.table(table_name(LUMI_BRONZE_SCHEMA, "order_reviews"))
orders_bz = spark.table(table_name(LUMI_BRONZE_SCHEMA, "orders"))
products_bz = spark.table(table_name(LUMI_BRONZE_SCHEMA, "products"))
sellers_bz = spark.table(table_name(LUMI_BRONZE_SCHEMA, "sellers"))
translation_bz = spark.table(table_name(LUMI_BRONZE_SCHEMA, "product_category_translation"))
bagazo_bz = spark.table(table_name(BAGAZO_BRONZE_SCHEMA, BAGAZO_TABLE))

print("✅ Tablas Bronze cargadas")
print_shape("orders_bz", orders_bz)
print_shape("bagazo_bz", bagazo_bz)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Construcción de Lumi Silver
# MAGIC
# MAGIC La meta de Silver para Lumi no es construir KPIs finales todavía. La meta es dejar cada tabla con estructura confiable: fechas convertidas, montos tipados, pagos consolidados, categorías traducidas, banderas analíticas y validaciones mínimas.

# COMMAND ----------

# =========================================
# 5. orders_clean
# =========================================

orders_clean = (
    orders_bz
    .select(
        "order_id", "customer_id",
        F.lower(F.trim(F.col("order_status"))).alias("order_status"),
        F.to_timestamp("order_purchase_timestamp").alias("order_purchase_ts"),
        F.to_timestamp("order_approved_at").alias("order_approved_ts"),
        F.to_timestamp("order_delivered_carrier_date").alias("order_delivered_carrier_ts"),
        F.to_timestamp("order_delivered_customer_date").alias("order_delivered_customer_ts"),
        F.to_timestamp("order_estimated_delivery_date").alias("order_estimated_delivery_ts")
    )
    .withColumn("order_purchase_date", F.to_date("order_purchase_ts"))
    .withColumn("year_month", F.date_format("order_purchase_date", "yyyy-MM"))
    .withColumn("delivery_days", F.when(F.col("order_delivered_customer_ts").isNotNull() & F.col("order_purchase_ts").isNotNull(), F.datediff(F.to_date("order_delivered_customer_ts"), F.to_date("order_purchase_ts"))))
    .withColumn("delay_days", F.when(F.col("order_delivered_customer_ts").isNotNull() & F.col("order_estimated_delivery_ts").isNotNull(), F.greatest(F.datediff(F.to_date("order_delivered_customer_ts"), F.to_date("order_estimated_delivery_ts")), F.lit(0))))
    .withColumn("is_late", F.coalesce(F.col("delay_days") > 0, F.lit(False)))
    .withColumn("is_delivered", F.col("order_status") == F.lit("delivered"))
    .withColumn("has_missing_purchase_ts", F.col("order_purchase_ts").isNull())
)

show_df(orders_clean, 5)
print_shape("orders_clean", orders_clean)

# COMMAND ----------

# =========================================
# 6. order_items_clean
# =========================================

order_items_clean = (
    order_items_bz
    .select(
        "order_id",
        F.col("order_item_id").cast("int").alias("order_item_id"),
        "product_id", "seller_id",
        F.to_timestamp("shipping_limit_date").alias("shipping_limit_ts"),
        F.col("price").cast("double").alias("price"),
        F.col("freight_value").cast("double").alias("freight_value")
    )
    .withColumn("total_item_value", F.round(F.coalesce(F.col("price"), F.lit(0.0)) + F.coalesce(F.col("freight_value"), F.lit(0.0)), 2))
    .withColumn("has_price_issue", F.col("price").isNull() | (F.col("price") < 0))
    .withColumn("has_freight_issue", F.col("freight_value").isNull() | (F.col("freight_value") < 0))
)

show_df(order_items_clean, 5)
print_shape("order_items_clean", order_items_clean)

# COMMAND ----------

# =========================================
# 7. payments_clean: consolidar pagos por pedido
# =========================================

payments_base = (
    order_payments_bz
    .select(
        "order_id",
        F.col("payment_sequential").cast("int").alias("payment_sequential"),
        F.lower(F.trim(F.col("payment_type"))).alias("payment_type"),
        F.col("payment_installments").cast("int").alias("payment_installments"),
        F.col("payment_value").cast("double").alias("payment_value")
    )
)

payment_type_counts = (
    payments_base
    .groupBy("order_id", "payment_type")
    .agg(F.count("*").alias("payment_type_rows"), F.sum("payment_value").alias("payment_type_value"))
    .withColumn("rn", F.row_number().over(Window.partitionBy("order_id").orderBy(F.desc("payment_type_rows"), F.desc("payment_type_value"), F.asc("payment_type"))))
    .filter(F.col("rn") == 1)
    .select("order_id", F.col("payment_type").alias("main_payment_type"))
)

payments_clean = (
    payments_base
    .groupBy("order_id")
    .agg(
        F.round(F.sum("payment_value"), 2).alias("total_payment_value"),
        F.max("payment_installments").alias("payment_installments_max"),
        F.countDistinct("payment_type").alias("payment_methods_count"),
        F.count("*").alias("payment_rows"),
        F.max(F.when(F.col("payment_type") == "not_defined", F.lit(1)).otherwise(F.lit(0))).alias("has_not_defined_payment")
    )
    .join(payment_type_counts, on="order_id", how="left")
    .withColumn("has_payment_quality_issue", F.col("has_not_defined_payment") == 1)
)

show_df(payments_clean, 5)
print_shape("payments_clean", payments_clean)

# COMMAND ----------

# =========================================
# 8. products_clean: traducción y categoría limpia
# =========================================

products_clean = (
    products_bz.alias("p")
    .join(translation_bz.alias("t"), on="product_category_name", how="left")
    .select(
        F.col("p.product_id"),
        F.lower(F.trim(F.col("p.product_category_name"))).alias("product_category_name"),
        F.lower(F.trim(F.col("t.product_category_name_english"))).alias("product_category_name_english"),
        F.col("p.product_name_lenght").cast("int").alias("product_name_length"),
        F.col("p.product_description_lenght").cast("int").alias("product_description_length"),
        F.col("p.product_photos_qty").cast("int").alias("product_photos_qty"),
        F.col("p.product_weight_g").cast("double").alias("product_weight_g"),
        F.col("p.product_length_cm").cast("double").alias("product_length_cm"),
        F.col("p.product_height_cm").cast("double").alias("product_height_cm"),
        F.col("p.product_width_cm").cast("double").alias("product_width_cm")
    )
    .withColumn("product_category_clean", F.coalesce(F.col("product_category_name_english"), F.col("product_category_name"), F.lit("sin_categoria")))
    .withColumn("product_category_clean", F.when(F.col("product_category_clean") == "", F.lit("sin_categoria")).otherwise(F.col("product_category_clean")))
    .withColumn("has_missing_category", F.col("product_category_name").isNull() | (F.col("product_category_clean") == "sin_categoria"))
)

show_df(products_clean, 5)
print_shape("products_clean", products_clean)

# COMMAND ----------

# =========================================
# 9. reviews_clean
# =========================================

reviews_clean = (
    order_reviews_bz
    .select(
        "review_id", "order_id",
        F.col("review_score").cast("int").alias("review_score"),
        F.trim(F.col("review_comment_title")).alias("review_comment_title"),
        F.trim(F.col("review_comment_message")).alias("review_comment_message"),
        F.to_timestamp("review_creation_date").alias("review_creation_ts"),
        F.to_timestamp("review_answer_timestamp").alias("review_answer_ts")
    )
    .withColumn("review_creation_date", F.to_date("review_creation_ts"))
    .withColumn("is_low_review", F.col("review_score") <= 2)
    .withColumn("has_review_comment", (F.length(F.coalesce(F.col("review_comment_title"), F.lit(""))) > 0) | (F.length(F.coalesce(F.col("review_comment_message"), F.lit(""))) > 0))
    .withColumn("has_review_score_issue", F.col("review_score").isNull() | ~F.col("review_score").between(1, 5))
)

show_df(reviews_clean, 5)
print_shape("reviews_clean", reviews_clean)

# COMMAND ----------

# =========================================
# 10. customers_clean, sellers_clean y geolocation_clean
# =========================================

customers_clean = (
    customers_bz
    .select(
        "customer_id", "customer_unique_id",
        F.col("customer_zip_code_prefix").cast("int").alias("customer_zip_code_prefix"),
        F.lower(F.trim(F.col("customer_city"))).alias("customer_city"),
        F.upper(F.trim(F.col("customer_state"))).alias("customer_state")
    )
)

sellers_clean = (
    sellers_bz
    .select(
        "seller_id",
        F.col("seller_zip_code_prefix").cast("int").alias("seller_zip_code_prefix"),
        F.lower(F.trim(F.col("seller_city"))).alias("seller_city"),
        F.upper(F.trim(F.col("seller_state"))).alias("seller_state")
    )
)

geolocation_clean = (
    geolocation_bz
    .select(
        F.col("geolocation_zip_code_prefix").cast("int").alias("geolocation_zip_code_prefix"),
        F.col("geolocation_lat").cast("double").alias("geolocation_lat"),
        F.col("geolocation_lng").cast("double").alias("geolocation_lng"),
        F.lower(F.trim(F.col("geolocation_city"))).alias("geolocation_city"),
        F.upper(F.trim(F.col("geolocation_state"))).alias("geolocation_state")
    )
)

print_shape("customers_clean", customers_clean)
print_shape("sellers_clean", sellers_clean)
print_shape("geolocation_clean", geolocation_clean)
show_df(customers_clean, 5)

# COMMAND ----------

# =========================================
# 11. Escribir tablas Lumi Silver
# =========================================

lumi_silver_outputs = {
    "customers_clean": customers_clean,
    "orders_clean": orders_clean,
    "order_items_clean": order_items_clean,
    "payments_clean": payments_clean,
    "reviews_clean": reviews_clean,
    "products_clean": products_clean,
    "sellers_clean": sellers_clean,
    "geolocation_clean": geolocation_clean,
}

for table, df in lumi_silver_outputs.items():
    write_managed_delta(df, LUMI_SILVER_SCHEMA, table)

print("✅ Capa Lumi Silver escrita")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Construcción de Bagazo Silver
# MAGIC
# MAGIC Bagazo es el caso espejo empresarial. Aquí la limpieza no es solo técnica: también es interpretación operativa.
# MAGIC
# MAGIC Un cero en caña o bagazo puede significar mantenimiento, paro, falta de caña por lluvia, no recepción de bagazo o un posible problema de dato. Por eso conservamos el valor y creamos banderas de contexto.

# COMMAND ----------

# =========================================
# 12. Preparar Bagazo: renombrar columnas y descartar columna sin nombre
# =========================================

bagazo_base = bagazo_bz

rename_map = {
    "caa_molida_providencia_toneladas": "cana_molida_providencia_toneladas",
    "caa_molida_ilc_toneladas": "cana_molida_ilc_toneladas",
    "caa_molida_incauca_toneladas": "cana_molida_incauca_toneladas",
}

for old_col, new_col in rename_map.items():
    if old_col in bagazo_base.columns and new_col not in bagazo_base.columns:
        bagazo_base = bagazo_base.withColumnRenamed(old_col, new_col)

if "columna_sin_nombre" in bagazo_base.columns:
    bagazo_base = bagazo_base.drop("columna_sin_nombre")

print("✅ Columnas Bagazo preparadas")
print(bagazo_base.columns)

# COMMAND ----------

# =========================================
# 13. Bagazo: formato ancho -> formato largo
# =========================================

bagazo_wide = (
    bagazo_base
    .withColumn("fecha", F.to_date("fecha"))
    .withColumn("comentarios_providencia", clean_text_basic("comentarios_providencia"))
    .withColumn("comentarios_ilc", clean_text_basic("comentarios_ilc"))
    .withColumn("comentarios_incauca", clean_text_basic("comentarios_incauca"))
)

bagazo_long_raw = bagazo_wide.selectExpr(
    "fecha",
    """
    stack(
        3,
 
        'Providencia',
        CAST(`promedio_lluvias_providencia_mm` AS STRING),
        CAST(`cana_molida_providencia_toneladas` AS STRING),
        CAST(`bagazo_entregado_providencia_toneladas` AS STRING),
        CAST(`comentarios_providencia` AS STRING),
 
        'ILC',
        CAST(`promedio_lluvias_ilc_mm` AS STRING),
        CAST(`cana_molida_ilc_toneladas` AS STRING),
        CAST(`bagazo_entregado_ilc_toneladas` AS STRING),
        CAST(`comentarios_ilc` AS STRING),
 
        'Incauca',
        CAST(`promedio_lluvias_incauca_mm` AS STRING),
        CAST(`cana_molida_incauca_toneladas` AS STRING),
        CAST(`bagazo_entregado_incauca_toneladas` AS STRING),
        CAST(`comentarios_incauca` AS STRING)
 
    ) AS (ingenio, lluvia_raw, cana_raw, bagazo_raw, comentario)
    """
)

bagazo_long_clean = (
    bagazo_long_raw
    .withColumn("lluvia_mm", safe_double_es("lluvia_raw"))
    .withColumn("cana_molida_ton", safe_double_es("cana_raw"))
    .withColumn("bagazo_entregado_ton", safe_double_es("bagazo_raw"))
    .drop("lluvia_raw", "cana_raw", "bagazo_raw")
    .withColumn("comentario", F.when(F.length(F.trim(F.col("comentario"))) == 0, F.lit(None)).otherwise(F.trim(F.col("comentario"))))
    .withColumn("anio", F.year("fecha"))
    .withColumn("mes", F.month("fecha"))
    .withColumn("year_month", F.date_format("fecha", "yyyy-MM"))
    .withColumn("dia_semana", F.date_format("fecha", "E"))
    .withColumn("tiene_comentario_operativo", F.col("comentario").isNotNull())
)

show_df(bagazo_long_clean, 10)
print_shape("bagazo_long_clean", bagazo_long_clean)

# COMMAND ----------

# =========================================
# 14. Bagazo: crear banderas operativas y de calidad
# =========================================

comentario_norm = F.lower(F.coalesce(F.col("comentario"), F.lit("")))

bagazo_silver = (
    bagazo_long_clean
    .withColumn("es_mantenimiento", comentario_norm.contains("mantenimiento"))
    .withColumn("es_falta_cana_por_lluvia", comentario_norm.contains("falta") & comentario_norm.contains("lluvia"))
    .withColumn("es_paro", comentario_norm.contains("paro"))
    .withColumn("es_sin_recepcion_bagazo", comentario_norm.contains("no se recibió bagazo") | comentario_norm.contains("no se recibio bagazo") | comentario_norm.contains("sin recepción") | comentario_norm.contains("sin recepcion"))
    .withColumn("cana_cero", F.coalesce(F.col("cana_molida_ton"), F.lit(0.0)) == 0.0)
    .withColumn("bagazo_cero", F.coalesce(F.col("bagazo_entregado_ton"), F.lit(0.0)) == 0.0)
    .withColumn("lluvia_alta", F.col("lluvia_mm") >= F.lit(LLUVIA_ALTA_UMBRAL))
    .withColumn("riesgo_bajo_bagazo", (F.col("cana_molida_ton") > 0) & (F.col("bagazo_entregado_ton") >= 0) & ((F.col("bagazo_entregado_ton") / F.col("cana_molida_ton")) < F.lit(RIESGO_BAJO_BAGAZO_RATIO)))
    .select(
        "fecha", "ingenio", "lluvia_mm", "cana_molida_ton", "bagazo_entregado_ton", "comentario",
        "anio", "mes", "year_month", "dia_semana", "tiene_comentario_operativo",
        "es_mantenimiento", "es_falta_cana_por_lluvia", "es_paro", "es_sin_recepcion_bagazo",
        "cana_cero", "bagazo_cero", "lluvia_alta", "riesgo_bajo_bagazo"
    )
)

show_df(bagazo_silver, 10)
print_shape("bagazo_silver", bagazo_silver)

# COMMAND ----------

# =========================================
# 15. Validación rápida de conversión numérica Bagazo
# =========================================

validacion_bagazo = bagazo_silver.agg(
    F.count("*").alias("registros"),
    F.sum(F.when(F.col("fecha").isNull(), 1).otherwise(0)).alias("fechas_nulas"),
    F.sum(F.when(F.col("lluvia_mm").isNull(), 1).otherwise(0)).alias("lluvia_no_convertida"),
    F.sum(F.when(F.col("cana_molida_ton").isNull(), 1).otherwise(0)).alias("cana_no_convertida"),
    F.sum(F.when(F.col("bagazo_entregado_ton").isNull(), 1).otherwise(0)).alias("bagazo_no_convertido"),
)

show_df(validacion_bagazo, 1)
print("✅ Validación numérica de Bagazo completada")

# COMMAND ----------

# =========================================
# 16. Escribir Bagazo Silver
# =========================================

write_managed_delta(bagazo_silver, BAGAZO_SILVER_SCHEMA, "operacion_ingenios_clean")
print("✅ Capa Bagazo Silver escrita")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tabla de control de calidad
# MAGIC
# MAGIC Silver no solo transforma datos: también deja evidencia. La tabla `workspace.control.quality_summary_sesion_04` resume controles mínimos por tabla para que el equipo pueda revisar riesgos antes de construir SQL, Gold o dashboards.

# COMMAND ----------

# =========================================
# 17. Funciones para resumen de calidad
# =========================================

def count_nulls(df, critical_cols):
    if not critical_cols:
        return 0
    exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in critical_cols]
    row = df.agg(*exprs).collect()[0].asDict()
    return int(sum(v or 0 for v in row.values()))


def count_duplicates(df, key_cols):
    if not key_cols:
        return 0
    return df.groupBy(*key_cols).agg(F.count("*").alias("n")).filter(F.col("n") > 1).count()


def count_failed_rules(df, rules):
    total = 0
    for rule_name, condition in rules.items():
        failed = df.filter(condition).count()
        print(f"Regla {rule_name}: {failed} registros fallidos")
        total += failed
    return int(total)


def quality_row(dataset, table, capa, df, critical_cols=None, key_cols=None, rules=None, observaciones=""):
    critical_cols = critical_cols or []
    key_cols = key_cols or []
    rules = rules or {}
    filas = df.count()
    columnas = len(df.columns)
    nulos_criticos = count_nulls(df, critical_cols)
    duplicados_clave = count_duplicates(df, key_cols)
    reglas_fallidas = count_failed_rules(df, rules)
    estado = "OK" if nulos_criticos == 0 and duplicados_clave == 0 and reglas_fallidas == 0 else "REVISAR"
    return (dataset, table, capa, int(filas), int(columnas), int(nulos_criticos), int(duplicados_clave), int(reglas_fallidas), estado, datetime.now(), observaciones)

print("✅ Funciones de calidad cargadas")

# COMMAND ----------

# =========================================
# 18. Construir quality_summary_sesion_04
# =========================================

# TODO pedagógico 2: agrega una regla adicional si quieres reforzar el contrato de calidad.
# Ejemplo sugerido: dimensiones o peso negativos en products_clean.

quality_rows = []

quality_rows.append(quality_row("Lumi", "orders_clean", "silver", orders_clean, ["order_id", "customer_id", "order_purchase_date"], ["order_id"], {"delivery_days_negative": F.col("delivery_days") < 0}, "Fechas tipadas, banderas de entrega y demora creadas."))
quality_rows.append(quality_row("Lumi", "order_items_clean", "silver", order_items_clean, ["order_id", "product_id", "seller_id"], ["order_id", "order_item_id"], {"negative_price_or_freight": (F.col("price") < 0) | (F.col("freight_value") < 0)}, "Montos tipados y valor total de item calculado."))
quality_rows.append(quality_row("Lumi", "payments_clean", "silver", payments_clean, ["order_id", "total_payment_value"], ["order_id"], {"negative_payment_value": F.col("total_payment_value") < 0}, "Pagos consolidados por orden; not_defined queda marcado como calidad."))
quality_rows.append(quality_row("Lumi", "products_clean", "silver", products_clean, ["product_id", "product_category_clean"], ["product_id"], {}, "Categorías traducidas y nulos manejados como sin_categoria."))
quality_rows.append(quality_row("Lumi", "reviews_clean", "silver", reviews_clean, ["review_id", "order_id", "review_score"], ["review_id"], {"review_score_out_of_range": ~F.col("review_score").between(1, 5)}, "Score conservado, banderas de baja calificación y comentario creadas."))
quality_rows.append(quality_row("Bagazo", "operacion_ingenios_clean", "silver", bagazo_silver, ["fecha", "ingenio", "lluvia_mm", "cana_molida_ton", "bagazo_entregado_ton"], ["fecha", "ingenio"], {"negative_lluvia": F.col("lluvia_mm") < 0, "negative_cana": F.col("cana_molida_ton") < 0, "negative_bagazo": F.col("bagazo_entregado_ton") < 0, "invalid_ingenio": ~F.col("ingenio").isin("Providencia", "ILC", "Incauca")}, "Formato largo, números regionales normalizados y banderas operativas creadas."))

quality_schema = StructType([
    StructField("dataset", StringType(), False),
    StructField("tabla", StringType(), False),
    StructField("capa", StringType(), False),
    StructField("filas", LongType(), False),
    StructField("columnas", IntegerType(), False),
    StructField("nulos_criticos", LongType(), False),
    StructField("duplicados_clave", LongType(), False),
    StructField("reglas_fallidas", LongType(), False),
    StructField("estado_calidad", StringType(), False),
    StructField("fecha_validacion", TimestampType(), False),
    StructField("observaciones", StringType(), True),
])

quality_summary_df = spark.createDataFrame(quality_rows, schema=quality_schema)
write_managed_delta(quality_summary_df, CONTROL_SCHEMA, QUALITY_TABLE)

show_df(quality_summary_df, 20)
print("✅ Tabla de control de calidad creada")

# COMMAND ----------

# =========================================
# 19. Validación posterior a la escritura
# =========================================

silver_tables_to_validate = [
    (LUMI_SILVER_SCHEMA, "customers_clean"),
    (LUMI_SILVER_SCHEMA, "orders_clean"),
    (LUMI_SILVER_SCHEMA, "order_items_clean"),
    (LUMI_SILVER_SCHEMA, "payments_clean"),
    (LUMI_SILVER_SCHEMA, "reviews_clean"),
    (LUMI_SILVER_SCHEMA, "products_clean"),
    (LUMI_SILVER_SCHEMA, "sellers_clean"),
    (LUMI_SILVER_SCHEMA, "geolocation_clean"),
    (BAGAZO_SILVER_SCHEMA, "operacion_ingenios_clean"),
    (CONTROL_SCHEMA, QUALITY_TABLE),
]

validation_rows = []
for schema, table in silver_tables_to_validate:
    df = spark.table(table_name(schema, table))
    validation_rows.append((schema, table, df.count(), len(df.columns)))

validation_df = spark.createDataFrame(validation_rows, ["schema", "table", "filas", "columnas"])
show_df(validation_df, 20)
print("✅ Validación posterior a escritura completada")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- =========================================
# MAGIC -- 20. Consulta SQL rápida sobre Silver
# MAGIC -- =========================================
# MAGIC
# MAGIC SELECT
# MAGIC   estado_calidad,
# MAGIC   COUNT(*) AS tablas
# MAGIC FROM workspace.control.quality_summary_sesion_04
# MAGIC GROUP BY estado_calidad
# MAGIC ORDER BY tablas DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Checkpoint final de la sesión
# MAGIC
# MAGIC Al terminar este notebook deben existir estas tablas:
# MAGIC
# MAGIC ```text
# MAGIC workspace.lumi_silver.customers_clean
# MAGIC workspace.lumi_silver.orders_clean
# MAGIC workspace.lumi_silver.order_items_clean
# MAGIC workspace.lumi_silver.payments_clean
# MAGIC workspace.lumi_silver.reviews_clean
# MAGIC workspace.lumi_silver.products_clean
# MAGIC workspace.lumi_silver.sellers_clean
# MAGIC workspace.lumi_silver.geolocation_clean
# MAGIC workspace.bagazo_silver.operacion_ingenios_clean
# MAGIC workspace.control.quality_summary_sesion_04
# MAGIC ```
# MAGIC
# MAGIC Estas tablas serán el insumo directo de la **Sesión 5: SQL de negocio y análisis operativo**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Retos finales
# MAGIC
# MAGIC Los retos no bloquean el flujo principal. Úsalos como práctica adicional, cierre o tarea.
# MAGIC
# MAGIC ### Reto nivel 1 — Validación adicional Lumi
# MAGIC
# MAGIC Crea una validación que identifique productos con peso o dimensiones negativas en `products_clean`.
# MAGIC
# MAGIC ### Reto nivel 2 — Extender tabla de calidad
# MAGIC
# MAGIC Agrega una métrica adicional a `quality_summary_sesion_04`, por ejemplo `porcentaje_nulos_criticos` o `porcentaje_reglas_fallidas`.
# MAGIC
# MAGIC ### Reto consultor — Explicación ejecutiva
# MAGIC
# MAGIC Redacta una explicación para gerencia: ¿por qué no deberíamos construir dashboards Gold directamente desde Bronze?
# MAGIC
# MAGIC ### TODO pedagógico 3 — Reflexión final
# MAGIC
# MAGIC Escribe en una celda Markdown una conclusión corta:
# MAGIC
# MAGIC > ¿Qué problema de negocio evita la capa Silver en Lumi o Bagazo?

# COMMAND ----------

print("--- RESULTADOS DEL RETO NIVEL 1 ---")

# Contar cuántos productos violan las restricciones físicas de peso y dimensiones
productos_con_error_fisico = products_clean.filter(
    (F.col("product_weight_g") <= 0) | 
    (F.col("product_length_cm") <= 0) | 
    (F.col("product_height_cm") <= 0) | 
    (F.col("product_width_cm") <= 0)
)
total_errores = productos_con_error_fisico.count()
print(f"Se encontraron {total_errores:,} productos con peso o dimensiones inválidas (<= 0).")
if total_errores > 0:
    print("Muestra de registros a revisar:")
    show_df(productos_con_error_fisico, 5)

# COMMAND ----------

print("--- RESULTADOS DEL RETO NIVEL 2 ---")
# Leemos la tabla de resumen de calidad generada en la sesión
quality_summary_extended = (
    spark.table(table_name(CONTROL_SCHEMA, QUALITY_TABLE))
    # Calculamos el porcentaje matemático y lo redondeamos a 2 decimales
    .withColumn("pct_reglas_fallidas", F.round((F.col("reglas_fallidas") / F.col("filas")) * 100, 2))
)
# Guardamos los cambios sobreescribiendo la tabla de control para incluir la nueva métrica
write_managed_delta(quality_summary_extended, CONTROL_SCHEMA, QUALITY_TABLE)

# Mostramos el resultado final extendido
show_df(quality_summary_extended, 10)

# COMMAND ----------

# MAGIC %md ¿Por qué NO debemos construir dashboards ejecutivos (Gold) directamente desde la capa Bronze?
# MAGIC
# MAGIC 1. Decisiones basadas en datos distorsionados: 
# MAGIC Bronze almacena la información tal y como llega del origen. En el caso de 'Lumi', si calculamos el ticket promedio desde Bronze, sumaríamos transacciones corruptas con montos duplicados por pedidos con múltiples métodos de pago, inflando artificialmente las ventas reales de la compañía.
# MAGIC
# MAGIC 2. Quiebres en la lógica de negocio por formatos regionales: 
# MAGIC En el caso espejo 'Bagazo', los sistemas operativos registraron números con formato latino (ej. '1.486,2'). Si un reporte de negocio lee esto directamente sin pasar por Silver, Spark interpretará el punto de miles como un punto decimal, reportando '1.486' toneladas en lugar de las '1486.2' reales. Una subestimación masiva del inventario para la planta.
# MAGIC
# MAGIC 3. Falta de contexto analítico (Ceros Operativos): 
# MAGIC Un cero en los datos crudos no discrimina si un ingenio no entregó bagazo por ineficiencia o porque estaba en una parada de mantenimiento programada. Sin las banderas operativas generadas en Silver, la gerencia castigará métricas de cumplimiento basándose en ruido, no en realidades operativas.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC > Conclusión: 
# MAGIC La capa Silver en Lumi y Bagazo evita el "silenciar" errores operativos invisibles. En Lumi, previene la duplicidad de ingresos financieros al unificar los pagos estructurados por orden. En Bagazo, mitiga el riesgo de desabastecimiento energético al transformar textos ambiguos (como "Paro por fin de año" o "Falta de caña por lluvias") en variables booleanas lógicas que los modelos predictivos pueden procesar para anticipar contingencias. Silver garantiza que el negocio discuta estrategias sobre la verdad de los datos, no sobre su limpieza.

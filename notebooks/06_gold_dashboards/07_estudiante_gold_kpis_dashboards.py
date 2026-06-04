# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# ///
# MAGIC %md
# MAGIC # Sesión 7 — Capa Gold, KPIs y Dashboards
# MAGIC
# MAGIC **Objetivo:** construir la primera capa Gold formal del curso desde Silver, publicar KPIs comerciales de Lumi y KPIs operativos de Bagazo, y preparar datasets para dashboards.
# MAGIC
# MAGIC ## Reglas de seguridad
# MAGIC
# MAGIC - Solo se lee desde `workspace.lumi_silver`, `workspace.bagazo_silver` y `workspace.control`.
# MAGIC - Solo se escribe en `workspace.lumi_gold`, `workspace.bagazo_gold` y `workspace.control`.
# MAGIC - `workspace.delta_lab` se mantiene como evidencia pedagógica de la Sesión 6, pero **no alimenta Gold**.
# MAGIC - No ejecutar `VACUUM`, `RESTORE`, `UPDATE`, `DELETE` ni `MERGE` sobre Silver.
# MAGIC
# MAGIC > Nota importante: este notebook fue ajustado para usar los nombres reales observados en Silver, por ejemplo `order_purchase_ts`, `order_purchase_date`, `order_approved_ts` y `order_delivered_customer_ts`. Además, incluye helpers de PySpark para evitar errores por pequeñas diferencias de nombres de columnas.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG workspace;
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.lumi_gold;
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.bagazo_gold;
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.control;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Retomar calidad y confiabilidad de sesiones anteriores
# MAGIC
# MAGIC Antes de publicar Gold, revisamos dos evidencias:
# MAGIC
# MAGIC 1. Calidad de Silver creada en la Sesión 4.
# MAGIC 2. Confiabilidad Delta revisada en la Sesión 6.
# MAGIC
# MAGIC `reviews_clean` continúa en estado **REVISAR**, por eso sus métricas se agregan a nivel `order_id` antes de cruzarlas con órdenes, ítems o categorías.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM workspace.control.quality_summary_sesion_04
# MAGIC ORDER BY dataset, tabla;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM workspace.control.delta_reliability_summary_sesion_06
# MAGIC ORDER BY tabla;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inventario de fuentes oficiales
# MAGIC
# MAGIC Gold parte de Silver. Las tablas de `workspace.delta_lab` no se usan como fuente de negocio porque fueron tablas de laboratorio de la Sesión 6.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN workspace.lumi_silver;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN workspace.bagazo_silver;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Helpers robustos para construir Gold
# MAGIC
# MAGIC Las tablas Silver pueden tener nombres ya estandarizados, por ejemplo `order_purchase_ts` en lugar de `order_purchase_timestamp`. Para evitar errores de columnas no resueltas, esta sección detecta columnas disponibles y selecciona la primera columna válida de una lista de candidatos.
# MAGIC

# COMMAND ----------

from functools import reduce
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame

CATALOGO = "workspace"
LUMI_SILVER = "workspace.lumi_silver"
BAGAZO_SILVER = "workspace.bagazo_silver"
LUMI_GOLD = "workspace.lumi_gold"
BAGAZO_GOLD = "workspace.bagazo_gold"
CONTROL = "workspace.control"

spark.sql(f"USE CATALOG {CATALOGO}")


def qcol(nombre: str) -> str:
    """Escapa nombres de columnas para expresiones SQL."""
    return "`" + nombre.replace("`", "``") + "`"


def existing_col(df: DataFrame, candidates):
    """Devuelve el primer nombre de columna existente dentro de una lista de candidatos."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def pick(df: DataFrame, candidates, alias: str, default=None, data_type: str | None = None):
    """Selecciona una columna existente o un valor por defecto, con cast tolerante cuando aplica."""
    c = existing_col(df, candidates)
    if c is None:
        expr = F.lit(default)
        if data_type:
            expr = expr.cast(data_type)
        return expr.alias(alias)
    if data_type and data_type.upper() not in ["STRING"]:
        expr = F.expr(f"try_cast({qcol(c)} AS {data_type})")
    elif data_type:
        expr = F.col(c).cast(data_type)
    else:
        expr = F.col(c)
    return expr.alias(alias)


def coalesce_pick(df: DataFrame, candidates, alias: str, data_type: str, default=None):
    """Hace coalesce entre todas las columnas candidatas existentes."""
    exprs = []
    for c in candidates:
        if c in df.columns:
            if data_type.upper() == "DATE":
                exprs.append(F.to_date(F.col(c)))
            elif data_type.upper() == "TIMESTAMP":
                exprs.append(F.to_timestamp(F.col(c)))
            elif data_type.upper() == "STRING":
                exprs.append(F.col(c).cast("string"))
            else:
                exprs.append(F.expr(f"try_cast({qcol(c)} AS {data_type})"))
    if not exprs:
        return F.lit(default).cast(data_type).alias(alias)
    return F.coalesce(*exprs).alias(alias)


def write_table(df: DataFrame, full_name: str):
    """Escribe una tabla Delta administrada, reemplazando solo tablas Gold/Control."""
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_name)
    print(f"OK -> {full_name}: {spark.table(full_name).count():,} filas")


def preview_schema(table_name: str):
    df = spark.table(table_name)
    return spark.createDataFrame([(table_name, c, t) for c, t in df.dtypes], ["tabla", "columna", "tipo"])


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Diagnóstico rápido de esquemas Silver
# MAGIC
# MAGIC Este bloque ayuda a confirmar los nombres reales de columnas. No modifica nada.
# MAGIC

# COMMAND ----------

schemas_df = (
    preview_schema("workspace.lumi_silver.orders_clean")
    .unionByName(preview_schema("workspace.lumi_silver.order_items_clean"))
    .unionByName(preview_schema("workspace.lumi_silver.payments_clean"))
    .unionByName(preview_schema("workspace.lumi_silver.reviews_clean"))
    .unionByName(preview_schema("workspace.bagazo_silver.operacion_ingenios_clean"))
)
display(schemas_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Gold Lumi — dimensiones
# MAGIC
# MAGIC Se crean dimensiones simples y explicables: producto, cliente, vendedor y fecha.
# MAGIC

# COMMAND ----------

products = spark.table("workspace.lumi_silver.products_clean")
customers = spark.table("workspace.lumi_silver.customers_clean")
sellers = spark.table("workspace.lumi_silver.sellers_clean")
orders = spark.table("workspace.lumi_silver.orders_clean")

# Dimensión producto
categoria = F.coalesce(
    *[F.col(c).cast("string") for c in [
        "categoria_producto",
        "product_category_name_english",
        "category_name_english",
        "product_category_name"
    ] if c in products.columns],
    F.lit("sin_categoria")
) if any(c in products.columns for c in ["categoria_producto", "product_category_name_english", "category_name_english", "product_category_name"]) else F.lit("sin_categoria")

dim_product = products.select(
    pick(products, ["product_id"], "product_id", data_type="STRING"),
    categoria.alias("categoria_producto"),
    pick(products, ["product_category_name"], "product_category_name", data_type="STRING"),
    pick(products, ["product_category_name_english", "category_name_english", "categoria_producto"], "product_category_name_english", data_type="STRING"),
    pick(products, ["product_weight_g"], "product_weight_g", data_type="DOUBLE"),
    pick(products, ["product_length_cm"], "product_length_cm", data_type="DOUBLE"),
    pick(products, ["product_height_cm"], "product_height_cm", data_type="DOUBLE"),
    pick(products, ["product_width_cm"], "product_width_cm", data_type="DOUBLE")
).withColumn(
    "categoria_requiere_revision",
    F.when(
        F.col("categoria_producto").isNull() |
        (F.lower(F.col("categoria_producto")).isin("sin_categoria", "not_defined", "sin_definir")),
        F.lit(True)
    ).otherwise(F.lit(False))
)
write_table(dim_product, "workspace.lumi_gold.dim_product")

# Dimensión cliente
dim_customer = customers.select(
    pick(customers, ["customer_id"], "customer_id", data_type="STRING"),
    pick(customers, ["customer_unique_id"], "customer_unique_id", data_type="STRING"),
    pick(customers, ["customer_zip_code_prefix"], "customer_zip_code_prefix", data_type="STRING"),
    pick(customers, ["customer_city"], "customer_city", data_type="STRING"),
    pick(customers, ["customer_state"], "customer_state", data_type="STRING")
)
write_table(dim_customer, "workspace.lumi_gold.dim_customer")

# Dimensión vendedor
dim_seller = sellers.select(
    pick(sellers, ["seller_id"], "seller_id", data_type="STRING"),
    pick(sellers, ["seller_zip_code_prefix"], "seller_zip_code_prefix", data_type="STRING"),
    pick(sellers, ["seller_city"], "seller_city", data_type="STRING"),
    pick(sellers, ["seller_state"], "seller_state", data_type="STRING")
)
write_table(dim_seller, "workspace.lumi_gold.dim_seller")

# Dimensión fecha construida desde las fechas reales disponibles en orders_clean
fecha_groups = [
    ["order_purchase_date", "order_purchase_ts", "order_purchase_timestamp"],
    ["order_approved_date", "order_approved_ts", "order_approved_at"],
    ["order_delivered_customer_date", "order_delivered_customer_ts"],
    ["order_estimated_delivery_date", "order_estimated_delivery_ts"]
]
fecha_dfs = []
for group in fecha_groups:
    if existing_col(orders, group):
        fecha_dfs.append(
            orders.select(coalesce_pick(orders, group, "fecha", "DATE"))
            .where(F.col("fecha").isNotNull())
        )

if fecha_dfs:
    fechas = reduce(lambda a, b: a.unionByName(b), fecha_dfs).distinct()
else:
    fechas = spark.createDataFrame([], T.StructType([T.StructField("fecha", T.DateType(), True)]))

dim_date = fechas.select(
    F.col("fecha"),
    F.year("fecha").alias("anio"),
    F.month("fecha").alias("mes"),
    F.date_format("fecha", "yyyy-MM").alias("anio_mes"),
    F.dayofmonth("fecha").alias("dia_mes"),
    F.dayofweek("fecha").alias("dia_semana"),
    F.quarter("fecha").alias("trimestre")
)
write_table(dim_date, "workspace.lumi_gold.dim_date")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Gold Lumi — hechos
# MAGIC
# MAGIC Granularidades de referencia:
# MAGIC
# MAGIC - `fact_orders`: una fila por `order_id`.
# MAGIC - `fact_sales_items`: una fila por `order_id + order_item_id`.
# MAGIC - `fact_payments`: una fila por `order_id`.
# MAGIC - `fact_delivery_experience`: una fila por `order_id`, con reviews agregadas previamente.
# MAGIC

# COMMAND ----------

orders = spark.table("workspace.lumi_silver.orders_clean")
customers = spark.table("workspace.lumi_silver.customers_clean")
items = spark.table("workspace.lumi_silver.order_items_clean")
payments = spark.table("workspace.lumi_silver.payments_clean")
reviews = spark.table("workspace.lumi_silver.reviews_clean")
dim_product = spark.table("workspace.lumi_gold.dim_product")

# -----------------------------
# fact_orders
# -----------------------------
purchase_date_expr = coalesce_pick(orders, ["order_purchase_date", "order_purchase_ts", "order_purchase_timestamp"], "purchase_date", "DATE")
purchase_ts_expr = coalesce_pick(orders, ["order_purchase_ts", "order_purchase_timestamp", "order_purchase_date"], "order_purchase_ts", "TIMESTAMP")
approved_ts_expr = coalesce_pick(orders, ["order_approved_ts", "order_approved_at", "order_approved_date"], "order_approved_ts", "TIMESTAMP")
delivered_date_expr = coalesce_pick(orders, ["order_delivered_customer_date", "order_delivered_customer_ts"], "delivered_date", "DATE")
delivered_ts_expr = coalesce_pick(orders, ["order_delivered_customer_ts", "order_delivered_customer_date"], "order_delivered_customer_ts", "TIMESTAMP")
estimated_date_expr = coalesce_pick(orders, ["order_estimated_delivery_date", "order_estimated_delivery_ts"], "estimated_delivery_date", "DATE")

orders_base = orders.select(
    pick(orders, ["order_id"], "order_id", data_type="STRING"),
    pick(orders, ["customer_id"], "customer_id", data_type="STRING"),
    pick(orders, ["order_status"], "order_status", data_type="STRING"),
    purchase_date_expr,
    purchase_ts_expr,
    approved_ts_expr,
    delivered_date_expr,
    delivered_ts_expr,
    estimated_date_expr,
    pick(orders, ["delivery_days"], "delivery_days_original", data_type="INT"),
    pick(orders, ["delay_days"], "delay_days_original", data_type="INT"),
    pick(orders, ["is_late"], "is_late_original", data_type="BOOLEAN")
).withColumn(
    "purchase_month", F.date_format("purchase_date", "yyyy-MM")
).withColumn(
    "delivery_days",
    F.coalesce(F.col("delivery_days_original"), F.datediff(F.col("delivered_date"), F.col("purchase_date")))
).withColumn(
    "delay_days",
    F.coalesce(F.col("delay_days_original"), F.datediff(F.col("delivered_date"), F.col("estimated_delivery_date")))
).withColumn(
    "is_late",
    F.coalesce(F.col("is_late_original"), F.when(F.col("delay_days").isNull(), F.lit(False)).otherwise(F.col("delay_days") > 0))
).drop("delivery_days_original", "delay_days_original", "is_late_original")

customers_sel = customers.select(
    pick(customers, ["customer_id"], "customer_id", data_type="STRING"),
    pick(customers, ["customer_unique_id"], "customer_unique_id", data_type="STRING"),
    pick(customers, ["customer_state"], "customer_state", data_type="STRING"),
    pick(customers, ["customer_city"], "customer_city", data_type="STRING")
)

fact_orders = orders_base.join(customers_sel, on="customer_id", how="left")
write_table(fact_orders, "workspace.lumi_gold.fact_orders")

# -----------------------------
# fact_sales_items
# -----------------------------
items_base = items.select(
    pick(items, ["order_id"], "order_id", data_type="STRING"),
    pick(items, ["order_item_id"], "order_item_id", data_type="INT"),
    pick(items, ["product_id"], "product_id", data_type="STRING"),
    pick(items, ["seller_id"], "seller_id", data_type="STRING"),
    coalesce_pick(items, ["shipping_limit_date", "shipping_limit_ts"], "shipping_limit_date", "DATE"),
    pick(items, ["price", "item_price"], "item_price", default=0.0, data_type="DOUBLE"),
    pick(items, ["freight_value", "valor_flete"], "freight_value", default=0.0, data_type="DOUBLE")
).withColumn(
    "item_total_value", F.coalesce(F.col("item_price"), F.lit(0.0)) + F.coalesce(F.col("freight_value"), F.lit(0.0))
)

fact_sales_items = items_base.join(
    dim_product.select("product_id", "categoria_producto"),
    on="product_id",
    how="left"
).withColumn("categoria_producto", F.coalesce(F.col("categoria_producto"), F.lit("sin_categoria")))
write_table(fact_sales_items, "workspace.lumi_gold.fact_sales_items")

# -----------------------------
# fact_payments agregado a nivel order_id
# -----------------------------
payments_base = payments.select(
    pick(payments, ["order_id"], "order_id", data_type="STRING"),
    pick(payments, ["payment_type", "main_payment_type"], "payment_type", default="sin_dato", data_type="STRING"),
    pick(payments, ["payment_installments", "total_installments"], "payment_installments", default=0, data_type="INT"),
    pick(payments, ["payment_value", "payment_value_total", "valor_pagado_total"], "payment_value", default=0.0, data_type="DOUBLE")
)

fact_payments = payments_base.groupBy("order_id").agg(
    F.count(F.lit(1)).alias("payment_records"),
    F.countDistinct("payment_type").alias("payment_type_count"),
    F.max("payment_type").alias("main_payment_type"),
    F.sum("payment_installments").alias("total_installments"),
    F.sum("payment_value").alias("payment_value_total")
)
write_table(fact_payments, "workspace.lumi_gold.fact_payments")

# -----------------------------
# fact_delivery_experience agregado a nivel order_id
# -----------------------------
comment_expr = F.coalesce(
    *[F.col(c).cast("string") for c in ["review_comment_message", "review_comment_title", "comment", "comentario"] if c in reviews.columns],
    F.lit(None).cast("string")
) if any(c in reviews.columns for c in ["review_comment_message", "review_comment_title", "comment", "comentario"]) else F.lit(None).cast("string")

reviews_base = reviews.select(
    pick(reviews, ["order_id"], "order_id", data_type="STRING"),
    pick(reviews, ["review_score"], "review_score", data_type="DOUBLE"),
    comment_expr.alias("review_comment_text")
)

reviews_por_orden = reviews_base.groupBy("order_id").agg(
    F.count(F.lit(1)).alias("review_records"),
    F.avg("review_score").alias("review_score_avg"),
    F.min("review_score").cast("int").alias("review_score_min"),
    F.max("review_score").cast("int").alias("review_score_max"),
    F.sum(F.when(F.col("review_score") <= 2, 1).otherwise(0)).alias("low_review_records"),
    F.max(F.when(F.length(F.trim(F.col("review_comment_text"))) > 0, 1).otherwise(0)).alias("has_review_comment")
)

fact_delivery_experience = spark.table("workspace.lumi_gold.fact_orders").join(
    reviews_por_orden,
    on="order_id",
    how="left"
).select(
    "order_id", "customer_id", "order_status", "purchase_date", "purchase_month", "delivery_days", "delay_days", "is_late",
    F.coalesce(F.col("review_records"), F.lit(0)).alias("review_records"),
    "review_score_avg", "review_score_min", "review_score_max",
    F.coalesce(F.col("low_review_records"), F.lit(0)).alias("low_review_records"),
    F.coalesce(F.col("has_review_comment"), F.lit(0)).alias("has_review_comment")
).withColumn(
    "tramo_demora",
    F.when(F.col("delay_days").isNull(), "sin_dato")
     .when(F.col("delay_days") <= 0, "a_tiempo")
     .when(F.col("delay_days").between(1, 7), "tarde_1_7_dias")
     .when(F.col("delay_days").between(8, 15), "tarde_8_15_dias")
     .otherwise("tarde_mas_15_dias")
)
write_table(fact_delivery_experience, "workspace.lumi_gold.fact_delivery_experience")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. KPIs Lumi
# MAGIC
# MAGIC Ahora convertimos hechos en tablas de KPI listas para dashboard. Todas estas tablas parten de Gold, no directamente de Silver.
# MAGIC

# COMMAND ----------

fo = spark.table("workspace.lumi_gold.fact_orders")
fi = spark.table("workspace.lumi_gold.fact_sales_items")
fp = spark.table("workspace.lumi_gold.fact_payments")
fe = spark.table("workspace.lumi_gold.fact_delivery_experience")

# Ventas mensuales
monthly_base = fo.join(fi, on="order_id", how="left").where(F.col("purchase_month").isNotNull())
kpi_monthly_sales = monthly_base.groupBy("purchase_month").agg(
    F.countDistinct("order_id").alias("pedidos"),
    F.sum("item_price").alias("venta_items"),
    F.sum("freight_value").alias("valor_flete"),
    F.sum("item_total_value").alias("venta_total_items"),
    F.avg(F.col("is_late").cast("int")).alias("tasa_entrega_tardia")
).withColumn(
    "ticket_promedio_items",
    F.round(F.col("venta_total_items") / F.when(F.col("pedidos") == 0, None).otherwise(F.col("pedidos")), 2)
).withColumn("tasa_entrega_tardia", F.round("tasa_entrega_tardia", 4))
write_table(kpi_monthly_sales, "workspace.lumi_gold.kpi_monthly_sales")

# Desempeño por categoría
category_base = fi.join(fe.select("order_id", "review_score_avg", "is_late"), on="order_id", how="left")
kpi_category_performance = category_base.groupBy("categoria_producto").agg(
    F.countDistinct("order_id").alias("pedidos"),
    F.count(F.lit(1)).alias("items_vendidos"),
    F.sum("item_total_value").alias("venta_total_items"),
    F.round(F.avg("review_score_avg"), 2).alias("review_promedio"),
    F.round(F.avg(F.col("is_late").cast("int")), 4).alias("tasa_entrega_tardia")
)
write_table(kpi_category_performance, "workspace.lumi_gold.kpi_category_performance")

# Métodos de pago
kpi_payment_methods = fp.groupBy("main_payment_type").agg(
    F.countDistinct("order_id").alias("pedidos"),
    F.sum("payment_value_total").alias("valor_pagado_total"),
    F.round(F.avg("payment_value_total"), 2).alias("pago_promedio_por_pedido"),
    F.round(F.avg("total_installments"), 2).alias("cuotas_promedio")
).withColumnRenamed("main_payment_type", "payment_type")
write_table(kpi_payment_methods, "workspace.lumi_gold.kpi_payment_methods")

# Demora vs review
kpi_delivery_review = fe.groupBy("tramo_demora").agg(
    F.countDistinct("order_id").alias("pedidos"),
    F.round(F.avg("review_score_avg"), 2).alias("review_promedio"),
    F.round(F.avg("delay_days"), 2).alias("demora_promedio_dias"),
    F.round(F.avg(F.col("is_late").cast("int")), 4).alias("tasa_entrega_tardia"),
    F.sum("low_review_records").alias("reviews_bajas")
)
write_table(kpi_delivery_review, "workspace.lumi_gold.kpi_delivery_review")

# Experiencia por estado del cliente
kpi_customer_experience = fo.join(fe.select("order_id", "review_score_avg", "low_review_records"), on="order_id", how="left") \
    .groupBy("customer_state").agg(
        F.countDistinct("order_id").alias("pedidos"),
        F.round(F.avg("review_score_avg"), 2).alias("review_promedio"),
        F.round(F.avg(F.col("is_late").cast("int")), 4).alias("tasa_entrega_tardia"),
        F.sum(F.coalesce(F.col("low_review_records"), F.lit(0))).alias("reviews_bajas")
    )
write_table(kpi_customer_experience, "workspace.lumi_gold.kpi_customer_experience")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Validación rápida Lumi
# MAGIC
# MAGIC Esta validación confirma que las tablas a nivel pedido conservan una fila por `order_id`.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'fact_orders' AS tabla, COUNT(*) AS filas, COUNT(DISTINCT order_id) AS ordenes_distintas FROM workspace.lumi_gold.fact_orders
# MAGIC UNION ALL
# MAGIC SELECT 'fact_payments' AS tabla, COUNT(*) AS filas, COUNT(DISTINCT order_id) AS ordenes_distintas FROM workspace.lumi_gold.fact_payments
# MAGIC UNION ALL
# MAGIC SELECT 'fact_delivery_experience' AS tabla, COUNT(*) AS filas, COUNT(DISTINCT order_id) AS ordenes_distintas FROM workspace.lumi_gold.fact_delivery_experience;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Gold Bagazo — fact y KPIs
# MAGIC
# MAGIC Bagazo mantiene granularidad `fecha + ingenio`. Esta tabla queda lista para KPIs y como base para feature engineering en la Sesión 8.
# MAGIC

# COMMAND ----------

bagazo = spark.table("workspace.bagazo_silver.operacion_ingenios_clean")

bagazo_base = bagazo.select(
    coalesce_pick(bagazo, ["fecha"], "fecha", "DATE"),
    pick(bagazo, ["ingenio"], "ingenio", data_type="STRING"),
    pick(bagazo, ["anio"], "anio_original", data_type="INT"),
    pick(bagazo, ["mes"], "mes_original", data_type="INT"),
    pick(bagazo, ["dia_semana"], "dia_semana_original", data_type="INT"),
    pick(bagazo, ["lluvia_mm", "promedio_lluvias_mm"], "lluvia_mm", data_type="DOUBLE"),
    pick(bagazo, ["cana_molida_ton", "cana_molida_toneladas"], "cana_molida_ton", data_type="DOUBLE"),
    pick(bagazo, ["bagazo_entregado_ton", "bagazo_entregado_toneladas"], "bagazo_entregado_ton", data_type="DOUBLE"),
    pick(bagazo, ["comentario", "comentarios", "comentario_operativo"], "comentario", data_type="STRING"),
    pick(bagazo, ["lluvia_alta"], "lluvia_alta_original", data_type="BOOLEAN"),
    pick(bagazo, ["riesgo_bajo_bagazo"], "riesgo_bajo_bagazo_original", data_type="BOOLEAN"),
    pick(bagazo, ["tiene_comentario_operativo"], "tiene_comentario_operativo_original", data_type="BOOLEAN"),
    pick(bagazo, ["es_temporada_lluviosa", "temporada_lluviosa"], "es_temporada_lluviosa_original", data_type="BOOLEAN")
).withColumn(
    "anio", F.coalesce(F.col("anio_original"), F.year("fecha"))
).withColumn(
    "mes", F.coalesce(F.col("mes_original"), F.month("fecha"))
).withColumn(
    "dia_semana", F.coalesce(F.col("dia_semana_original"), F.dayofweek("fecha"))
).drop("anio_original", "mes_original", "dia_semana_original")

umbrales = bagazo_base.groupBy("ingenio").agg(
    F.expr("percentile_approx(bagazo_entregado_ton, 0.25)").alias("bagazo_p25_ingenio")
)

fact_operacion_ingenios = bagazo_base.join(umbrales, on="ingenio", how="left").withColumn(
    "lluvia_alta",
    F.coalesce(F.col("lluvia_alta_original"), F.when(F.col("lluvia_mm") >= 30, True).otherwise(False))
).withColumn(
    "riesgo_bajo_bagazo",
    F.coalesce(
        F.col("riesgo_bajo_bagazo_original"),
        F.when(F.col("bagazo_entregado_ton").isNull() | F.col("bagazo_p25_ingenio").isNull(), False)
         .otherwise(F.col("bagazo_entregado_ton") <= F.col("bagazo_p25_ingenio"))
    )
).withColumn(
    "tiene_comentario_operativo",
    F.coalesce(
        F.col("tiene_comentario_operativo_original"),
        F.when(F.length(F.trim(F.col("comentario"))) > 0, True).otherwise(False)
    )
).withColumn(
    "es_temporada_lluviosa",
    F.coalesce(F.col("es_temporada_lluviosa_original"), F.col("mes").isin(4, 5, 10, 11))
).withColumn(
    "anio_mes", F.date_format("fecha", "yyyy-MM")
).withColumn(
    "tramo_lluvia",
    F.when(F.col("lluvia_mm").isNull(), "sin_dato_lluvia")
     .when(F.col("lluvia_mm") == 0, "seco")
     .when((F.col("lluvia_mm") > 0) & (F.col("lluvia_mm") < 10), "lluvia_baja")
     .when((F.col("lluvia_mm") >= 10) & (F.col("lluvia_mm") < 30), "lluvia_media")
     .otherwise("lluvia_alta")
).drop(
    "lluvia_alta_original", "riesgo_bajo_bagazo_original", "tiene_comentario_operativo_original",
    "es_temporada_lluviosa_original", "bagazo_p25_ingenio"
)

write_table(fact_operacion_ingenios, "workspace.bagazo_gold.fact_operacion_ingenios")

fb = spark.table("workspace.bagazo_gold.fact_operacion_ingenios")

kpi_lluvia_bagazo_mensual = fb.groupBy("anio_mes", "ingenio").agg(
    F.count(F.lit(1)).alias("dias_observados"),
    F.round(F.avg("lluvia_mm"), 2).alias("lluvia_promedio_mm"),
    F.round(F.sum("lluvia_mm"), 2).alias("lluvia_total_mm"),
    F.round(F.avg("cana_molida_ton"), 2).alias("cana_promedio_ton"),
    F.round(F.avg("bagazo_entregado_ton"), 2).alias("bagazo_promedio_ton"),
    F.round(F.sum("bagazo_entregado_ton"), 2).alias("bagazo_total_ton"),
    F.sum(F.col("riesgo_bajo_bagazo").cast("int")).alias("dias_riesgo_bajo_bagazo")
)
write_table(kpi_lluvia_bagazo_mensual, "workspace.bagazo_gold.kpi_lluvia_bagazo_mensual")

kpi_bagazo_por_ingenio = fb.groupBy("ingenio").agg(
    F.count(F.lit(1)).alias("dias_observados"),
    F.round(F.avg("bagazo_entregado_ton"), 2).alias("bagazo_promedio_ton"),
    F.round(F.sum("bagazo_entregado_ton"), 2).alias("bagazo_total_ton"),
    F.round(F.avg("cana_molida_ton"), 2).alias("cana_promedio_ton"),
    F.round(F.avg("lluvia_mm"), 2).alias("lluvia_promedio_mm"),
    F.sum(F.col("riesgo_bajo_bagazo").cast("int")).alias("dias_riesgo_bajo_bagazo")
)
write_table(kpi_bagazo_por_ingenio, "workspace.bagazo_gold.kpi_bagazo_por_ingenio")

kpi_riesgo_bajo_bagazo = fb.groupBy("ingenio", "anio_mes").agg(
    F.count(F.lit(1)).alias("dias_observados"),
    F.sum(F.col("riesgo_bajo_bagazo").cast("int")).alias("dias_riesgo"),
    F.round(F.avg(F.col("riesgo_bajo_bagazo").cast("int")), 4).alias("tasa_riesgo_bajo_bagazo"),
    F.round(F.avg("lluvia_mm"), 2).alias("lluvia_promedio_mm"),
    F.round(F.avg("bagazo_entregado_ton"), 2).alias("bagazo_promedio_ton")
)
write_table(kpi_riesgo_bajo_bagazo, "workspace.bagazo_gold.kpi_riesgo_bajo_bagazo")

kpi_dias_secos_vs_lluviosos = fb.groupBy("ingenio", "tramo_lluvia").agg(
    F.count(F.lit(1)).alias("dias_observados"),
    F.round(F.avg("lluvia_mm"), 2).alias("lluvia_promedio_mm"),
    F.round(F.avg("cana_molida_ton"), 2).alias("cana_promedio_ton"),
    F.round(F.avg("bagazo_entregado_ton"), 2).alias("bagazo_promedio_ton"),
    F.sum(F.col("riesgo_bajo_bagazo").cast("int")).alias("dias_riesgo_bajo_bagazo")
)
write_table(kpi_dias_secos_vs_lluviosos, "workspace.bagazo_gold.kpi_dias_secos_vs_lluviosos")

kpi_temporada_lluviosa = fb.groupBy("ingenio", "es_temporada_lluviosa").agg(
    F.count(F.lit(1)).alias("dias_observados"),
    F.round(F.avg("lluvia_mm"), 2).alias("lluvia_promedio_mm"),
    F.round(F.avg("bagazo_entregado_ton"), 2).alias("bagazo_promedio_ton"),
    F.round(F.avg(F.col("riesgo_bajo_bagazo").cast("int")), 4).alias("tasa_riesgo_bajo_bagazo")
)
write_table(kpi_temporada_lluviosa, "workspace.bagazo_gold.kpi_temporada_lluviosa")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Validación rápida Bagazo
# MAGIC
# MAGIC La tabla fact debe conservar la granularidad `fecha + ingenio`.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*) AS filas,
# MAGIC   COUNT(DISTINCT concat(CAST(fecha AS STRING), '||', ingenio)) AS combinaciones_fecha_ingenio
# MAGIC FROM workspace.bagazo_gold.fact_operacion_ingenios;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Control de publicación Gold
# MAGIC
# MAGIC Esta tabla resume qué productos Gold quedaron publicados, cuál es su fuente, su granularidad y observaciones de calidad.
# MAGIC

# COMMAND ----------

from datetime import datetime

publication_rows = [
    ("workspace.lumi_gold", "dim_product", "workspace.lumi_silver.products_clean", "product_id", "Dimensión de producto", spark.table("workspace.lumi_gold.dim_product").count(), "OK", "Mantiene sin_categoria como señal de calidad"),
    ("workspace.lumi_gold", "dim_customer", "workspace.lumi_silver.customers_clean", "customer_id", "Dimensión de cliente", spark.table("workspace.lumi_gold.dim_customer").count(), "OK", "Lista para segmentación geográfica"),
    ("workspace.lumi_gold", "dim_seller", "workspace.lumi_silver.sellers_clean", "seller_id", "Dimensión de vendedor", spark.table("workspace.lumi_gold.dim_seller").count(), "OK", "Lista para análisis por vendedor"),
    ("workspace.lumi_gold", "dim_date", "workspace.lumi_silver.orders_clean", "fecha", "Dimensión de fecha", spark.table("workspace.lumi_gold.dim_date").count(), "OK", "Construida desde fechas disponibles de pedidos"),
    ("workspace.lumi_gold", "fact_orders", "workspace.lumi_silver.orders_clean", "order_id", "Hecho de pedidos", spark.table("workspace.lumi_gold.fact_orders").count(), "OK", "No modifica Silver"),
    ("workspace.lumi_gold", "fact_sales_items", "workspace.lumi_silver.order_items_clean", "order_id + order_item_id", "Hecho de ventas por ítem", spark.table("workspace.lumi_gold.fact_sales_items").count(), "OK", "Evita doble conteo de pagos"),
    ("workspace.lumi_gold", "fact_payments", "workspace.lumi_silver.payments_clean", "order_id", "Pagos agregados por pedido", spark.table("workspace.lumi_gold.fact_payments").count(), "OK", "No cruzar directamente contra ítems"),
    ("workspace.lumi_gold", "fact_delivery_experience", "workspace.lumi_silver.reviews_clean + orders_clean", "order_id", "Experiencia agregada por pedido", spark.table("workspace.lumi_gold.fact_delivery_experience").count(), "REVISAR", "reviews_clean tiene duplicados; se agrega por order_id"),
    ("workspace.bagazo_gold", "fact_operacion_ingenios", "workspace.bagazo_silver.operacion_ingenios_clean", "fecha + ingenio", "Hecho operativo para KPIs y MLflow", spark.table("workspace.bagazo_gold.fact_operacion_ingenios").count(), "OK", "Base para Sesión 8"),
    ("workspace.bagazo_gold", "kpi_lluvia_bagazo_mensual", "workspace.bagazo_gold.fact_operacion_ingenios", "anio_mes + ingenio", "KPI lluvia vs bagazo mensual", spark.table("workspace.bagazo_gold.kpi_lluvia_bagazo_mensual").count(), "OK", "Listo para dashboard operativo")
]

publication_schema = T.StructType([
    T.StructField("schema_gold", T.StringType(), False),
    T.StructField("tabla_gold", T.StringType(), False),
    T.StructField("fuente_principal", T.StringType(), False),
    T.StructField("granularidad", T.StringType(), False),
    T.StructField("proposito", T.StringType(), False),
    T.StructField("filas", T.LongType(), False),
    T.StructField("estado_validacion", T.StringType(), False),
    T.StructField("observaciones", T.StringType(), True),
])

publication_df = spark.createDataFrame(publication_rows, publication_schema).withColumn("fecha_publicacion", F.current_timestamp())
write_table(publication_df, "workspace.control.gold_publication_summary_sesion_07")

display(spark.table("workspace.control.gold_publication_summary_sesion_07").orderBy("schema_gold", "tabla_gold"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Consultas listas para dashboard
# MAGIC
# MAGIC Estas consultas se pueden visualizar directamente desde el notebook. También pueden servir como base para Databricks Dashboards / AI-BI Dashboards si el entorno lo permite.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ventas mensuales Lumi
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT purchase_month, pedidos, venta_total_items, ticket_promedio_items, tasa_entrega_tardia
# MAGIC FROM workspace.lumi_gold.kpi_monthly_sales
# MAGIC ORDER BY purchase_month;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorías principales Lumi
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT categoria_producto, pedidos, venta_total_items, review_promedio, tasa_entrega_tardia
# MAGIC FROM workspace.lumi_gold.kpi_category_performance
# MAGIC ORDER BY venta_total_items DESC
# MAGIC LIMIT 15;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Métodos de pago Lumi
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT payment_type, pedidos, valor_pagado_total, pago_promedio_por_pedido
# MAGIC FROM workspace.lumi_gold.kpi_payment_methods
# MAGIC ORDER BY valor_pagado_total DESC;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demora vs review Lumi
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT tramo_demora, pedidos, review_promedio, demora_promedio_dias, tasa_entrega_tardia
# MAGIC FROM workspace.lumi_gold.kpi_delivery_review
# MAGIC ORDER BY demora_promedio_dias;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lluvia vs bagazo mensual
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT anio_mes, ingenio, lluvia_promedio_mm, bagazo_promedio_ton, dias_riesgo_bajo_bagazo
# MAGIC FROM workspace.bagazo_gold.kpi_lluvia_bagazo_mensual
# MAGIC ORDER BY anio_mes, ingenio;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bagazo por ingenio
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ingenio, dias_observados, bagazo_promedio_ton, lluvia_promedio_mm, dias_riesgo_bajo_bagazo
# MAGIC FROM workspace.bagazo_gold.kpi_bagazo_por_ingenio
# MAGIC ORDER BY dias_riesgo_bajo_bagazo DESC;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. TODOs finales de reflexión
# MAGIC
# MAGIC Estos TODOs no bloquean la ejecución del notebook. Son actividades pedagógicas para cerrar la sesión:
# MAGIC
# MAGIC 1. Selecciona una consulta y crea una visualización en notebook.
# MAGIC 2. Escribe una conclusión ejecutiva sobre Lumi.
# MAGIC 3. Escribe una recomendación operativa sobre Bagazo.
# MAGIC 4. Selecciona el KPI más importante para pasar a dashboard.
# MAGIC 5. Documenta un riesgo de interpretación para un KPI de reviews.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Retos
# MAGIC
# MAGIC ### Reto nivel 1 — Validar una tabla Gold
# MAGIC
# MAGIC Elige una tabla Gold y valida:
# MAGIC
# MAGIC - conteo de registros;
# MAGIC - columnas principales;
# MAGIC - fuente principal;
# MAGIC - granularidad;
# MAGIC - observaciones de calidad.
# MAGIC
# MAGIC ### Reto nivel 2 — Crear un KPI adicional
# MAGIC
# MAGIC Opciones sugeridas:
# MAGIC
# MAGIC - Lumi: `kpi_seller_performance`.
# MAGIC - Lumi: `kpi_delivery_by_state`.
# MAGIC - Bagazo: `kpi_eficiencia_bagazo_cana`.
# MAGIC - Bagazo: `kpi_variabilidad_operativa_ingenio`.
# MAGIC
# MAGIC ### Reto consultor — Diseño de mini dashboard ejecutivo
# MAGIC
# MAGIC Completa este formato:
# MAGIC
# MAGIC ```text
# MAGIC Nombre del dashboard:
# MAGIC Audiencia:
# MAGIC Preguntas que responde:
# MAGIC Tablas Gold usadas:
# MAGIC KPIs principales:
# MAGIC Visualizaciones sugeridas:
# MAGIC Filtros:
# MAGIC Riesgos de interpretación:
# MAGIC Recomendación ejecutiva:
# MAGIC ```
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Tabla Gold: workspace.lumi_gold.fact_delivery_experience.
# MAGIC
# MAGIC Conteo de registros: 99,441 filas.
# MAGIC
# MAGIC Columnas principales: order_id, customer_id, order_status, delivery_days, delay_days, is_late, review_score_avg.
# MAGIC
# MAGIC Fuente principal: workspace.lumi_silver.orders_clean y workspace.lumi_silver.reviews_clean.
# MAGIC
# MAGIC Granularidad: Una fila por cada pedido único (order_id).
# MAGIC
# MAGIC Observaciones de calidad: Su estado es REVISAR debido a que la tabla origen reviews_clean posee duplicados de clave. Para blindar la capa Gold de métricas infladas, se aplicó una agregación previa con un groupBy("order_id") calculando promedios y máximos de las calificaciones antes de realizar el join definitivo.

# COMMAND ----------


# RETO NIVEL 2: Creación de KPI - kpi_seller_performance (Lumi)

fi = spark.table("workspace.lumi_gold.fact_sales_items")
fo = spark.table("workspace.lumi_gold.fact_orders")

kpi_seller_performance = (
    fi.join(fo.select("order_id", "is_late"), on="order_id", how="left")
    .groupBy("seller_id")
    .agg(
        F.countDistinct("order_id").alias("pedidos_atendidos"),
        F.count("order_item_id").alias("items_vendidos"),
        F.round(F.sum("item_total_value"), 2).alias("venta_total_acumulada"),
        F.round(F.avg(F.col("is_late").cast(\"int\")), 4).alias("tasa_retraso_vendedor")
    )
)

# Escribir la nueva tabla en el esquema Gold
write_table(kpi_seller_performance, "workspace.lumi_gold.kpi_seller_performance")

# Previsualizar el resultado de los 5 vendedores top
display(spark.table("workspace.lumi_gold.kpi_seller_performance").orderBy(F.desc("venta_total_acumulada")).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Nombre del dashboard: Panel de Control de Suministro y Eficiencia Operativa (Módulo Bagazo)
# MAGIC Audiencia: Gerencia de Operaciones de iData Academy y Directores de Planta de Lumi Commerce.
# MAGIC Preguntas que responde: 1. ¿Qué ingenios presentan mayor vulnerabilidad ante el clima y paradas por mantenimiento?
# MAGIC 2. ¿Cuál es el impacto real de las lluvias sobre las entregas promedio de biomasa?
# MAGIC 3. ¿Cuáles meses concentran el mayor riesgo de desabastecimiento energético?
# MAGIC
# MAGIC Tablas Gold usadas: * workspace.bagazo_gold.fact_operacion_ingenios
# MAGIC
# MAGIC workspace.bagazo_gold.kpi_bagazo_por_ingenio
# MAGIC
# MAGIC workspace.bagazo_gold.kpi_dias_secos_vs_lluviosos
# MAGIC
# MAGIC KPIs principales: * Total de Días Observados (Suma total del histórico).
# MAGIC
# MAGIC Promedio Diario de Bagazo Entregado por ingenio.
# MAGIC
# MAGIC Tasa de Riesgo Bajo de Bagazo (Porcentaje de días con desabastecimiento).
# MAGIC
# MAGIC Días de Riesgo Acumulado ante Lluvia Alta.
# MAGIC
# MAGIC Visualizaciones sugeridas: * Tarjetas de KPI con el recuento de días de riesgo bajo actuales para cada ingenio.
# MAGIC
# MAGIC Gráfico de Barras Agrupadas para comparar el Promedio de Bagazo Entregado en "Día con lluvia alta" frente a "Día sin lluvia alta" en cada ingenio.
# MAGIC
# MAGIC Gráfico de Líneas Temporal para observar la evolución de las lluvias promedio contra las toneladas de bagazo mes a mes.
# MAGIC
# MAGIC Filtros: Rango de Fechas (Año/Mes), Ingenio Específico y Tramo de Lluvia (Seco, Lluvia Baja, Media, Alta).
# MAGIC
# MAGIC Riesgos de interpretación: Un aumento repentino en los días de riesgo bajo de bagazo no siempre indica una mala gestión del proveedor. Puede estar ligado a la estacionalidad climática o a paradas programadas de mantenimiento general, por lo que las métricas de rendimiento siempre deben cruzarse visualmente con las banderas de contexto operativo.
# MAGIC
# MAGIC Recomendación ejecutiva: Utilizar el indicador de la tabla "kpi_dias_secos_vs_lluviosos" para activar contratos de suministro secundario de energía cuando se pronostiquen alertas de tramos de "lluvia alta" de forma consecutiva. Esto reducirá la dependencia crítica hacia el ingenio Providencia, el cual históricamente reporta la mayor cantidad de días bajo riesgo, acumulando 421 días detectados.

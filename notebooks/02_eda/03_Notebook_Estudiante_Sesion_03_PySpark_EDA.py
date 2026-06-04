# Databricks notebook source
# MAGIC %md
# MAGIC # Sesión 3 — Exploración de datos con PySpark
# MAGIC
# MAGIC **Capacitación práctica Databricks — iData Academy**  
# MAGIC Proyecto principal: **Lumi Commerce Lakehouse**  
# MAGIC Caso espejo empresarial: **Lluvia, Caña y Bagazo**
# MAGIC
# MAGIC ## Propósito de la sesión
# MAGIC
# MAGIC En la sesión 2 dejamos cargada la capa **Bronze**. En esta sesión vamos a inspeccionarla con PySpark para responder preguntas reales, detectar señales de calidad y preparar la construcción de **Silver**.
# MAGIC
# MAGIC > Mensaje central: Databricks no se aprende solo escribiendo comandos. Se aprende construyendo un producto de datos reproducible, versionado, gobernable y listo para escalar.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Continuidad con la sesión 2
# MAGIC
# MAGIC En la sesión anterior se cargaron los CSV de Lumi y Bagazo como tablas Bronze. Esta práctica asume que existen:
# MAGIC
# MAGIC ```text
# MAGIC workspace.lumi_bronze.*
# MAGIC workspace.bagazo_bronze.molienda_bagazo_y_lluvias_II
# MAGIC workspace.control.bronze_ingestion_audit
# MAGIC ```
# MAGIC
# MAGIC Si alguna tabla no existe, el notebook se detendrá con un mensaje claro para ejecutar primero la ingesta Bronze.

# COMMAND ----------

# =========================================
# 1. Parámetros de trabajo
# =========================================

CATALOG = "workspace"

LUMI_BRONZE_SCHEMA = "lumi_bronze"
BAGAZO_BRONZE_SCHEMA = "bagazo_bronze"
CONTROL_SCHEMA = "control"
AUDIT_TABLE = "bronze_ingestion_audit"

LUMI_RAW_PATH = "/Volumes/workspace/raw/lumi"
BAGAZO_RAW_PATH = "/Volumes/workspace/raw/bagazo"

BAGAZO_TABLE = "molienda_bagazo_y_lluvias_II"

# Parámetros pedagógicos fáciles de modificar
TOP_N = 10
SHOW_ROWS = 20

print("✅ Parámetros cargados")
print("CATALOG:", CATALOG)
print("LUMI_BRONZE_SCHEMA:", LUMI_BRONZE_SCHEMA)
print("BAGAZO_BRONZE_SCHEMA:", BAGAZO_BRONZE_SCHEMA)
print("TOP_N:", TOP_N)

# COMMAND ----------

# =========================================
# 2. Funciones auxiliares
# =========================================

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from functools import reduce


def table_name(schema: str, table: str) -> str:
    """Construye nombre fully qualified con backticks para evitar problemas de caracteres."""
    return f"`{CATALOG}`.`{schema}`.`{table}`"


def table_exists(schema: str, table: str) -> bool:
    """Valida si una tabla existe en el catálogo actual."""
    try:
        spark.table(table_name(schema, table)).limit(1).count()
        return True
    except Exception:
        return False


def load_table(schema: str, table: str) -> DataFrame:
    """Carga una tabla Bronze como DataFrame."""
    full_name = table_name(schema, table)
    print(f"Leyendo tabla: {CATALOG}.{schema}.{table}")
    return spark.table(full_name)


def show(df: DataFrame, n: int = SHOW_ROWS):
    """Muestra resultados de manera consistente en Databricks."""
    display(df.limit(n))


def print_shape(name: str, df: DataFrame):
    rows = df.count()
    cols = len(df.columns)
    print(f"✅ {name}: {rows:,} filas x {cols:,} columnas")


def null_profile(df: DataFrame, cols=None) -> DataFrame:
    """Perfil rápido de nulos para columnas seleccionadas."""
    if cols is None:
        cols = df.columns
    exprs = [
        F.sum(F.when(F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == ""), 1).otherwise(0)).alias(c)
        for c in cols
    ]
    return df.agg(*exprs)


def parse_number_es(col_name: str):
    """
    Convierte números con formato español/latam a double.
    Ejemplos:
    - '5,07' -> 5.07
    - '2.856' -> 2856 cuando parece separador de miles
    - '10.823' -> 10823 cuando parece separador de miles
    - '257' -> 257
    """
    s = F.trim(F.col(col_name).cast("string"))
    s = F.regexp_replace(s, "\\s+", "")
    s = F.when(s.isNull() | (s == ""), None).otherwise(s)
    # Si tiene coma decimal, quitamos puntos de miles y cambiamos coma por punto.
    s_comma = F.regexp_replace(F.regexp_replace(s, "\\.", ""), ",", ".")
    # Si solo tiene puntos y el patrón parece miles: 1.234, 10.823, 123.456
    s_thousands_dot = F.regexp_replace(s, "\\.", "")
    s_normal = s
    return (
        F.when(s.rlike(r"^[0-9]{1,3}(\\.[0-9]{3})+(,[0-9]+)?$"), s_thousands_dot)
         .when(s.rlike(r"^[0-9]+,[0-9]+$"), s_comma)
         .otherwise(s_normal)
         .cast("double")
    )


def to_date_flexible(col_name: str):
    """Convierte fechas comunes d/M/yyyy, dd/MM/yyyy y yyyy-MM-dd."""
    return F.coalesce(
        F.to_date(F.col(col_name).cast("string"), "d/M/yyyy"),
        F.to_date(F.col(col_name).cast("string"), "dd/MM/yyyy"),
        F.to_date(F.col(col_name).cast("string"), "yyyy-MM-dd"),
        F.to_date(F.col(col_name).cast("string"))
    )

print("✅ Funciones auxiliares listas")

# COMMAND ----------

# =========================================
# 3. Cargar tablas Bronze de Lumi y Bagazo
# =========================================

lumi_tables = [
    "customers",
    "geolocation",
    "order_items",
    "order_payments",
    "order_reviews",
    "orders",
    "products",
    "sellers",
    "product_category_translation",
]

missing_tables = []
for t in lumi_tables:
    if not table_exists(LUMI_BRONZE_SCHEMA, t):
        missing_tables.append(f"{CATALOG}.{LUMI_BRONZE_SCHEMA}.{t}")

if not table_exists(BAGAZO_BRONZE_SCHEMA, BAGAZO_TABLE):
    missing_tables.append(f"{CATALOG}.{BAGAZO_BRONZE_SCHEMA}.{BAGAZO_TABLE}")

if missing_tables:
    print("⚠️ Faltan tablas Bronze. Ejecuta primero el notebook de la sesión 2.")
    for t in missing_tables:
        print(" -", t)
    raise ValueError("No se encontraron todas las tablas Bronze requeridas para esta sesión.")

customers_bz = load_table(LUMI_BRONZE_SCHEMA, "customers")
geolocation_bz = load_table(LUMI_BRONZE_SCHEMA, "geolocation")
order_items_bz = load_table(LUMI_BRONZE_SCHEMA, "order_items")
order_payments_bz = load_table(LUMI_BRONZE_SCHEMA, "order_payments")
order_reviews_bz = load_table(LUMI_BRONZE_SCHEMA, "order_reviews")
orders_bz = load_table(LUMI_BRONZE_SCHEMA, "orders")
products_bz = load_table(LUMI_BRONZE_SCHEMA, "products")
sellers_bz = load_table(LUMI_BRONZE_SCHEMA, "sellers")
translation_bz = load_table(LUMI_BRONZE_SCHEMA, "product_category_translation")
bagazo_bz = load_table(BAGAZO_BRONZE_SCHEMA, BAGAZO_TABLE)

print("✅ Todas las tablas Bronze requeridas fueron cargadas como DataFrames")

# COMMAND ----------

# =========================================
# 4. Checkpoint de tablas y conteos
# =========================================

table_inventory = []

for table in lumi_tables:
    df = spark.table(table_name(LUMI_BRONZE_SCHEMA, table))
    table_inventory.append(("Lumi", LUMI_BRONZE_SCHEMA, table, df.count(), len(df.columns)))

bagazo_count = bagazo_bz.count()
table_inventory.append(("Bagazo", BAGAZO_BRONZE_SCHEMA, BAGAZO_TABLE, bagazo_count, len(bagazo_bz.columns)))

inventory_df = spark.createDataFrame(
    table_inventory,
    ["proyecto", "schema", "tabla", "filas", "columnas"]
)

show(inventory_df.orderBy("proyecto", "tabla"), 50)
print("✅ Checkpoint de inventario completado")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. EDA Lumi
# MAGIC
# MAGIC Vamos a usar PySpark para responder preguntas comerciales iniciales:
# MAGIC
# MAGIC - ¿Cuántos pedidos hay?
# MAGIC - ¿Cuántos clientes únicos?
# MAGIC - ¿Qué estados concentran clientes?
# MAGIC - ¿Qué métodos de pago dominan?
# MAGIC - ¿Cómo se distribuyen las reseñas?
# MAGIC - ¿Qué categorías venden más?

# COMMAND ----------

# =========================================
# 5. EDA Lumi: conteos base y primeras preguntas
# =========================================

print_shape("customers_bz", customers_bz)
print_shape("orders_bz", orders_bz)
print_shape("order_items_bz", order_items_bz)
print_shape("sellers_bz", sellers_bz)

print("Columnas principales de orders_bz:")
print(orders_bz.columns)

show(orders_bz, 5)

# COMMAND ----------

# =========================================
# TODO 1 — Ajustar parámetro pedagógico
# =========================================

# Cambia TOP_N a 5, 10 o 15 y vuelve a ejecutar las celdas de ranking.
# Este TODO no bloquea el flujo principal; solo permite practicar parametrización.

TOP_N = 10  # TODO: prueba con 5 o 15
print("TOP_N actualizado:", TOP_N)

# COMMAND ----------

# =========================================
# 6. Clientes por estado
# =========================================

clientes_por_estado = (
    customers_bz
    .groupBy("customer_state")
    .agg(F.countDistinct("customer_unique_id").alias("clientes_unicos"))
    .orderBy(F.desc("clientes_unicos"))
)

show(clientes_por_estado, TOP_N)
print("✅ Análisis de clientes por estado completado")

# COMMAND ----------

# =========================================
# 7. Pedidos por estado de orden
# =========================================

pedidos_por_estado = (
    orders_bz
    .groupBy("order_status")
    .agg(F.count("order_id").alias("total_pedidos"))
    .orderBy(F.desc("total_pedidos"))
)

show(pedidos_por_estado, TOP_N)
print("✅ Análisis de pedidos por estado completado")

# COMMAND ----------

# =========================================
# 8. Métodos de pago
# =========================================

pagos_por_tipo = (
    order_payments_bz
    .groupBy("payment_type")
    .agg(
        F.count("*").alias("transacciones"),
        F.round(F.sum("payment_value"), 2).alias("valor_total"),
        F.round(F.avg("payment_value"), 2).alias("pago_promedio")
    )
    .orderBy(F.desc("valor_total"))
)

show(pagos_por_tipo, TOP_N)
print("✅ Análisis de pagos completado")

# COMMAND ----------

# Import necesario para cálculos con ventanas
from pyspark.sql.window import Window
print("✅ Window importado")

# COMMAND ----------

# =========================================
# 9. Distribución de reseñas
# =========================================

reviews_distribution = (
    order_reviews_bz
    .groupBy("review_score")
    .agg(F.count("*").alias("total_reviews"))
    .withColumn("participacion_pct", F.round(F.col("total_reviews") / F.sum("total_reviews").over(Window.partitionBy()) * 100, 2))
    .orderBy("review_score")
)

show(reviews_distribution, TOP_N)
print("✅ Distribución de reseñas completada")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Join guiado Lumi
# MAGIC
# MAGIC Un producto de datos de negocio rara vez sale de una sola tabla. Ahora combinaremos pedidos, items, productos y traducciones de categoría para construir una vista exploratoria.

# COMMAND ----------

# =========================================
# 10. Join guiado de negocio: orders + items + products + categorías
# =========================================

orders_base = (
    orders_bz
    .select(
        "order_id",
        "customer_id",
        "order_status",
        F.to_timestamp("order_purchase_timestamp").alias("order_purchase_ts"),
        F.to_date("order_purchase_timestamp").alias("order_purchase_date")
    )
)

items_base = (
    order_items_bz
    .select(
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        F.col("price").cast("double").alias("price"),
        F.col("freight_value").cast("double").alias("freight_value")
    )
)

products_base = (
    products_bz
    .select("product_id", "product_category_name")
)

translation_base = (
    translation_bz
    .select("product_category_name", "product_category_name_english")
)

sales_lumi_exploratory = (
    orders_base
    .join(items_base, on="order_id", how="inner")
    .join(products_base, on="product_id", how="left")
    .join(translation_base, on="product_category_name", how="left")
    .withColumn("total_item_value", F.col("price") + F.col("freight_value"))
    .withColumn("year_month", F.date_format("order_purchase_date", "yyyy-MM"))
)

sales_lumi_exploratory.createOrReplaceTempView("vw_sales_lumi_exploratory")

print_shape("sales_lumi_exploratory", sales_lumi_exploratory)
show(sales_lumi_exploratory, 10)

# COMMAND ----------

# =========================================
# 11. Ventas por mes
# =========================================

ventas_por_mes = (
    sales_lumi_exploratory
    .filter(F.col("order_status") == "delivered")
    .groupBy("year_month")
    .agg(
        F.countDistinct("order_id").alias("pedidos"),
        F.round(F.sum("total_item_value"), 2).alias("ventas_estimadas"),
        F.round(F.avg("total_item_value"), 2).alias("ticket_item_promedio")
    )
    .orderBy("year_month")
)

show(ventas_por_mes, 100)
print("✅ Ventas mensuales exploratorias completadas")

# COMMAND ----------

# =========================================
# 12. Top categorías por ventas estimadas
# =========================================

categorias_top = (
    sales_lumi_exploratory
    .filter(F.col("order_status") == "delivered")
    .withColumn(
        "categoria",
        F.coalesce("product_category_name_english", "product_category_name", F.lit("sin_categoria"))
    )
    .groupBy("categoria")
    .agg(
        F.countDistinct("order_id").alias("pedidos"),
        F.round(F.sum("total_item_value"), 2).alias("ventas_estimadas")
    )
    .orderBy(F.desc("ventas_estimadas"))
)

show(categorias_top, TOP_N)
print("✅ Top categorías completado")

# COMMAND ----------

# =========================================
# TODO 2 — Agregación guiada sencilla
# =========================================

# Completa el groupBy para calcular pedidos por estado del cliente.
# Pista: usa customer_state.

pedidos_por_estado_cliente = (
    orders_bz
    .join(customers_bz.select("customer_id", "customer_state"), on="customer_id", how="left")
    # TODO: agrupa por customer_state
    .groupBy("customer_state")
    .agg(F.countDistinct("order_id").alias("pedidos"))
    .orderBy(F.desc("pedidos"))
)

show(pedidos_por_estado_cliente, TOP_N)
print("✅ TODO 2 completado")

# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO 3 — Interpretación rápida
# MAGIC
# MAGIC Escribe una conclusión de negocio con base en los resultados de ventas por categoría.
# MAGIC
# MAGIC **Conclusión:**  
# MAGIC > TODO: redacta aquí una conclusión breve: ¿qué categoría parece más relevante y qué validación pedirías antes de tomar decisiones?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. EDA Bagazo
# MAGIC
# MAGIC Ahora usamos el caso espejo empresarial. El objetivo no es construir todavía un modelo, sino entender la estructura del dato operativo y preparar la base para Silver y MLflow.

# COMMAND ----------

# =========================================
# 13. EDA Bagazo: estructura inicial
# =========================================

print_shape("bagazo_bz", bagazo_bz)
print("Columnas de Bagazo Bronze:")
for c in bagazo_bz.columns:
    print("-", c)

show(bagazo_bz, 5)

# COMMAND ----------

# =========================================
# 14. Rango de fechas Bagazo
# =========================================

from pyspark.sql import functions as F

def to_date_flexible(col_name):
    return F.to_date(
        F.coalesce(
            F.expr(f"try_to_timestamp({col_name}, 'yyyy-MM-dd')"),
            F.expr(f"try_to_timestamp({col_name}, 'dd/MM/yyyy')"),
            F.expr(f"try_to_timestamp({col_name}, 'd/M/yyyy')"),
            F.expr(f"try_to_timestamp({col_name}, 'dd-MM-yyyy')"),
            F.expr(f"try_to_timestamp({col_name}, 'yyyy/MM/dd')"),
            F.expr(f"try_to_timestamp({col_name}, 'yyyyMMdd')")
        )
    )


bagazo_base = (
    bagazo_bz
    .withColumn("fecha_dt", to_date_flexible("fecha"))
)

rango_fechas = bagazo_base.agg(
    F.min("fecha_dt").alias("fecha_min"),
    F.max("fecha_dt").alias("fecha_max"),
    F.count("*").alias("registros"),
    F.count(F.when(F.col("fecha_dt").isNull(), True)).alias("fechas_no_parseadas")
)

show(rango_fechas, 1)

print("✅ Rango de fechas validado")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Transformación exploratoria ancho → largo
# MAGIC
# MAGIC El archivo Bagazo viene en formato ancho: una columna por variable e ingenio. Para analizar y modelar, conviene llevarlo a formato largo:
# MAGIC
# MAGIC ```text
# MAGIC fecha | ingenio | lluvia_mm | cana_molida_ton | bagazo_entregado_ton | comentario
# MAGIC ```

# COMMAND ----------

from pyspark.sql import functions as F

# =========================================
# 15. Transformación Bagazo: formato ancho -> formato largo
# =========================================

# 1. Renombramiento definitivo de columnas con problema por la ñ
rename_map = {
    "caa_molida_providencia_toneladas": "cana_molida_providencia_toneladas",
    "caa_molida_ilc_toneladas": "cana_molida_ilc_toneladas",
    "caa_molida_incauca_toneladas": "cana_molida_incauca_toneladas"
}

for old_col, new_col in rename_map.items():
    if old_col in bagazo_base.columns and new_col not in bagazo_base.columns:
        bagazo_base = bagazo_base.withColumnRenamed(old_col, new_col)

print("✅ Renombramiento validado")
print(bagazo_base.columns)


# 2. Función segura para números en formato latino/español
# Ejemplos:
# "1.486,2" -> 1486.2
# "1486,2"  -> 1486.2
# "1,486.2" -> 1486.2
# "1486.2"  -> 1486.2

def parse_number_es_safe(col_name):
    texto = F.trim(F.col(col_name).cast("string"))

    limpio = F.regexp_replace(
        texto,
        r"[^0-9,.\-]",
        ""
    )

    normalizado = (
        F.when(
            limpio.rlike(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$"),
            F.regexp_replace(F.regexp_replace(limpio, r"\.", ""), ",", ".")
        )
        .when(
            limpio.rlike(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$"),
            F.regexp_replace(limpio, ",", "")
        )
        .when(
            limpio.rlike(r"^-?\d+,\d+$"),
            F.regexp_replace(limpio, ",", ".")
        )
        .otherwise(limpio)
    )

    numero_seguro = (
        F.when(
            texto.isNull() | (texto == "") | limpio.isin("", "-", ",", ".", "-,", "-."),
            F.lit(None)
        )
        .when(
            normalizado.rlike(r"^-?\d+(\.\d+)?$"),
            normalizado
        )
        .otherwise(F.lit(None))
    )

    return numero_seguro.cast("double")


# 3. Validación de columnas necesarias
required_cols = [
    "fecha_dt",

    "promedio_lluvias_providencia_mm",
    "cana_molida_providencia_toneladas",
    "bagazo_entregado_providencia_toneladas",
    "comentarios_providencia",

    "promedio_lluvias_ilc_mm",
    "cana_molida_ilc_toneladas",
    "bagazo_entregado_ilc_toneladas",
    "comentarios_ilc",

    "promedio_lluvias_incauca_mm",
    "cana_molida_incauca_toneladas",
    "bagazo_entregado_incauca_toneladas",
    "comentarios_incauca"
]

missing_cols = [c for c in required_cols if c not in bagazo_base.columns]

if missing_cols:
    raise ValueError(
        "❌ Faltan columnas requeridas:\n"
        + "\n".join(missing_cols)
        + "\n\nColumnas disponibles:\n"
        + "\n".join(bagazo_base.columns)
    )

print("✅ Columnas requeridas validadas")


# 4. Construcción formato largo ya con columnas numéricas seguras

bagazo_providencia = bagazo_base.select(
    F.col("fecha_dt").alias("fecha"),
    F.lit("Providencia").alias("ingenio"),
    parse_number_es_safe("promedio_lluvias_providencia_mm").alias("lluvia_mm"),
    parse_number_es_safe("cana_molida_providencia_toneladas").alias("cana_molida_ton"),
    parse_number_es_safe("bagazo_entregado_providencia_toneladas").alias("bagazo_entregado_ton"),
    F.col("comentarios_providencia").alias("comentario")
)

bagazo_ilc = bagazo_base.select(
    F.col("fecha_dt").alias("fecha"),
    F.lit("ILC").alias("ingenio"),
    parse_number_es_safe("promedio_lluvias_ilc_mm").alias("lluvia_mm"),
    parse_number_es_safe("cana_molida_ilc_toneladas").alias("cana_molida_ton"),
    parse_number_es_safe("bagazo_entregado_ilc_toneladas").alias("bagazo_entregado_ton"),
    F.col("comentarios_ilc").alias("comentario")
)

bagazo_incauca = bagazo_base.select(
    F.col("fecha_dt").alias("fecha"),
    F.lit("Incauca").alias("ingenio"),
    parse_number_es_safe("promedio_lluvias_incauca_mm").alias("lluvia_mm"),
    parse_number_es_safe("cana_molida_incauca_toneladas").alias("cana_molida_ton"),
    parse_number_es_safe("bagazo_entregado_incauca_toneladas").alias("bagazo_entregado_ton"),
    F.col("comentarios_incauca").alias("comentario")
)

bagazo_long_exploratory = (
    bagazo_providencia
    .unionByName(bagazo_ilc)
    .unionByName(bagazo_incauca)
    .withColumn("anio", F.year(F.col("fecha")))
    .withColumn("mes", F.month(F.col("fecha")))
    .withColumn("year_month", F.date_format(F.col("fecha"), "yyyy-MM"))
    .withColumn(
        "tiene_comentario_operativo",
        F.col("comentario").isNotNull() & (F.trim(F.col("comentario")) != "")
    )
)

bagazo_long_exploratory.createOrReplaceTempView("vw_bagazo_long_exploratory")

print_shape("bagazo_long_exploratory", bagazo_long_exploratory)
show(bagazo_long_exploratory, 20)

print("✅ Transformación Bagazo formato ancho -> largo completada sin errores de casteo")

# COMMAND ----------

from pyspark.sql import functions as F

# =========================================
# 16. Métricas operativas por ingenio
# =========================================

# Conversor robusto de números en formato español/latino:
# Ejemplos:
# "1.486,2"  -> 1486.2
# "1486,2"   -> 1486.2
# "1,486.2"  -> 1486.2
# "1486.2"   -> 1486.2
# ""         -> NULL
# textos raros -> NULL

def safe_double_es_expr(col_name):
    c = f"`{col_name}`"
    
    raw = f"""
        regexp_replace(
            trim(cast({c} as string)),
            '[^0-9,.-]',
            ''
        )
    """

    normalized = f"""
        CASE
            WHEN {c} IS NULL OR trim(cast({c} as string)) = '' THEN NULL

            WHEN {raw} IN ('', '-', ',', '.', '-,', '-.') THEN NULL

            -- Caso español: 1.486,2 -> 1486.2
            WHEN instr({raw}, ',') > 0
                 AND instr({raw}, '.') > 0
                 AND instr({raw}, '.') < instr({raw}, ',')
                THEN replace(replace({raw}, '.', ''), ',', '.')

            -- Caso inglés: 1,486.2 -> 1486.2
            WHEN instr({raw}, ',') > 0
                 AND instr({raw}, '.') > 0
                 AND instr({raw}, ',') < instr({raw}, '.')
                THEN replace({raw}, ',', '')

            -- Caso decimal con coma: 1486,2 -> 1486.2
            WHEN instr({raw}, ',') > 0
                THEN replace({raw}, ',', '.')

            -- Caso normal: 1486.2
            ELSE {raw}
        END
    """

    return F.expr(f"try_cast({normalized} AS DOUBLE)")


# 1. Crear versión numérica segura para métricas
bagazo_long_metricas = (
    bagazo_long_exploratory
    .withColumn("lluvia_mm_num", safe_double_es_expr("lluvia_mm"))
    .withColumn("cana_molida_ton_num", safe_double_es_expr("cana_molida_ton"))
    .withColumn("bagazo_entregado_ton_num", safe_double_es_expr("bagazo_entregado_ton"))
)


# 2. Validación rápida de conversión
validacion_numerica = bagazo_long_metricas.agg(
    F.count("*").alias("registros"),
    F.sum(
        F.when(
            F.col("lluvia_mm").isNotNull() & F.col("lluvia_mm_num").isNull(),
            1
        ).otherwise(0)
    ).alias("lluvia_no_convertida"),
    F.sum(
        F.when(
            F.col("cana_molida_ton").isNotNull() & F.col("cana_molida_ton_num").isNull(),
            1
        ).otherwise(0)
    ).alias("cana_no_convertida"),
    F.sum(
        F.when(
            F.col("bagazo_entregado_ton").isNotNull() & F.col("bagazo_entregado_ton_num").isNull(),
            1
        ).otherwise(0)
    ).alias("bagazo_no_convertido")
)

show(validacion_numerica, 1)


# 3. Métricas operativas por ingenio
metricas_por_ingenio = (
    bagazo_long_metricas
    .groupBy("ingenio")
    .agg(
        F.count("*").alias("dias_registrados"),
        F.round(F.avg("lluvia_mm_num"), 2).alias("lluvia_promedio_mm"),
        F.round(F.avg("cana_molida_ton_num"), 2).alias("cana_promedio_ton"),
        F.round(F.avg("bagazo_entregado_ton_num"), 2).alias("bagazo_promedio_ton"),
        F.round(F.stddev("bagazo_entregado_ton_num"), 2).alias("variabilidad_bagazo"),
        F.sum(
            F.when(F.col("cana_molida_ton_num") == 0, 1).otherwise(0)
        ).alias("dias_cana_cero"),
        F.sum(
            F.when(F.col("bagazo_entregado_ton_num") == 0, 1).otherwise(0)
        ).alias("dias_bagazo_cero"),
        F.sum(
            F.when(F.col("tiene_comentario_operativo"), 1).otherwise(0)
        ).alias("dias_con_comentario")
    )
    .orderBy(F.desc("bagazo_promedio_ton"))
)

show(metricas_por_ingenio, 20)

print("✅ Métricas por ingenio completadas")

# COMMAND ----------

# =========================================
# 17. Lluvia y bagazo por mes
# =========================================

bagazo_mensual = (
    bagazo_long_exploratory
    .groupBy("year_month", "ingenio")
    .agg(
        F.round(F.avg("lluvia_mm"), 2).alias("lluvia_promedio_mm"),
        F.round(F.avg("bagazo_entregado_ton"), 2).alias("bagazo_promedio_ton"),
        F.round(F.avg("cana_molida_ton"), 2).alias("cana_promedio_ton")
    )
    .orderBy("year_month", "ingenio")
)

show(bagazo_mensual, 100)
print("✅ Análisis mensual Bagazo completado")

# COMMAND ----------

# =========================================
# 18. Comentarios operativos
# =========================================

comentarios_operativos = (
    bagazo_long_exploratory
    .filter(F.col("tiene_comentario_operativo"))
    .groupBy("ingenio", "comentario")
    .agg(F.count("*").alias("dias"))
    .orderBy(F.desc("dias"))
)

show(comentarios_operativos, 30)
print("✅ Comentarios operativos explorados")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Consulta SQL opcional sobre las vistas temporales creadas con PySpark.
# MAGIC -- Sirve para mostrar que el mismo DataFrame puede ser consultado por perfiles SQL.
# MAGIC
# MAGIC SELECT
# MAGIC   ingenio,
# MAGIC   ROUND(AVG(lluvia_mm), 2) AS lluvia_promedio_mm,
# MAGIC   ROUND(AVG(bagazo_entregado_ton), 2) AS bagazo_promedio_ton,
# MAGIC   COUNT(*) AS dias
# MAGIC FROM vw_bagazo_long_exploratory
# MAGIC GROUP BY ingenio
# MAGIC ORDER BY bagazo_promedio_ton DESC;

# COMMAND ----------

# =========================================
# 19. Diagnóstico para Silver
# =========================================

problemas_lumi = spark.createDataFrame([
    ("orders", "Fechas como texto", "Convertir timestamps y fechas en Silver"),
    ("orders", "Estados de orden mezclan operaciones y negocio", "Estandarizar estado y calcular entregas completadas"),
    ("products", "Categorías en portugués y posibles nulos", "Traducir categorías y asignar sin_categoria"),
    ("payments", "Puede haber múltiples pagos por pedido", "Consolidar pagos por order_id"),
    ("reviews", "Comentarios textuales con muchos nulos", "Mantener score y bandera de comentario")
], ["tabla", "hallazgo", "accion_silver"])

problemas_bagazo = spark.createDataFrame([
    ("bagazo", "Archivo original en formato ancho", "Llevar a formato largo por fecha e ingenio"),
    ("bagazo", "Nombres de columnas con espacios, tildes y saltos", "Estandarizar nombres snake_case"),
    ("bagazo", "Números con formato regional", "Normalizar lluvia, caña y bagazo como double"),
    ("bagazo", "Ceros operativos", "Diferenciar cero real, parada, mantenimiento o dato faltante"),
    ("bagazo", "Comentarios operativos no estructurados", "Crear bandera tiene_comentario_operativo")
], ["tabla", "hallazgo", "accion_silver"])

print("Hallazgos Lumi para Silver:")
show(problemas_lumi, 20)

print("Hallazgos Bagazo para Silver:")
show(problemas_bagazo, 20)

print("✅ Diagnóstico Silver listo para la sesión 4")

# COMMAND ----------

# =========================================
# TODO 4 — Regla candidata para Silver
# =========================================

# Completa una regla candidata de calidad para la sesión 4.
# Ejemplo: lluvia no debería ser negativa.

regla_silver_bagazo = "lluvia_mm >= 0"  # TODO: cambia o agrega otra regla si aplica
print("Regla candidata Silver Bagazo:", regla_silver_bagazo)

# COMMAND ----------

# MAGIC %md
# MAGIC # Retos finales de la sesión
# MAGIC
# MAGIC Los retos quedan al final para no interrumpir el flujo principal. Puedes resolverlos en clase si hay tiempo o dejarlos como práctica posterior.
# MAGIC
# MAGIC ## Reto Nivel 1 — Inventario Bronze
# MAGIC
# MAGIC **Objetivo:** identificar la tabla con más filas, menos filas y más columnas.  
# MAGIC **Tiempo estimado:** 10 minutos.  
# MAGIC **Resultado esperado:** un DataFrame con las tres observaciones.
# MAGIC
# MAGIC ## Reto Nivel 2 — Métricas comerciales Lumi
# MAGIC
# MAGIC **Objetivo:** calcular ventas por mes, ticket promedio, top categorías, top estados y review promedio.  
# MAGIC **Tiempo estimado:** 20 minutos.  
# MAGIC **Resultado esperado:** 3 a 5 tablas exploratorias listas para discutir.
# MAGIC
# MAGIC ## Reto consultor — Hallazgos ejecutivos
# MAGIC
# MAGIC **Objetivo:** transformar resultados técnicos en conversación de negocio.  
# MAGIC **Tiempo estimado:** 15 minutos.  
# MAGIC **Formato esperado:**
# MAGIC
# MAGIC ```text
# MAGIC Insight:
# MAGIC La estabilidad operativa del suministro de bagazo depende críticamente del comportamiento climático y de las paradas programadas de los ingenios, afectando directamente a la planta principal.
# MAGIC
# MAGIC Evidencia:
# MAGIC Al transformar los datos a formato largo, se descubrió que el ingenio ILC tiene el promedio diario de molienda de caña más bajo (2,957.02 ton) pero registra el mayor número de días en cero por mantenimiento (75 días). Adicionalmente, Incauca registra 42 días de inactividad operativa con el comentario explícito "Falta de caña por lluvias".
# MAGIC
# MAGIC Riesgo o impacto:
# MAGIC Si el modelo predictivo de Machine Learning que se construirá en las siguientes fases no detecta la estacionalidad de lluvias ni los eventos de mantenimiento (ceros operativos), generará falsas expectativas de inventario de bagazo, arriesgando la continuidad de la producción de papel/energía.
# MAGIC
# MAGIC Recomendación:
# MAGIC Para la capa Silver, no basta con limpiar nulos. Debemos crear obligatoriamente una columna "flag" o bandera que clasifique si un valor "0" es una parada por mantenimiento o un problema de desabastecimiento por clima. Además, se sugiere cruzar esta data con un API meteorológico histórico externo antes de entrenar modelos en MLflow.
# MAGIC
# MAGIC Tabla o consulta usada:
# MAGIC Consultas ejecutadas sobre las vistas temporales 'vw_bagazo_long_exploratory' mediante agrupaciones de métricas operativas y comentarios operativos.
# MAGIC ```

# COMMAND ----------

# =========================================
# Espacio de trabajo para retos
# =========================================

# Reto Nivel 1: usa inventory_df
tabla_mas_filas = inventory_df.orderBy(F.desc("filas")).first()
print(f" Tabla con MÁS filas: '{tabla_mas_filas['tabla']}' ({tabla_mas_filas['filas']:,} filas)")
tabla_menos_filas = inventory_df.orderBy(F.asc("filas")).first()
print(f"Tabla con MENOS filas: '{tabla_menos_filas['tabla']}' ({tabla_menos_filas['filas']:,} filas)")
tabla_mas_columnas = inventory_df.orderBy(F.desc("columnas")).first()
print(f"Tabla con MÁS columnas: '{tabla_mas_columnas['tabla']}' ({tabla_mas_columnas['columnas']} columnas)")

# Reto Nivel 2: usa sales_lumi_exploratory, customers_bz y order_reviews_bz
ventas_por_estado = (
    sales_lumi_exploratory
    .filter(F.col("order_status") == "delivered")
    .join(customers_bz.select("customer_id", "customer_state"), on="customer_id", how="left")
    .groupBy("customer_state")
    .agg(
        F.countDistinct("order_id").alias("pedidos"),
        F.round(F.sum("total_item_value"), 2).alias("ventas_estimadas")
    )
    .orderBy(F.desc("ventas_estimadas"))
)

print("Top Estados por Ventas:")
show(ventas_por_estado, 5)

reviews_limpio = order_reviews_bz.withColumn("review_score_num", F.col("review_score").cast("int"))
calidad_por_categoria = (
    sales_lumi_exploratory
    .withColumn("categoria", F.coalesce("product_category_name_english", "product_category_name", F.lit("sin_categoria")))
    .join(reviews_limpio.select("order_id", "review_score_num"), on="order_id", how="inner")
    .groupBy("categoria")
    .agg(
        F.count("order_id").alias("total_resenas"),
        F.round(F.avg("review_score_num"), 2).alias("review_promedio")
    )
    .orderBy(F.desc("total_resenas")) # Ordenado por relevancia (las que más se venden)
)

print("Calidad y Review Promedio por Categoría:")
show(calidad_por_categoria, TOP_N)

# Reto consultor: usa cualquier resultado obtenido durante la sesión



# COMMAND ----------

# MAGIC %md
# MAGIC # Cierre de la sesión
# MAGIC
# MAGIC ## Aprendizajes clave
# MAGIC
# MAGIC - PySpark permite explorar datos grandes con operaciones familiares: `select`, `filter`, `groupBy`, `agg`, `withColumn`, `join`.
# MAGIC - Bronze no debe asumirse como confiable: primero se inspecciona.
# MAGIC - Lumi permite conectar datos transaccionales con preguntas comerciales.
# MAGIC - Bagazo muestra por qué muchas veces hay que rediseñar la estructura de datos antes de hacer analítica o ML.
# MAGIC - La sesión 4 usará estos hallazgos para construir Silver.
# MAGIC
# MAGIC ## Preparación para la sesión 4
# MAGIC
# MAGIC Debe quedar claro qué transformaciones se aplicarán:
# MAGIC
# MAGIC - Fechas y tipos de datos.
# MAGIC - Categorías traducidas.
# MAGIC - Pagos consolidados.
# MAGIC - Bagazo en formato largo.
# MAGIC - Reglas de calidad.
# MAGIC - Banderas operativas.

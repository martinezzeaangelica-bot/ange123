# Databricks notebook source
# MAGIC %md
# MAGIC # Sesión 2 — Ingesta de datos y capa Bronze
# MAGIC
# MAGIC **Notebook del estudiante — versión fluida y completa**  
# MAGIC Proyecto principal: **Lumi Commerce Lakehouse**  
# MAGIC Caso espejo empresarial: **Lluvia, Caña y Bagazo**
# MAGIC
# MAGIC ## Propósito de este notebook
# MAGIC
# MAGIC Este notebook está diseñado para que la clase fluya sin que el estudiante tenga que completar demasiadas piezas de código.
# MAGIC
# MAGIC La idea es que el grupo aprenda el flujo real de trabajo en Databricks:
# MAGIC
# MAGIC 1. Confirmar rutas de archivos en **Volumes**.
# MAGIC 2. Crear esquemas Bronze.
# MAGIC 3. Leer archivos CSV desde `/Volumes`.
# MAGIC 4. Normalizar nombres de columnas.
# MAGIC 5. Escribir tablas Delta administradas.
# MAGIC 6. Registrar auditoría de carga.
# MAGIC 7. Validar conteos, columnas y tablas generadas.
# MAGIC
# MAGIC > En esta versión solo deberías ajustar la celda de parámetros si tu workspace usa otros nombres de catálogo, esquema o ruta.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Contexto de la sesión
# MAGIC
# MAGIC En la sesión 1 ya quedó creada la estructura del repositorio GitHub y se definió una organización tipo Kedro para el proyecto.
# MAGIC
# MAGIC En esta sesión trabajaremos con dos zonas:
# MAGIC
# MAGIC | Zona | Uso |
# MAGIC |---|---|
# MAGIC | `notebooks/` | Notebooks versionados en GitHub |
# MAGIC | `/Volumes/workspace/raw/lumi` | Archivos crudos de Lumi Commerce |
# MAGIC | `/Volumes/workspace/raw/bagazo` | Archivo crudo del caso lluvia/caña/bagazo |
# MAGIC | `workspace.lumi_bronze` | Tablas Bronze de Lumi |
# MAGIC | `workspace.bagazo_bronze` | Tabla Bronze del caso Bagazo |
# MAGIC | `workspace.control` | Auditoría de cargas |
# MAGIC
# MAGIC La regla de buenas prácticas es simple: **GitHub versiona código; Volumes almacena archivos de datos**.

# COMMAND ----------

# =========================================
# 0. Validación inicial del entorno
# =========================================

print("Spark version:", spark.version)

current_catalog = spark.sql("SELECT current_catalog() AS catalog").collect()[0]["catalog"]
print("Current catalog:", current_catalog)

# En Databricks Free Edition normalmente el catálogo de trabajo es `workspace`.
# Aun así, este notebook permite ajustar el catálogo manualmente en la siguiente celda.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Parámetros principales
# MAGIC
# MAGIC Esta es la celda más importante de configuración.  
# MAGIC En condiciones normales, los estudiantes solo deberían verificar estas rutas:
# MAGIC
# MAGIC ```text
# MAGIC /Volumes/workspace/raw/lumi
# MAGIC /Volumes/workspace/raw/bagazo
# MAGIC ```
# MAGIC
# MAGIC Si el instructor creó otros nombres de catálogo o volumen, se ajustan aquí una sola vez.

# COMMAND ----------

# =========================================
# 1. Parámetros de trabajo
# =========================================

CATALOG = "workspace"

LUMI_RAW_PATH = "/Volumes/workspace/raw/lumi"
BAGAZO_RAW_PATH = "/Volumes/workspace/raw/bagazo"

LUMI_BRONZE_SCHEMA = "lumi_bronze"
BAGAZO_BRONZE_SCHEMA = "bagazo_bronze"
CONTROL_SCHEMA = "control"

AUDIT_TABLE = "bronze_ingestion_audit"

print("CATALOG:", CATALOG)
print("LUMI_RAW_PATH:", LUMI_RAW_PATH)
print("BAGAZO_RAW_PATH:", BAGAZO_RAW_PATH)
print("LUMI_BRONZE_SCHEMA:", LUMI_BRONZE_SCHEMA)
print("BAGAZO_BRONZE_SCHEMA:", BAGAZO_BRONZE_SCHEMA)
print("CONTROL_SCHEMA:", CONTROL_SCHEMA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validación de archivos cargados en Volumes
# MAGIC
# MAGIC Antes de crear tablas Bronze, validamos que los archivos estén realmente en los volúmenes.  
# MAGIC Esta celda sirve como primer checkpoint de clase.

# COMMAND ----------

# =========================================
# 2. Validar archivos en Volumes
# =========================================

print("Archivos encontrados en Lumi:")
display(dbutils.fs.ls(LUMI_RAW_PATH))

print("Archivos encontrados en Bagazo:")
display(dbutils.fs.ls(BAGAZO_RAW_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Diccionario de datasets esperados
# MAGIC
# MAGIC Aquí dejamos explícito qué archivos esperamos, qué tabla Bronze se creará y cuántas filas debería tener cada archivo.
# MAGIC
# MAGIC Esto ayuda a explicar una práctica muy importante: **una ingesta profesional no solo carga datos; también valida lo cargado**.

# COMMAND ----------

# =========================================
# 3. Configuración de datasets Lumi y Bagazo
# =========================================

LUMI_DATASETS = {
    "customers": {
        "file": "olist_customers_dataset.csv",
        "expected_rows": 99441,
        "expected_columns": [
            "customer_id", "customer_unique_id", "customer_zip_code_prefix",
            "customer_city", "customer_state"
        ]
    },
    "geolocation": {
        "file": "olist_geolocation_dataset.csv",
        "expected_rows": 1000163,
        "expected_columns": [
            "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
            "geolocation_city", "geolocation_state"
        ]
    },
    "order_items": {
        "file": "olist_order_items_dataset.csv",
        "expected_rows": 112650,
        "expected_columns": [
            "order_id", "order_item_id", "product_id", "seller_id",
            "shipping_limit_date", "price", "freight_value"
        ]
    },
    "order_payments": {
        "file": "olist_order_payments_dataset.csv",
        "expected_rows": 103886,
        "expected_columns": [
            "order_id", "payment_sequential", "payment_type",
            "payment_installments", "payment_value"
        ]
    },
    "order_reviews": {
        "file": "olist_order_reviews_dataset.csv",
        "expected_rows": 99224,
        "expected_columns": [
            "review_id", "order_id", "review_score", "review_comment_title",
            "review_comment_message", "review_creation_date", "review_answer_timestamp"
        ]
    },
    "orders": {
        "file": "olist_orders_dataset.csv",
        "expected_rows": 99441,
        "expected_columns": [
            "order_id", "customer_id", "order_status", "order_purchase_timestamp",
            "order_approved_at", "order_delivered_carrier_date",
            "order_delivered_customer_date", "order_estimated_delivery_date"
        ]
    },
    "products": {
        "file": "olist_products_dataset.csv",
        "expected_rows": 32951,
        "expected_columns": [
            "product_id", "product_category_name", "product_name_lenght",
            "product_description_lenght", "product_photos_qty", "product_weight_g",
            "product_length_cm", "product_height_cm", "product_width_cm"
        ]
    },
    "sellers": {
        "file": "olist_sellers_dataset.csv",
        "expected_rows": 3095,
        "expected_columns": [
            "seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"
        ]
    },
    "product_category_translation": {
        "file": "product_category_name_translation.csv",
        "expected_rows": 71,
        "expected_columns": [
            "product_category_name", "product_category_name_english"
        ]
    }
}

BAGAZO_DATASET = "molienda_bagazo_y_lluvias_II"
BAGAZO_EXPECTED_ROWS = 798
BAGAZO_EXPECTED_COLUMNS = [
    "fecha",
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

print("Datasets Lumi configurados:", len(LUMI_DATASETS))
print("Dataset Bagazo configurado:", BAGAZO_DATASET)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Crear esquemas Bronze y esquema de control
# MAGIC
# MAGIC Un esquema en Databricks nos permite organizar tablas por dominio o capa.
# MAGIC
# MAGIC En esta sesión usaremos:
# MAGIC
# MAGIC - `lumi_bronze`: datos crudos del e-commerce.
# MAGIC - `bagazo_bronze`: datos crudos del caso operativo.
# MAGIC - `control`: auditoría de cargas.

# COMMAND ----------

# =========================================
# 4. Crear esquemas
# =========================================

schemas = [LUMI_BRONZE_SCHEMA, BAGAZO_BRONZE_SCHEMA, CONTROL_SCHEMA]

for schema in schemas:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{schema}`")
    print(f"✅ Esquema listo: {CATALOG}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Funciones auxiliares
# MAGIC
# MAGIC Estas funciones hacen que el flujo sea repetible:
# MAGIC
# MAGIC - Normalizan nombres de columnas.
# MAGIC - Detectan separador de CSV.
# MAGIC - Eliminan columnas vacías accidentales.
# MAGIC - Escriben tablas Delta.
# MAGIC - Registran auditoría.
# MAGIC
# MAGIC No es necesario memorizar todas las funciones. Lo importante es entender **por qué existen**: reducen trabajo manual y evitan errores repetidos.

# COMMAND ----------

# =========================================
# 5. Funciones auxiliares
# =========================================

import re
import unicodedata
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType


def normalize_column_name(col_name: str) -> str:
    # Convierte nombres de columnas a snake_case sin tildes ni caracteres especiales.
    col_name = str(col_name)
    col_name = col_name.replace("\n", " ")
    col_name = unicodedata.normalize("NFKD", col_name)
    col_name = col_name.encode("ascii", "ignore").decode("ascii")
    col_name = col_name.lower().strip()
    col_name = re.sub(r"[^a-z0-9]+", "_", col_name)
    col_name = re.sub(r"_+", "_", col_name)
    col_name = col_name.strip("_")
    return col_name


def normalize_columns(df):
    # Normaliza columnas y evita duplicados si dos nombres terminan iguales.
    new_names = []
    seen = {}

    for old_col in df.columns:
        base_name = normalize_column_name(old_col)
        if base_name == "":
            base_name = "columna_sin_nombre"

        count = seen.get(base_name, 0)
        new_name = base_name if count == 0 else f"{base_name}_{count + 1}"
        seen[base_name] = count + 1
        new_names.append(new_name)

    for old_col, new_col in zip(df.columns, new_names):
        df = df.withColumnRenamed(old_col, new_col)

    return df


def drop_empty_generated_columns(df):
    # Elimina columnas tipo c13, c14, c15 si vienen de separadores sobrantes y están vacías.
    suspect_cols = [c for c in df.columns if re.fullmatch(r"c\d+", c)]

    if not suspect_cols:
        return df

    exprs = [
        F.count(
            F.when(
                F.col(c).isNotNull() & (F.trim(F.col(c).cast("string")) != ""),
                c
            )
        ).alias(c)
        for c in suspect_cols
    ]

    counts = df.agg(*exprs).collect()[0].asDict()
    cols_to_drop = [c for c, non_empty_count in counts.items() if non_empty_count == 0]

    if cols_to_drop:
        print("Columnas vacías eliminadas:", cols_to_drop)
        df = df.drop(*cols_to_drop)

    return df


def read_csv_auto(path: str):
    # Lee un CSV detectando separador entre coma, punto y coma, tab y pipe.
    candidates = []

    for sep in [",", ";", "\t", "|"]:
        temp_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .option("sep", sep)
            .option("multiLine", True)
            .option("quote", '"')
            .option("escape", '"')
            .csv(path)
        )
        candidates.append((sep, len(temp_df.columns), temp_df))

    best_sep, best_cols, best_df = max(candidates, key=lambda x: x[1])

    print(f"Separador detectado: '{best_sep}'")
    print(f"Columnas detectadas antes de limpieza: {best_cols}")

    best_df = normalize_columns(best_df)
    best_df = drop_empty_generated_columns(best_df)

    print(f"Columnas finales: {len(best_df.columns)}")
    return best_df


def validate_columns(df, expected_columns):
    # Compara columnas reales contra columnas esperadas.
    actual_columns = df.columns
    missing = [c for c in expected_columns if c not in actual_columns]
    extra = [c for c in actual_columns if c not in expected_columns]

    if not missing:
        print("✅ Columnas esperadas OK")
    else:
        print("⚠️ Columnas faltantes:", missing)

    if extra:
        print("ℹ️ Columnas adicionales:", extra)

    return missing, extra


def table_fqn(schema_name: str, table_name: str) -> str:
    # Nombre fully qualified con backticks.
    return f"`{CATALOG}`.`{schema_name}`.`{table_name}`"


def write_delta_table(df, schema_name: str, table_name: str):
    # Escribe un DataFrame como tabla Delta administrada.
    fqn = table_fqn(schema_name, table_name)
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(fqn)
    )
    print(f"✅ Tabla Delta creada/actualizada: {CATALOG}.{schema_name}.{table_name}")


def register_audit(audit_rows, dataset, archivo_origen, tabla_destino, filas_cargadas, columnas, estado, observaciones):
    # Agrega una fila de auditoría a una lista en memoria.
    audit_rows.append({
        "fecha_carga": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "archivo_origen": archivo_origen,
        "tabla_destino": tabla_destino,
        "filas_cargadas": int(filas_cargadas) if filas_cargadas is not None else None,
        "columnas": int(columnas) if columnas is not None else None,
        "estado": estado,
        "observaciones": observaciones
    })

print("✅ Funciones auxiliares listas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Ingesta de Lumi hacia tablas Bronze
# MAGIC
# MAGIC Ahora cargaremos los 9 archivos de Lumi desde el volumen `raw/lumi` y los escribiremos como tablas Delta en `lumi_bronze`.
# MAGIC
# MAGIC Este es el corazón de la sesión: **pasar de archivos crudos a tablas consultables en Databricks**.

# COMMAND ----------

# =========================================
# 6. Ingesta Lumi: CSV -> Delta Bronze
# =========================================

audit_rows = []
loaded_lumi_tables = []

for dataset_name, cfg in LUMI_DATASETS.items():
    file_name = cfg["file"]
    expected_rows = cfg["expected_rows"]
    expected_columns = cfg["expected_columns"]

    source_path = f"{LUMI_RAW_PATH}/{file_name}"
    target_table = dataset_name

    print("\n" + "=" * 90)
    print(f"Cargando dataset: {dataset_name}")
    print(f"Archivo origen: {source_path}")
    print(f"Tabla destino: {CATALOG}.{LUMI_BRONZE_SCHEMA}.{target_table}")

    try:
        df = read_csv_auto(source_path)
        missing, extra = validate_columns(df, expected_columns)

        rows = df.count()
        cols = len(df.columns)

        if rows == expected_rows:
            row_msg = f"Conteo OK: {rows:,} filas"
            print(f"✅ {row_msg}")
        else:
            row_msg = f"Conteo diferente. Esperado: {expected_rows:,}; obtenido: {rows:,}"
            print(f"⚠️ {row_msg}")

        write_delta_table(df, LUMI_BRONZE_SCHEMA, target_table)
        loaded_lumi_tables.append(target_table)

        estado = "OK" if not missing and rows == expected_rows else "REVISAR"
        obs = f"{row_msg}. Faltantes: {missing}. Extras: {extra}"

        register_audit(
            audit_rows,
            dataset_name,
            file_name,
            f"{CATALOG}.{LUMI_BRONZE_SCHEMA}.{target_table}",
            rows,
            cols,
            estado,
            obs
        )

    except Exception as e:
        print(f"❌ Error cargando {dataset_name}: {e}")
        register_audit(
            audit_rows,
            dataset_name,
            file_name,
            f"{CATALOG}.{LUMI_BRONZE_SCHEMA}.{target_table}",
            None,
            None,
            "ERROR",
            str(e)
        )

print("\nTablas Lumi cargadas:", loaded_lumi_tables)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validación rápida de tablas Lumi
# MAGIC
# MAGIC En una ingesta real siempre se valida que las tablas existan, que tengan filas y que sean consultables.

# COMMAND ----------

# =========================================
# 7. Validaciones Lumi
# =========================================

validation_rows = []

for table_name in loaded_lumi_tables:
    fqn = table_fqn(LUMI_BRONZE_SCHEMA, table_name)
    count_rows = spark.table(fqn).count()
    count_cols = len(spark.table(fqn).columns)
    validation_rows.append((table_name, count_rows, count_cols))

validation_df = spark.createDataFrame(validation_rows, ["tabla", "filas", "columnas"])
display(validation_df.orderBy("tabla"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Preparar archivo Bagazo
# MAGIC
# MAGIC El caso Bagazo puede llegar como `.csv` o como `.xlsx`.
# MAGIC
# MAGIC Para una clase fluida, lo ideal es trabajar con CSV. Esta celda hace lo siguiente:
# MAGIC
# MAGIC 1. Si ya existe CSV, lo usa.
# MAGIC 2. Si no existe CSV pero sí existe Excel, intenta convertirlo a CSV usando `pandas`.
# MAGIC 3. Si no encuentra ninguno, muestra un error claro.

# COMMAND ----------

# =========================================
# 8. Preparar archivo Bagazo
# =========================================

import os


def prepare_bagazo_csv(raw_path: str, output_file: str = "molienda_bagazo_y_lluvias_II.csv"):
    files = dbutils.fs.ls(raw_path)
    file_names = [f.name for f in files]

    print("Archivos disponibles en Bagazo:", file_names)

    # 1. Preferimos el archivo esperado si existe.
    if output_file in file_names:
        print(f"✅ CSV encontrado: {output_file}")
        return f"{raw_path}/{output_file}", output_file

    # 2. Si existe otro CSV, usamos el primer CSV disponible.
    csv_candidates = [name for name in file_names if name.lower().endswith(".csv")]
    if csv_candidates:
        selected = csv_candidates[0]
        print(f"✅ CSV encontrado con otro nombre: {selected}")
        return f"{raw_path}/{selected}", selected

    # 3. Si no hay CSV, intentamos convertir desde Excel.
    xlsx_candidates = [name for name in file_names if name.lower().endswith((".xlsx", ".xls"))]
    if not xlsx_candidates:
        raise FileNotFoundError(
            f"No se encontró archivo CSV ni Excel en {raw_path}. "
            "Sube molienda_bagazo_y_lluvias_II.csv o molienda_bagazo_y_lluvias_II.xlsx."
        )

    selected_xlsx = xlsx_candidates[0]
    xlsx_path = f"{raw_path}/{selected_xlsx}"
    csv_path = f"{raw_path}/{output_file}"

    print(f"Convirtiendo Excel a CSV: {xlsx_path} -> {csv_path}")

    import pandas as pd
    pdf = pd.read_excel(xlsx_path, sheet_name=0)
    pdf = pdf.dropna(axis=1, how="all")
    pdf.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"✅ CSV creado: {csv_path}")
    return csv_path, output_file


bagazo_csv_path, bagazo_file_name = prepare_bagazo_csv(BAGAZO_RAW_PATH)
print("Ruta CSV Bagazo:", bagazo_csv_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Ingesta de Bagazo hacia Bronze
# MAGIC
# MAGIC Ahora cargaremos el caso espejo empresarial.
# MAGIC
# MAGIC Este dataset se mantiene en formato ancho en Bronze porque Bronze debe conservar la fuente lo más cercana posible a su estructura original. La transformación a formato largo se realizará en sesiones posteriores.

# COMMAND ----------

# =========================================
# 9. Ingesta Bagazo: CSV -> Delta Bronze
# =========================================

print(f"Cargando Bagazo desde: {bagazo_csv_path}")

try:
    bagazo_df = read_csv_auto(bagazo_csv_path)

    missing, extra = validate_columns(bagazo_df, BAGAZO_EXPECTED_COLUMNS)

    rows = bagazo_df.count()
    cols = len(bagazo_df.columns)

    if rows == BAGAZO_EXPECTED_ROWS:
        row_msg = f"Conteo OK: {rows:,} filas"
        print(f"✅ {row_msg}")
    else:
        row_msg = f"Conteo diferente. Esperado: {BAGAZO_EXPECTED_ROWS:,}; obtenido: {rows:,}"
        print(f"⚠️ {row_msg}")

    write_delta_table(bagazo_df, BAGAZO_BRONZE_SCHEMA, BAGAZO_DATASET)

    estado = "OK" if not missing and rows == BAGAZO_EXPECTED_ROWS else "REVISAR"
    obs = f"{row_msg}. Faltantes: {missing}. Extras: {extra}"

    register_audit(
        audit_rows,
        BAGAZO_DATASET,
        bagazo_file_name,
        f"{CATALOG}.{BAGAZO_BRONZE_SCHEMA}.{BAGAZO_DATASET}",
        rows,
        cols,
        estado,
        obs
    )

    print(f"✅ Tabla Bagazo creada: {CATALOG}.{BAGAZO_BRONZE_SCHEMA}.{BAGAZO_DATASET}")
    display(bagazo_df.limit(10))

except Exception as e:
    print(f"❌ Error cargando Bagazo: {e}")

    register_audit(
        audit_rows,
        BAGAZO_DATASET,
        bagazo_file_name if 'bagazo_file_name' in globals() else None,
        f"{CATALOG}.{BAGAZO_BRONZE_SCHEMA}.{BAGAZO_DATASET}",
        None,
        None,
        "ERROR",
        str(e)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Crear tabla de auditoría
# MAGIC
# MAGIC La auditoría nos permite responder preguntas básicas:
# MAGIC
# MAGIC - ¿Qué archivo se cargó?
# MAGIC - ¿Cuántas filas llegaron?
# MAGIC - ¿A qué tabla se escribió?
# MAGIC - ¿La carga fue correcta o quedó para revisar?
# MAGIC - ¿Qué observaciones dejó el proceso?

# COMMAND ----------

# =========================================
# 10. Crear tabla de auditoría
# =========================================

audit_schema = StructType([
    StructField("fecha_carga", StringType(), True),
    StructField("dataset", StringType(), True),
    StructField("archivo_origen", StringType(), True),
    StructField("tabla_destino", StringType(), True),
    StructField("filas_cargadas", LongType(), True),
    StructField("columnas", IntegerType(), True),
    StructField("estado", StringType(), True),
    StructField("observaciones", StringType(), True),
])

audit_df = spark.createDataFrame(audit_rows, schema=audit_schema)

write_delta_table(audit_df, CONTROL_SCHEMA, AUDIT_TABLE)

display(audit_df.orderBy("dataset"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Consultas de validación con SQL
# MAGIC
# MAGIC Aunque estamos usando PySpark para cargar datos, Databricks permite consultar las tablas con SQL.
# MAGIC
# MAGIC Esta sección conecta el mundo de ingeniería de datos con el mundo de analítica.

# COMMAND ----------

# =========================================
# 11. SQL de validación: tablas creadas
# =========================================

spark.sql(f"""
SHOW TABLES IN `{CATALOG}`.`{LUMI_BRONZE_SCHEMA}`
""").display()

spark.sql(f"""
SHOW TABLES IN `{CATALOG}`.`{BAGAZO_BRONZE_SCHEMA}`
""").display()

spark.sql(f"""
SELECT dataset, estado, filas_cargadas, columnas, archivo_origen
FROM `{CATALOG}`.`{CONTROL_SCHEMA}`.`{AUDIT_TABLE}`
ORDER BY dataset
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Primeras preguntas de negocio sobre Bronze
# MAGIC
# MAGIC Bronze no es todavía la capa analítica final, pero ya nos permite validar que los datos están disponibles.

# COMMAND ----------

# =========================================
# 12. Primeras consultas exploratorias
# =========================================

print("Pedidos por estado de orden:")
spark.sql(f"""
SELECT order_status, COUNT(*) AS total_pedidos
FROM `{CATALOG}`.`{LUMI_BRONZE_SCHEMA}`.`orders`
GROUP BY order_status
ORDER BY total_pedidos DESC
""").display()

print("Clientes por estado:")
spark.sql(f"""
SELECT customer_state, COUNT(*) AS total_clientes
FROM `{CATALOG}`.`{LUMI_BRONZE_SCHEMA}`.`customers`
GROUP BY customer_state
ORDER BY total_clientes DESC
LIMIT 10
""").display()

print("Primeros registros del caso Bagazo:")
spark.sql(f"""
SELECT *
FROM `{CATALOG}`.`{BAGAZO_BRONZE_SCHEMA}`.`{BAGAZO_DATASET}`
LIMIT 10
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Identificación de llaves candidatas y relaciones
# MAGIC
# MAGIC Este paso es conceptual pero muy útil para preparar las siguientes sesiones.
# MAGIC
# MAGIC En Bronze todavía no modelamos formalmente, pero sí podemos empezar a reconocer relaciones:
# MAGIC
# MAGIC | Tabla | Llave candidata | Relación esperada |
# MAGIC |---|---|---|
# MAGIC | `orders` | `order_id` | Se conecta con items, pagos y reviews |
# MAGIC | `customers` | `customer_id` | Se conecta con orders |
# MAGIC | `products` | `product_id` | Se conecta con order_items |
# MAGIC | `sellers` | `seller_id` | Se conecta con order_items |
# MAGIC | `product_category_translation` | `product_category_name` | Traduce categorías de productos |

# COMMAND ----------

# =========================================
# 13. Validaciones simples de llaves candidatas
# =========================================

key_checks = [
    ("orders", "order_id"),
    ("customers", "customer_id"),
    ("products", "product_id"),
    ("sellers", "seller_id"),
]

key_results = []

for table_name, key_col in key_checks:
    df = spark.table(table_fqn(LUMI_BRONZE_SCHEMA, table_name))
    total_rows = df.count()
    distinct_keys = df.select(key_col).distinct().count()
    duplicated_keys = total_rows - distinct_keys

    key_results.append((table_name, key_col, total_rows, distinct_keys, duplicated_keys))

key_df = spark.createDataFrame(
    key_results,
    ["tabla", "llave_candidata", "filas", "llaves_distintas", "duplicados_aparentes"]
)

display(key_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Checkpoint final de la sesión
# MAGIC
# MAGIC Al finalizar esta práctica deberías tener creadas estas tablas:
# MAGIC
# MAGIC ```text
# MAGIC workspace.lumi_bronze.customers
# MAGIC workspace.lumi_bronze.geolocation
# MAGIC workspace.lumi_bronze.order_items
# MAGIC workspace.lumi_bronze.order_payments
# MAGIC workspace.lumi_bronze.order_reviews
# MAGIC workspace.lumi_bronze.orders
# MAGIC workspace.lumi_bronze.products
# MAGIC workspace.lumi_bronze.sellers
# MAGIC workspace.lumi_bronze.product_category_translation
# MAGIC workspace.bagazo_bronze.molienda_bagazo_y_lluvias_II
# MAGIC workspace.control.bronze_ingestion_audit
# MAGIC ```

# COMMAND ----------

# =========================================
# 14. Checkpoint final
# =========================================

final_tables = []

for schema_name in [LUMI_BRONZE_SCHEMA, BAGAZO_BRONZE_SCHEMA, CONTROL_SCHEMA]:
    tables_df = spark.sql(f"SHOW TABLES IN `{CATALOG}`.`{schema_name}`")
    for row in tables_df.collect():
        table_name = row["tableName"]
        full_name = f"{CATALOG}.{schema_name}.{table_name}"
        rows = spark.table(f"`{CATALOG}`.`{schema_name}`.`{table_name}`").count()
        final_tables.append((schema_name, table_name, full_name, rows))

final_df = spark.createDataFrame(final_tables, ["schema", "tabla", "nombre_completo", "filas"])
display(final_df.orderBy("schema", "tabla"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Retos opcionales para cerrar
# MAGIC
# MAGIC Estos retos no bloquean el flujo principal. Son para practicar, discutir o dejar como actividad posterior.
# MAGIC
# MAGIC ### Reto Nivel 1 — Verificación rápida
# MAGIC Identifica cuál tabla Lumi tiene más filas y cuál tiene menos filas.
# MAGIC
# MAGIC ### Reto Nivel 2 — Auditoría ejecutiva
# MAGIC Filtra la tabla de auditoría para mostrar únicamente datasets con estado diferente de `OK`.
# MAGIC
# MAGIC ### Reto consultor — Conversación empresarial
# MAGIC Responde en una celda Markdown:
# MAGIC
# MAGIC 1. ¿Por qué no conviene limpiar demasiado en Bronze?
# MAGIC R/: La capa Bronze es el espejo de tu origen de datos (Raw Data). No se deberia aplicar transformaciones pesadas o limpieza estricta por razones de perdida de historial, reprocesamiento. y rendimiento
# MAGIC
# MAGIC 2. ¿Qué controles mínimos pedirías antes de pasar a Silver?
# MAGIC R/: 
# MAGIC - Validación de Esquema: Verificar que los tipos de datos sean los correctos y manejar la evolución del esquema (Schema Enforcement / Evolution).
# MAGIC - Control de Nulos y Claves: Validar campos críticos (ej. que los IDs o llaves primarias no sean nulos: IS NOT NULL).
# MAGIC - Deduplicación: Eliminar registros duplicados utilizando marcas de tiempo (timestamps) o IDs únicos.
# MAGIC - Formateo Estándar: Fechas en formato ISO (YYYY-MM-DD), strings limpios (sin espacios extras al inicio/final) y nombres de columnas estandarizados (ej. snake_case).
# MAGIC
# MAGIC 3. ¿Qué cambiaría si este flujo se ejecutara en Azure Databricks empresarial conectado a ADLS Gen2?
# MAGIC R/: Almacenamiento y Rutas: En lugar de usar almacenamiento local o DBFS, los datos residen en contenedores de Azure Data Lake Storage Gen2 (ADLS Gen2) usando el protocolo abfss://

# COMMAND ----------

# =========================================
# Reto Nivel 1 y Nivel 2: solución guiada
# =========================================

print("Tabla con más y menos filas:")
final_df.orderBy(F.desc("filas")).display()

print("Auditoría con estados diferentes de OK:")
spark.sql(f"""
SELECT *
FROM `{CATALOG}`.`{CONTROL_SCHEMA}`.`{AUDIT_TABLE}`
WHERE estado <> 'OK'
ORDER BY dataset
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Cierre
# MAGIC
# MAGIC Hoy completamos la parte clave de la sesión 2:
# MAGIC
# MAGIC - Los archivos crudos quedaron en Volumes.
# MAGIC - Las tablas Bronze quedaron creadas como Delta tables.
# MAGIC - Lumi quedó disponible para próximas sesiones de PySpark, SQL, Silver, Gold y dashboards.
# MAGIC - Bagazo quedó disponible como caso espejo para análisis operativo y MLflow más adelante.
# MAGIC - Se generó una tabla de auditoría para controlar cargas.
# MAGIC
# MAGIC **Siguiente sesión:** exploración de datos con PySpark usando preguntas reales de negocio.

# Databricks notebook source
# MAGIC %md
# MAGIC # Sesión 6 — Delta Lake, versionado, historia y confiabilidad
# MAGIC
# MAGIC **Objetivo:** comprender por qué Delta Lake es la base confiable del Lakehouse en Databricks, usando tablas de laboratorio para inspeccionar historia, simular cambios, consultar versiones anteriores, restaurar datos y preparar el camino hacia Gold.
# MAGIC
# MAGIC > Mensaje central: SQL nos ayudó a descubrir insights. Delta Lake nos ayuda a confiar en el camino que produjo esos insights.
# MAGIC
# MAGIC ## Regla de seguridad
# MAGIC
# MAGIC En esta sesión **no modificaremos tablas Silver reales**. Todas las operaciones de escritura, actualización o restauración se harán únicamente sobre tablas de laboratorio en:
# MAGIC
# MAGIC ```text
# MAGIC workspace.delta_lab
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Preparación del entorno
# MAGIC
# MAGIC Validamos el catálogo `workspace` y los schemas disponibles. Si algún schema no aparece, revisa que las sesiones anteriores se hayan ejecutado correctamente.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Nota de corrección aplicada
# MAGIC
# MAGIC En el laboratorio de Bagazo se evita el patrón `WHERE (fecha, ingenio) IN (...)` dentro de `UPDATE`, porque Delta Lake no soporta predicados `IN` multicolumna en condiciones de actualización. La lógica queda igual, pero se expresa con condiciones separadas por columna y joins seguros para las consultas de validación.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG workspace;
# MAGIC SHOW SCHEMAS;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Retomar la salud de Silver
# MAGIC
# MAGIC La tabla de control creada en la Sesión 4 sigue siendo el punto de entrada para hablar de confiabilidad. Recuerda: Delta Lake permite auditar cambios, pero no reemplaza reglas de calidad.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   dataset,
# MAGIC   tabla,
# MAGIC   capa,
# MAGIC   filas,
# MAGIC   columnas,
# MAGIC   nulos_criticos,
# MAGIC   duplicados_clave,
# MAGIC   reglas_fallidas,
# MAGIC   estado_calidad,
# MAGIC   fecha_validacion,
# MAGIC   observaciones
# MAGIC FROM workspace.control.quality_summary_sesion_04
# MAGIC ORDER BY estado_calidad DESC, dataset, tabla;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   estado_calidad,
# MAGIC   COUNT(*) AS tablas
# MAGIC FROM workspace.control.quality_summary_sesion_04
# MAGIC GROUP BY estado_calidad
# MAGIC ORDER BY estado_calidad;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reflexión rápida
# MAGIC
# MAGIC `reviews_clean` quedó en estado **REVISAR** por duplicados de clave. Esto no significa que Delta Lake haya fallado; significa que la auditoría técnica y la calidad de negocio son controles complementarios.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tablas disponibles para la sesión
# MAGIC
# MAGIC Estas tablas se consultan como fuente, pero no se modifican directamente.

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN workspace.lumi_silver;
# MAGIC SHOW TABLES IN workspace.bagazo_silver;
# MAGIC SHOW TABLES IN workspace.control;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Inspeccionar metadata de tablas Silver
# MAGIC
# MAGIC `DESCRIBE DETAIL` permite revisar información técnica de una tabla Delta: formato, ubicación, tamaño, propiedades y otra metadata disponible.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL workspace.lumi_silver.orders_clean;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL workspace.bagazo_silver.operacion_ingenios_clean;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Revisar historia de tablas Silver
# MAGIC
# MAGIC Podemos consultar la historia de una tabla Silver. **Solo la revisamos; no la modificamos.**

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.lumi_silver.orders_clean;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.bagazo_silver.operacion_ingenios_clean;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Crear el schema de laboratorio
# MAGIC
# MAGIC A partir de aquí trabajaremos en una caja de arena segura.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.delta_lab;
# MAGIC SHOW TABLES IN workspace.delta_lab;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Crear tablas de laboratorio
# MAGIC
# MAGIC Para garantizar reproducibilidad, eliminamos y recreamos únicamente las tablas lab. Esto permite que `VERSION AS OF 0` sea la versión inicial de esta ejecución.
# MAGIC
# MAGIC > Seguridad: estas operaciones solo afectan `workspace.delta_lab`.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS workspace.delta_lab.orders_reliability_lab;
# MAGIC
# MAGIC CREATE TABLE workspace.delta_lab.orders_reliability_lab AS
# MAGIC SELECT *
# MAGIC FROM workspace.lumi_silver.orders_clean
# MAGIC WHERE order_status = 'delivered'
# MAGIC LIMIT 5000;

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS workspace.delta_lab.bagazo_reliability_lab;
# MAGIC
# MAGIC CREATE TABLE workspace.delta_lab.bagazo_reliability_lab AS
# MAGIC SELECT *
# MAGIC FROM workspace.bagazo_silver.operacion_ingenios_clean;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validar tablas lab
# MAGIC
# MAGIC Confirmamos que las tablas lab existen y tienen datos.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'orders_reliability_lab' AS tabla, COUNT(*) AS registros
# MAGIC FROM workspace.delta_lab.orders_reliability_lab
# MAGIC UNION ALL
# MAGIC SELECT 'bagazo_reliability_lab' AS tabla, COUNT(*) AS registros
# MAGIC FROM workspace.delta_lab.bagazo_reliability_lab;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.delta_lab.orders_reliability_lab;
# MAGIC DESCRIBE HISTORY workspace.delta_lab.bagazo_reliability_lab;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Laboratorio Lumi — seleccionar una orden candidata
# MAGIC
# MAGIC Seleccionaremos una orden de laboratorio para simular un cambio controlado. La vista temporal evita copiar manualmente un `order_id`.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW v_order_lab_candidate AS
# MAGIC SELECT order_id
# MAGIC FROM workspace.delta_lab.orders_reliability_lab
# MAGIC ORDER BY order_id
# MAGIC LIMIT 1;
# MAGIC
# MAGIC SELECT * FROM v_order_lab_candidate;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT order_id, order_status, delay_days, is_late
# MAGIC FROM workspace.delta_lab.orders_reliability_lab
# MAGIC WHERE order_id = (SELECT order_id FROM v_order_lab_candidate);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Laboratorio Lumi — simular cambio controlado
# MAGIC
# MAGIC Vamos a modificar `order_status` en una sola orden de laboratorio. Esta operación debe generar una nueva versión.
# MAGIC
# MAGIC > Antes de ejecutar: verifica que la tabla pertenece a `workspace.delta_lab`.

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE workspace.delta_lab.orders_reliability_lab
# MAGIC SET order_status = 'canceled'
# MAGIC WHERE order_id = (SELECT order_id FROM v_order_lab_candidate);

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.delta_lab.orders_reliability_lab;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Time Travel — consultar versión anterior
# MAGIC
# MAGIC Ahora consultaremos la versión inicial de la tabla lab. Como recreamos la tabla en este notebook, normalmente la versión inicial es `0`.
# MAGIC
# MAGIC **TODO 1:** si tu historial muestra otra versión inicial por una ejecución anterior, cambia `VERSION AS OF 0` por la versión correcta.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT order_id, order_status, delay_days, is_late
# MAGIC FROM workspace.delta_lab.orders_reliability_lab VERSION AS OF 0
# MAGIC WHERE order_id = (SELECT order_id FROM v_order_lab_candidate);

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT order_id, order_status, delay_days, is_late
# MAGIC FROM workspace.delta_lab.orders_reliability_lab
# MAGIC WHERE order_id = (SELECT order_id FROM v_order_lab_candidate);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Comparar versión actual vs versión anterior
# MAGIC
# MAGIC Esta consulta muestra la diferencia entre el estado inicial y el estado actual de la orden.

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH version_inicial AS (
# MAGIC   SELECT
# MAGIC     order_id,
# MAGIC     order_status AS order_status_version_0,
# MAGIC     delay_days AS delay_days_version_0
# MAGIC   FROM workspace.delta_lab.orders_reliability_lab VERSION AS OF 0
# MAGIC   WHERE order_id = (SELECT order_id FROM v_order_lab_candidate)
# MAGIC ), version_actual AS (
# MAGIC   SELECT
# MAGIC     order_id,
# MAGIC     order_status AS order_status_actual,
# MAGIC     delay_days AS delay_days_actual
# MAGIC   FROM workspace.delta_lab.orders_reliability_lab
# MAGIC   WHERE order_id = (SELECT order_id FROM v_order_lab_candidate)
# MAGIC )
# MAGIC SELECT
# MAGIC   a.order_id,
# MAGIC   i.order_status_version_0,
# MAGIC   a.order_status_actual,
# MAGIC   i.delay_days_version_0,
# MAGIC   a.delay_days_actual
# MAGIC FROM version_actual a
# MAGIC JOIN version_inicial i
# MAGIC   ON a.order_id = i.order_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Restore controlado de Lumi lab
# MAGIC
# MAGIC Restauramos la tabla lab a la versión inicial. Recuerda: `RESTORE` también queda registrado en la historia.

# COMMAND ----------

# MAGIC %sql
# MAGIC RESTORE TABLE workspace.delta_lab.orders_reliability_lab TO VERSION AS OF 0;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.delta_lab.orders_reliability_lab;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT order_id, order_status, delay_days, is_late
# MAGIC FROM workspace.delta_lab.orders_reliability_lab
# MAGIC WHERE order_id = (SELECT order_id FROM v_order_lab_candidate);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Laboratorio Bagazo — simular una lectura inválida
# MAGIC
# MAGIC Ahora trabajaremos con el caso espejo empresarial. Simularemos una lectura de lluvia negativa para mostrar que Delta Lake audita el cambio, pero las reglas de calidad son las que detectan el problema.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW v_bagazo_lab_candidate AS
# MAGIC SELECT fecha, ingenio
# MAGIC FROM workspace.delta_lab.bagazo_reliability_lab
# MAGIC WHERE ingenio = 'Providencia'
# MAGIC ORDER BY fecha
# MAGIC LIMIT 1;
# MAGIC
# MAGIC SELECT * FROM v_bagazo_lab_candidate;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Corrección: evitamos predicados IN multicolumna en operaciones Delta.
# MAGIC SELECT t.fecha, t.ingenio, t.lluvia_mm, t.cana_molida_ton, t.bagazo_entregado_ton
# MAGIC FROM workspace.delta_lab.bagazo_reliability_lab AS t
# MAGIC INNER JOIN v_bagazo_lab_candidate AS c
# MAGIC   ON t.fecha = c.fecha
# MAGIC  AND t.ingenio = c.ingenio;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Ejecutar el cambio inválido en Bagazo lab
# MAGIC
# MAGIC > Seguridad: la tabla sigue siendo `workspace.delta_lab.bagazo_reliability_lab`.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Seguridad: esta operación modifica SOLO la tabla de laboratorio.
# MAGIC -- Corrección aplicada: Delta no soporta IN multicolumna dentro de UPDATE.
# MAGIC UPDATE workspace.delta_lab.bagazo_reliability_lab
# MAGIC SET lluvia_mm = -10
# MAGIC WHERE fecha = (SELECT fecha FROM v_bagazo_lab_candidate)
# MAGIC   AND ingenio = (SELECT ingenio FROM v_bagazo_lab_candidate);

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT fecha, ingenio, lluvia_mm, cana_molida_ton, bagazo_entregado_ton
# MAGIC FROM workspace.delta_lab.bagazo_reliability_lab
# MAGIC WHERE lluvia_mm < 0;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.delta_lab.bagazo_reliability_lab;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Time Travel y restore en Bagazo
# MAGIC
# MAGIC **TODO 2:** revisa el historial y confirma cuál es la versión inicial antes de ejecutar el restore. En una ejecución limpia será `0`.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Time Travel seguro sobre la versión inicial de la tabla lab.
# MAGIC WITH bagazo_version_inicial AS (
# MAGIC   SELECT fecha, ingenio, lluvia_mm, cana_molida_ton, bagazo_entregado_ton
# MAGIC   FROM workspace.delta_lab.bagazo_reliability_lab VERSION AS OF 0
# MAGIC )
# MAGIC SELECT t.fecha, t.ingenio, t.lluvia_mm, t.cana_molida_ton, t.bagazo_entregado_ton
# MAGIC FROM bagazo_version_inicial AS t
# MAGIC INNER JOIN v_bagazo_lab_candidate AS c
# MAGIC   ON t.fecha = c.fecha
# MAGIC  AND t.ingenio = c.ingenio;

# COMMAND ----------

# MAGIC %sql
# MAGIC RESTORE TABLE workspace.delta_lab.bagazo_reliability_lab TO VERSION AS OF 0;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Corrección: evitamos predicados IN multicolumna en operaciones Delta.
# MAGIC SELECT t.fecha, t.ingenio, t.lluvia_mm, t.cana_molida_ton, t.bagazo_entregado_ton
# MAGIC FROM workspace.delta_lab.bagazo_reliability_lab AS t
# MAGIC INNER JOIN v_bagazo_lab_candidate AS c
# MAGIC   ON t.fecha = c.fecha
# MAGIC  AND t.ingenio = c.ingenio;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Crear resumen de confiabilidad de la sesión
# MAGIC
# MAGIC Esta tabla opcional resume la revisión de confiabilidad realizada antes de construir Gold en la Sesión 7.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.control.delta_reliability_summary_sesion_06 AS
# MAGIC SELECT
# MAGIC   'workspace.delta_lab.orders_reliability_lab' AS tabla,
# MAGIC   'Lumi: laboratorio de versionado sobre pedidos' AS proposito,
# MAGIC   'DESCRIBE HISTORY + VERSION AS OF + RESTORE' AS evidencia_usada,
# MAGIC   'REVISADA' AS estado_revision,
# MAGIC   'No modifica Silver real. Lista como práctica de auditoría previa a Gold.' AS observaciones,
# MAGIC   current_timestamp() AS fecha_revision
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   'workspace.delta_lab.bagazo_reliability_lab' AS tabla,
# MAGIC   'Bagazo: laboratorio de trazabilidad operativa' AS proposito,
# MAGIC   'DESCRIBE HISTORY + detección de rango inválido + RESTORE' AS evidencia_usada,
# MAGIC   'REVISADA' AS estado_revision,
# MAGIC   'No modifica Silver real. Útil para explicar calidad + confiabilidad.' AS observaciones,
# MAGIC   current_timestamp() AS fecha_revision;
# MAGIC
# MAGIC SELECT * FROM workspace.control.delta_reliability_summary_sesion_06;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Checklist de confiabilidad pre-Gold
# MAGIC
# MAGIC Completa este checklist antes de pasar a la Sesión 7.
# MAGIC
# MAGIC **TODO 3:** completa la conclusión ejecutiva.
# MAGIC
# MAGIC ```text
# MAGIC Tabla:
# MAGIC Propósito analítico:
# MAGIC Última versión revisada:
# MAGIC Operaciones recientes:
# MAGIC Riesgos detectados:
# MAGIC ¿Puede alimentar Gold?: Sí / No / Con observaciones
# MAGIC Evidencia SQL usada:
# MAGIC Recomendación:
# MAGIC ```
# MAGIC
# MAGIC ### Conclusión ejecutiva
# MAGIC
# MAGIC Escribe aquí tu conclusión:
# MAGIC
# MAGIC > ...

# COMMAND ----------

# MAGIC %md
# MAGIC # Retos finales
# MAGIC
# MAGIC Los retos no interrumpen el flujo principal. Úsalos como práctica de cierre o trabajo autónomo.
# MAGIC
# MAGIC ## Reto Nivel 1 — Historia de tabla
# MAGIC
# MAGIC Elige una tabla lab, ejecuta `DESCRIBE HISTORY` e identifica:
# MAGIC
# MAGIC - versión inicial;
# MAGIC - última versión;
# MAGIC - última operación;
# MAGIC - evidencia más importante.
# MAGIC
# MAGIC ## Reto Nivel 2 — Time Travel y comparación
# MAGIC
# MAGIC Consulta una versión anterior con `VERSION AS OF`, compárala contra la versión actual y explica qué cambió.
# MAGIC
# MAGIC ## Reto consultor — Checklist pre-Gold
# MAGIC
# MAGIC Entrega el checklist de confiabilidad para una tabla que podría alimentar Gold en la Sesión 7.

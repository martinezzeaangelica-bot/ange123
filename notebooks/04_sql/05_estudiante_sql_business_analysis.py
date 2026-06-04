# Databricks notebook source
# MAGIC %md
# MAGIC # Sesión 5 — SQL de negocio y análisis operativo
# MAGIC
# MAGIC **Mensaje central:** Silver nos da datos confiables; SQL nos permite convertirlos en evidencia, insights y decisiones.
# MAGIC
# MAGIC En esta sesión trabajaremos sobre las tablas Silver creadas en la Sesión 4. El objetivo no es crear Gold todavía, sino dejar consultas candidatas y entender bien granularidad, joins, agregaciones y riesgo de doble conteo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Contexto técnico esperado
# MAGIC
# MAGIC Catálogo: `workspace`
# MAGIC
# MAGIC Tablas principales:
# MAGIC
# MAGIC - `workspace.lumi_silver.orders_clean`
# MAGIC - `workspace.lumi_silver.order_items_clean`
# MAGIC - `workspace.lumi_silver.payments_clean`
# MAGIC - `workspace.lumi_silver.reviews_clean`
# MAGIC - `workspace.lumi_silver.products_clean`
# MAGIC - `workspace.lumi_silver.sellers_clean`
# MAGIC - `workspace.bagazo_silver.operacion_ingenios_clean`
# MAGIC - `workspace.control.quality_summary_sesion_04`

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG workspace;
# MAGIC SHOW SCHEMAS;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Salud de Silver
# MAGIC
# MAGIC Abrimos la clase con gobierno de datos. Antes de analizar, revisamos qué tan confiables son las tablas que vamos a consultar.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM workspace.control.quality_summary_sesion_04
# MAGIC ORDER BY dataset, tabla;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   estado_calidad,
# MAGIC   COUNT(*) AS tablas
# MAGIC FROM workspace.control.quality_summary_sesion_04
# MAGIC GROUP BY estado_calidad
# MAGIC ORDER BY tablas DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   dataset, tabla, filas, columnas, duplicados_clave, reglas_fallidas, estado_calidad, observaciones
# MAGIC FROM workspace.control.quality_summary_sesion_04
# MAGIC WHERE estado_calidad <> 'OK'
# MAGIC ORDER BY dataset, tabla;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reflexión guiada
# MAGIC
# MAGIC `reviews_clean` quedó en estado **REVISAR** por duplicados de clave. Esto no bloquea el análisis, pero sí cambia la forma correcta de hacer joins. Cuando crucemos reviews con pedidos o ítems, primero agregaremos reviews por `order_id`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Granularidad: la regla que evita métricas infladas
# MAGIC
# MAGIC - `orders_clean`: una fila por pedido.
# MAGIC - `payments_clean`: consolidada por `order_id`.
# MAGIC - `order_items_clean`: una fila por ítem de pedido.
# MAGIC - `reviews_clean`: puede tener más de una fila por pedido; requiere agregación previa.
# MAGIC - `operacion_ingenios_clean`: una fila por fecha e ingenio.
# MAGIC
# MAGIC **Regla de oro:** si dos tablas tienen granularidad diferente, agrega primero y une después.

# COMMAND ----------

# MAGIC %md
# MAGIC # Bloque A — Lumi Commerce Lakehouse

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ventas por mes
# MAGIC
# MAGIC Usamos `order_items_clean` porque la venta está a nivel de ítem. Unimos con `orders_clean` para filtrar pedidos entregados y agrupar por mes.
# MAGIC
# MAGIC **TODO pedagógico 1:** cambia temporalmente el filtro `order_status = 'delivered'` por otro estado y compara el resultado.

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH ventas_item AS (
# MAGIC   SELECT
# MAGIC     o.year_month,
# MAGIC     oi.order_id,
# MAGIC     oi.order_item_id,
# MAGIC     oi.total_item_value
# MAGIC   FROM workspace.lumi_silver.order_items_clean oi
# MAGIC   INNER JOIN workspace.lumi_silver.orders_clean o
# MAGIC     ON oi.order_id = o.order_id
# MAGIC   WHERE o.order_status = 'delivered'
# MAGIC )
# MAGIC SELECT
# MAGIC   year_month,
# MAGIC   COUNT(DISTINCT order_id) AS pedidos_entregados,
# MAGIC   COUNT(*) AS items_vendidos,
# MAGIC   ROUND(SUM(total_item_value), 2) AS venta_total_items,
# MAGIC   ROUND(AVG(total_item_value), 2) AS valor_promedio_item
# MAGIC FROM ventas_item
# MAGIC GROUP BY year_month
# MAGIC ORDER BY year_month;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Ticket promedio por pedido
# MAGIC
# MAGIC Aquí usamos `payments_clean`, que ya quedó consolidada por pedido en Silver. Por eso podemos calcular ticket promedio sin unir contra ítems.
# MAGIC
# MAGIC **TODO pedagógico 2:** identifica por qué usamos `COUNT(DISTINCT o.order_id)` y no simplemente `COUNT(*)` en análisis con joins.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   o.year_month,
# MAGIC   COUNT(DISTINCT o.order_id) AS pedidos_entregados,
# MAGIC   ROUND(SUM(COALESCE(p.total_payment_value, 0)), 2) AS valor_pagado_total,
# MAGIC   ROUND(AVG(COALESCE(p.total_payment_value, 0)), 2) AS ticket_promedio_pedido
# MAGIC FROM workspace.lumi_silver.orders_clean o
# MAGIC LEFT JOIN workspace.lumi_silver.payments_clean p
# MAGIC   ON o.order_id = p.order_id
# MAGIC WHERE o.order_status = 'delivered'
# MAGIC GROUP BY o.year_month
# MAGIC ORDER BY o.year_month;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Pedidos por estado

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   order_status,
# MAGIC   COUNT(*) AS pedidos,
# MAGIC   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS porcentaje
# MAGIC FROM workspace.lumi_silver.orders_clean
# MAGIC GROUP BY order_status
# MAGIC ORDER BY pedidos DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Ventas por categoría
# MAGIC
# MAGIC Cuidamos la granularidad: ventas por categoría se calcula desde ítems, no desde pagos.

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH ventas_categoria AS (
# MAGIC   SELECT
# MAGIC     COALESCE(p.product_category_clean, 'sin_categoria') AS categoria,
# MAGIC     oi.order_id,
# MAGIC     oi.order_item_id,
# MAGIC     oi.total_item_value
# MAGIC   FROM workspace.lumi_silver.order_items_clean oi
# MAGIC   INNER JOIN workspace.lumi_silver.orders_clean o
# MAGIC     ON oi.order_id = o.order_id
# MAGIC   LEFT JOIN workspace.lumi_silver.products_clean p
# MAGIC     ON oi.product_id = p.product_id
# MAGIC   WHERE o.order_status = 'delivered'
# MAGIC )
# MAGIC SELECT
# MAGIC   categoria,
# MAGIC   COUNT(DISTINCT order_id) AS pedidos,
# MAGIC   COUNT(*) AS items,
# MAGIC   ROUND(SUM(total_item_value), 2) AS venta_total,
# MAGIC   ROUND(AVG(total_item_value), 2) AS valor_promedio_item
# MAGIC FROM ventas_categoria
# MAGIC GROUP BY categoria
# MAGIC ORDER BY venta_total DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Métodos de pago

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   main_payment_type,
# MAGIC   COUNT(*) AS pedidos,
# MAGIC   ROUND(SUM(total_payment_value), 2) AS valor_total_pagado,
# MAGIC   ROUND(AVG(total_payment_value), 2) AS ticket_promedio
# MAGIC FROM workspace.lumi_silver.payments_clean
# MAGIC GROUP BY main_payment_type
# MAGIC ORDER BY valor_total_pagado DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Reviews por categoría
# MAGIC
# MAGIC Antes de cruzar reviews con categorías, agregamos reviews por pedido.
# MAGIC
# MAGIC **TODO pedagógico 3:** en la CTE `reviews_by_order`, identifica la métrica que representa satisfacción promedio por pedido.

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH reviews_by_order AS (
# MAGIC   SELECT
# MAGIC     order_id,
# MAGIC     ROUND(AVG(review_score), 2) AS review_promedio_pedido,
# MAGIC     MAX(CASE WHEN is_low_review THEN 1 ELSE 0 END) AS tiene_review_bajo
# MAGIC   FROM workspace.lumi_silver.reviews_clean
# MAGIC   GROUP BY order_id
# MAGIC ),
# MAGIC category_orders AS (
# MAGIC   SELECT DISTINCT
# MAGIC     oi.order_id,
# MAGIC     COALESCE(p.product_category_clean, 'sin_categoria') AS categoria
# MAGIC   FROM workspace.lumi_silver.order_items_clean oi
# MAGIC   LEFT JOIN workspace.lumi_silver.products_clean p
# MAGIC     ON oi.product_id = p.product_id
# MAGIC )
# MAGIC SELECT
# MAGIC   co.categoria,
# MAGIC   COUNT(DISTINCT co.order_id) AS pedidos_con_categoria,
# MAGIC   ROUND(AVG(r.review_promedio_pedido), 2) AS review_promedio,
# MAGIC   SUM(CASE WHEN r.tiene_review_bajo = 1 THEN 1 ELSE 0 END) AS pedidos_con_review_bajo
# MAGIC FROM category_orders co
# MAGIC LEFT JOIN reviews_by_order r
# MAGIC   ON co.order_id = r.order_id
# MAGIC GROUP BY co.categoria
# MAGIC HAVING COUNT(DISTINCT co.order_id) >= 50
# MAGIC ORDER BY review_promedio ASC, pedidos_con_categoria DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Entregas tardías

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   year_month,
# MAGIC   COUNT(*) AS pedidos_entregados,
# MAGIC   SUM(CASE WHEN is_late THEN 1 ELSE 0 END) AS pedidos_tarde,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN is_late THEN 1 ELSE 0 END) / COUNT(*), 2) AS tasa_entrega_tarde,
# MAGIC   ROUND(AVG(delay_days), 2) AS demora_promedio_dias
# MAGIC FROM workspace.lumi_silver.orders_clean
# MAGIC WHERE order_status = 'delivered'
# MAGIC GROUP BY year_month
# MAGIC ORDER BY year_month;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Relación entre demora y review

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH reviews_by_order AS (
# MAGIC   SELECT
# MAGIC     order_id,
# MAGIC     ROUND(AVG(review_score), 2) AS review_promedio_pedido
# MAGIC   FROM workspace.lumi_silver.reviews_clean
# MAGIC   GROUP BY order_id
# MAGIC )
# MAGIC SELECT
# MAGIC   CASE
# MAGIC     WHEN o.delay_days <= 0 THEN 'A tiempo o antes'
# MAGIC     WHEN o.delay_days BETWEEN 1 AND 3 THEN '1 a 3 días tarde'
# MAGIC     WHEN o.delay_days BETWEEN 4 AND 7 THEN '4 a 7 días tarde'
# MAGIC     ELSE 'Más de 7 días tarde'
# MAGIC   END AS tramo_demora,
# MAGIC   COUNT(DISTINCT o.order_id) AS pedidos,
# MAGIC   ROUND(AVG(o.delay_days), 2) AS demora_promedio,
# MAGIC   ROUND(AVG(r.review_promedio_pedido), 2) AS review_promedio
# MAGIC FROM workspace.lumi_silver.orders_clean o
# MAGIC LEFT JOIN reviews_by_order r
# MAGIC   ON o.order_id = r.order_id
# MAGIC WHERE o.order_status = 'delivered'
# MAGIC GROUP BY tramo_demora
# MAGIC ORDER BY demora_promedio;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Ranking de vendedores

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   s.seller_id,
# MAGIC   s.seller_state,
# MAGIC   COUNT(DISTINCT oi.order_id) AS pedidos,
# MAGIC   COUNT(*) AS items,
# MAGIC   ROUND(SUM(oi.total_item_value), 2) AS venta_total,
# MAGIC   ROUND(AVG(oi.total_item_value), 2) AS valor_promedio_item
# MAGIC FROM workspace.lumi_silver.order_items_clean oi
# MAGIC INNER JOIN workspace.lumi_silver.orders_clean o
# MAGIC   ON oi.order_id = o.order_id
# MAGIC LEFT JOIN workspace.lumi_silver.sellers_clean s
# MAGIC   ON oi.seller_id = s.seller_id
# MAGIC WHERE o.order_status = 'delivered'
# MAGIC GROUP BY s.seller_id, s.seller_state
# MAGIC ORDER BY venta_total DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Categorías con alta venta y baja satisfacción

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH ventas_categoria AS (
# MAGIC   SELECT
# MAGIC     COALESCE(p.product_category_clean, 'sin_categoria') AS categoria,
# MAGIC     COUNT(DISTINCT oi.order_id) AS pedidos,
# MAGIC     ROUND(SUM(oi.total_item_value), 2) AS venta_total
# MAGIC   FROM workspace.lumi_silver.order_items_clean oi
# MAGIC   INNER JOIN workspace.lumi_silver.orders_clean o
# MAGIC     ON oi.order_id = o.order_id
# MAGIC   LEFT JOIN workspace.lumi_silver.products_clean p
# MAGIC     ON oi.product_id = p.product_id
# MAGIC   WHERE o.order_status = 'delivered'
# MAGIC   GROUP BY COALESCE(p.product_category_clean, 'sin_categoria')
# MAGIC ),
# MAGIC reviews_by_order AS (
# MAGIC   SELECT order_id, AVG(review_score) AS review_promedio_pedido
# MAGIC   FROM workspace.lumi_silver.reviews_clean
# MAGIC   GROUP BY order_id
# MAGIC ),
# MAGIC category_orders AS (
# MAGIC   SELECT DISTINCT
# MAGIC     oi.order_id,
# MAGIC     COALESCE(p.product_category_clean, 'sin_categoria') AS categoria
# MAGIC   FROM workspace.lumi_silver.order_items_clean oi
# MAGIC   LEFT JOIN workspace.lumi_silver.products_clean p
# MAGIC     ON oi.product_id = p.product_id
# MAGIC ),
# MAGIC reviews_categoria AS (
# MAGIC   SELECT
# MAGIC     co.categoria,
# MAGIC     ROUND(AVG(r.review_promedio_pedido), 2) AS review_promedio
# MAGIC   FROM category_orders co
# MAGIC   LEFT JOIN reviews_by_order r
# MAGIC     ON co.order_id = r.order_id
# MAGIC   GROUP BY co.categoria
# MAGIC )
# MAGIC SELECT
# MAGIC   v.categoria,
# MAGIC   v.pedidos,
# MAGIC   v.venta_total,
# MAGIC   r.review_promedio,
# MAGIC   CASE
# MAGIC     WHEN v.venta_total >= 100000 AND r.review_promedio < 4.0 THEN 'Alta prioridad'
# MAGIC     WHEN v.venta_total >= 50000 AND r.review_promedio < 4.0 THEN 'Revisar'
# MAGIC     ELSE 'Monitorear'
# MAGIC   END AS prioridad_analitica
# MAGIC FROM ventas_categoria v
# MAGIC LEFT JOIN reviews_categoria r
# MAGIC   ON v.categoria = r.categoria
# MAGIC WHERE v.pedidos >= 100
# MAGIC ORDER BY prioridad_analitica, v.venta_total DESC, r.review_promedio ASC
# MAGIC LIMIT 25;

# COMMAND ----------

# MAGIC %md
# MAGIC # Bloque B — Lluvia, caña y bagazo

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Lectura operativa del caso Bagazo
# MAGIC
# MAGIC La tabla quedó en granularidad limpia: una fila por `fecha` e `ingenio`. Aquí los ceros operativos no se tratan automáticamente como errores: se interpretan con comentarios y banderas.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM workspace.bagazo_silver.operacion_ingenios_clean
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Lluvia y bagazo por mes

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   year_month,
# MAGIC   ROUND(AVG(lluvia_mm), 2) AS lluvia_promedio_mm,
# MAGIC   ROUND(AVG(bagazo_entregado_ton), 2) AS bagazo_promedio_ton,
# MAGIC   ROUND(AVG(cana_molida_ton), 2) AS cana_promedio_ton,
# MAGIC   COUNT(*) AS registros
# MAGIC FROM workspace.bagazo_silver.operacion_ingenios_clean
# MAGIC GROUP BY year_month
# MAGIC ORDER BY year_month;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Caña y bagazo por ingenio

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ingenio,
# MAGIC   ROUND(SUM(cana_molida_ton), 2) AS cana_total_ton,
# MAGIC   ROUND(SUM(bagazo_entregado_ton), 2) AS bagazo_total_ton,
# MAGIC   ROUND(AVG(bagazo_entregado_ton), 2) AS bagazo_promedio_ton,
# MAGIC   SUM(CASE WHEN riesgo_bajo_bagazo THEN 1 ELSE 0 END) AS dias_riesgo_bajo
# MAGIC FROM workspace.bagazo_silver.operacion_ingenios_clean
# MAGIC GROUP BY ingenio
# MAGIC ORDER BY bagazo_total_ton DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Días con lluvia alta y bajo bagazo

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   fecha, ingenio, lluvia_mm, cana_molida_ton, bagazo_entregado_ton,
# MAGIC   lluvia_alta, riesgo_bajo_bagazo, tiene_comentario_operativo, comentario
# MAGIC FROM workspace.bagazo_silver.operacion_ingenios_clean
# MAGIC WHERE lluvia_alta = true OR riesgo_bajo_bagazo = true
# MAGIC ORDER BY fecha, ingenio
# MAGIC LIMIT 80;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Promedio de bagazo en días secos vs lluviosos

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ingenio,
# MAGIC   CASE WHEN lluvia_alta THEN 'Día con lluvia alta' ELSE 'Día sin lluvia alta' END AS tipo_dia,
# MAGIC   COUNT(*) AS dias,
# MAGIC   ROUND(AVG(lluvia_mm), 2) AS lluvia_promedio_mm,
# MAGIC   ROUND(AVG(cana_molida_ton), 2) AS cana_promedio_ton,
# MAGIC   ROUND(AVG(bagazo_entregado_ton), 2) AS bagazo_promedio_ton
# MAGIC FROM workspace.bagazo_silver.operacion_ingenios_clean
# MAGIC GROUP BY ingenio, CASE WHEN lluvia_alta THEN 'Día con lluvia alta' ELSE 'Día sin lluvia alta' END
# MAGIC ORDER BY ingenio, tipo_dia;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Correlación simple por ingenio

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ingenio,
# MAGIC   COUNT(*) AS observaciones,
# MAGIC   ROUND(CORR(lluvia_mm, bagazo_entregado_ton), 4) AS corr_lluvia_bagazo,
# MAGIC   ROUND(CORR(cana_molida_ton, bagazo_entregado_ton), 4) AS corr_cana_bagazo,
# MAGIC   ROUND(CORR(lluvia_mm, cana_molida_ton), 4) AS corr_lluvia_cana
# MAGIC FROM workspace.bagazo_silver.operacion_ingenios_clean
# MAGIC GROUP BY ingenio
# MAGIC ORDER BY ingenio;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Días críticos explicados por comentarios operativos

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   fecha, ingenio, lluvia_mm, cana_molida_ton, bagazo_entregado_ton,
# MAGIC   es_mantenimiento, es_falta_cana_por_lluvia, es_paro, es_sin_recepcion_bagazo,
# MAGIC   comentario
# MAGIC FROM workspace.bagazo_silver.operacion_ingenios_clean
# MAGIC WHERE riesgo_bajo_bagazo = true
# MAGIC   AND tiene_comentario_operativo = true
# MAGIC ORDER BY fecha, ingenio
# MAGIC LIMIT 50;

# COMMAND ----------

# MAGIC %md
# MAGIC # Cierre — De SQL a insight ejecutivo
# MAGIC
# MAGIC **TODO pedagógico 4:** escribe un insight ejecutivo usando este formato:
# MAGIC
# MAGIC ```text
# MAGIC Insight:
# MAGIC Evidencia:
# MAGIC Impacto operativo o de negocio:
# MAGIC Recomendación:
# MAGIC Consulta SQL usada:
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # Retos finales
# MAGIC
# MAGIC Los retos no bloquean el flujo principal. Úsalos para práctica individual o trabajo por grupos.
# MAGIC
# MAGIC ## Reto nivel 1
# MAGIC Identifica las 10 categorías con mayor review promedio o los 10 estados con más pedidos entregados.
# MAGIC
# MAGIC ## Reto nivel 2
# MAGIC Construye una CTE que combine ventas, demora y review promedio por categoría sin duplicar pagos ni pedidos.
# MAGIC
# MAGIC ## Reto consultor
# MAGIC Entrega 3 insights en formato ejecutivo, cada uno respaldado por una consulta SQL.

# COMMAND ----------


## Reto 1
%sql
SELECT
  c.customer_state AS estado,
  COUNT(DISTINCT o.order_id) AS total_pedidos_entregados
FROM workspace.lumi_silver.orders_clean o
INNER JOIN workspace.lumi_silver.customers_clean c
  ON o.customer_id = c.customer_id
WHERE o.order_status = 'delivered'
GROUP BY c.customer_state
ORDER BY total_pedidos_entregados DESC
LIMIT 10;

# COMMAND ----------

##  reto 2

%sql
--RETO NIVEL 2: CTE Consolidada de Ventas, Logística y Satisfacción por Categoría
WITH v_items AS (
  -- Lógica de ventas e ítems a nivel de detalle físico
  SELECT
    COALESCE(p.product_category_clean, 'sin_categoria') AS categoria,
    oi.order_id,
    oi.total_item_value
  FROM workspace.lumi_silver.order_items_clean oi
  LEFT JOIN workspace.lumi_silver.products_clean p
    ON oi.product_id = p.product_id
),
v_orders AS (
  -- Lógica logística y de tiempos a nivel de pedido único
  SELECT
    order_id,
    delay_days,
    CASE WHEN is_late THEN 1 ELSE 0 END AS es_tardio
  FROM workspace.lumi_silver.orders_clean
  WHERE order_status = 'delivered'
),
v_reviews AS (
  -- Lógica de satisfacción agrupada previamente para evitar duplicar el Join
  SELECT
    order_id,
    AVG(review_score) AS review_promedio_pedido
  FROM workspace.lumi_silver.reviews_clean
  GROUP BY order_id
)
SELECT
  vi.categoria,
  COUNT(DISTINCT vi.order_id) AS total_pedidos,
  ROUND(SUM(vi.total_item_value), 2) AS ingresos_totales,
  ROUND(AVG(vo.delay_days), 2) AS promedio_dias_demora,
  ROUND(100.0 * SUM(vo.es_tardio) / COUNT(DISTINCT vi.order_id), 2) AS tasa_entregas_tardias_pct,
  ROUND(AVG(vr.review_promedio_pedido), 2) AS score_satisfaccion_promedio
FROM v_items vi
INNER JOIN v_orders vo 
  ON vi.order_id = vo.order_id
LEFT JOIN v_reviews vr 
  ON vi.order_id = vr.order_id
GROUP BY vi.categoria
HAVING COUNT(DISTINCT vi.order_id) >= 100
ORDER BY ingresos_totales DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC - Insight 1: Crisis de satisfacción en mobiliario de oficina (Office Furniture)
# MAGIC Evidencia: La categoría de productos "office_furniture" genera un acumulado de más de $335,211.36 en ventas, pero registra la peor calificación promedio de todo el e-commerce, con apenas 3.62 estrellas.  
# MAGIC
# MAGIC Impacto operativo o de negocio: Esto representa un riesgo muy alto de pérdida de clientes corporativos, un incremento directo en los costos por solicitudes de devolución y un fuerte daño reputacional en un sector con tickets de venta elevados.
# MAGIC
# MAGIC Recomendación: Se debe auditar de inmediato a los proveedores y vendedores principales de esta categoría, evaluar sus tiempos de empaque y verificar si las descripciones e imágenes de los productos en la plataforma coinciden estrictamente con las dimensiones reales.Consulta
# MAGIC
# MAGIC SQL usada: Consulta de la sección 12 del archivo "05_estudiante_sql_business_analysis.ipynb", la cual filtra las categorías con alta venta y baja satisfacción.  
# MAGIC
# MAGIC - Insight 2: El impacto financiero del "Efecto Diciembre" en la logísticaEvidencia: 
# MAGIC
# MAGIC Durante noviembre de 2017, la tasa de entregas tardías sufrió un pico drástico alcanzando el 12.40% de los pedidos (con una demora promedio de 1.39 días). Esto arrastró la satisfacción de los meses posteriores, donde se demuestra que los pedidos con más de 7 días de retraso caen a una calificación crítica de 1.71 estrellas.  
# MAGIC
# MAGIC Impacto operativo o de negocio: El colapso logístico provocado por la temporada alta de fin de año (Black Friday y Navidad) destruye la experiencia de compra, impactando negativamente en la retención y recompra de clientes durante el primer trimestre del año siguiente.
# MAGIC
# MAGIC Recomendación: Es necesario establecer acuerdos de nivel de servicio más estrictos con las transportadoras para el último trimestre del año e implementar un algoritmo que ajuste las fechas estimadas de entrega de forma dinámica en la web durante eventos masivos.
# MAGIC
# MAGIC Consulta SQL usada: Consultas de las secciones 9 y 10 del archivo "05_estudiante_sql_business_analysis.ipynb", referentes a la tasa de entregas tardías y la relación entre demora y calificación.  
# MAGIC
# MAGIC - Insight 3: Alerta climática e inactividad operativa en el suministro de biomasaEvidencia:
# MAGIC El ingenio "Providencia" presenta la mayor interrupción operativa del ecosistema, acumulando un total de 421 días clasificados bajo la bandera de "Riesgo Bajo de Bagazo". Adicionalmente, las matrices de correlación estadística demuestran que las lluvias tienen un impacto negativo directo sobre el volumen de bagazo entregado (-0.2249) y la caña molida (-0.1426). 
# MAGIC
# MAGIC Impacto operativo o de negocio: Existe un riesgo inminente de paros no programados en la planta papelera principal debido a la falta de biomasa energética durante los meses de alta precipitación (como abril y mayo), donde las entregas caen a sus niveles más bajos.  
# MAGIC
# MAGIC Recomendación: Se recomienda diseñar y construir un stock de seguridad físico (inventario de amortiguación) en la planta para almacenar bagazo excedente durante los meses secos (enero a marzo), disminuyendo la dependencia del suministro diario en las épocas invernales más críticas.
# MAGIC
# MAGIC Consulta SQL usada: Consultas de las secciones 14, 15 y 18 del bloque B en el archivo "05_estudiante_sql_business_analysis.ipynb", enfocadas en caña y bagazo por ingenio, análisis mensual y correlaciones simples.  

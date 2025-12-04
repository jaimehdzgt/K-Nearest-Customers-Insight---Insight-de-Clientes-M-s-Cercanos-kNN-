# K-Nearest-Customers-Insight - Clientes Mas Cercanos-kNN
Ayuda a encontrar clientes “parecidos” entre sí, usando la información que ya tenemos de su comportamiento (compras, frecuencia, ticket, etc.).

## 🎯 Objetivo del proyecto

En NovaRetail Group, las áreas de **Marketing y CRM** necesitan identificar de forma sistemática qué clientes se parecen más a sus **mejores compradores**, con el fin de:

- Mejorar la **segmentación de campañas** (adquisición, retención, win-back, cross-sell y upsell).
- Aumentar la **tasa de conversión** en campañas ya existentes.
- Optimizar el uso del presupuesto de marketing enfocándolo en clientes con **alto potencial de valor**.

El **Customer Lookalike Finder** es un motor basado en **k-Nearest Neighbors (kNN)** que, dado un conjunto de clientes objetivo (por ejemplo, clientes VIP o de alto valor), encuentra otros clientes con un **perfil de comportamiento y valor similar** en la base de clientes de NovaRetail.

---

## 🧩 Enunciado de negocio

> “¿Cómo podemos aprovechar el histórico de comportamiento transaccional y de interacción de nuestros clientes para encontrar, de forma automatizada, aquellos que se parecen más a nuestros ‘mejores clientes’ y así dirigirles campañas personalizadas que aumenten la recurrencia y el ticket promedio?”

Con este proyecto se busca **demostrar y habilitar**:

1. Que el uso de modelos de similitud de clientes (lookalike modeling) permite construir segmentos **más precisos** que los filtros tradicionales por reglas (edad, región, ticket promedio, etc.).
2. Que un motor de lookalikes puede integrarse a los procesos actuales de CRM y campañas, entregando **listas accionables** de clientes para:
   - Campañas de retención de clientes de alto valor.
   - Venta cruzada de categorías estratégicas.
   - Reactivación de clientes con riesgo de fuga pero con alto potencial.
3. Que es posible medir el **impacto real** del uso del motor en indicadores de negocio:
   - Incremento del **ingreso incremental por campaña**.
   - Mejora del **CTR / tasa de apertura** en campañas digitales.
   - Reducción del **costo por conversión**.

---

## 👥 Stakeholders involucrados

El proyecto requiere la colaboración coordinada de varios equipos:

### Stakeholders de negocio

- **Chief Marketing Officer (CMO)**  
  - Sponsor del proyecto; define objetivos de negocio y KPIs de impacto.

- **Gerente de CRM / Marketing Relacional**  
  - Define la estrategia de segmentos, listas objetivo y uso de los lookalikes en campañas.
  - Valida criterios de negocio para segmentación y pruebas A/B.

- **Gerente de Ventas / Retail Operations**  
  - Aporta contexto sobre comportamiento en tienda física y necesidades comerciales.
  - Valida que los segmentos resultantes tengan sentido para la operación.

### Stakeholders de datos y tecnología

- **Data Science Lead / Data Scientist**  
  - Responsable del diseño del modelo de similitud (kNN).
  - Define features, métricas y criterios de evaluación.

- **BI & Analytics Team**  
  - Construye visualizaciones y tableros para interpretar los segmentos y lookalikes.
  - Da soporte a stakeholders de negocio para el uso del motor en campañas.

- **Data Engineering / IT**  
  - Se asegura de la disponibilidad y calidad de los datos de origen (DWH / Lakehouse / CRM).
  - Prepara las canalizaciones de datos (ETL/ELT) que alimentan el motor.

- **Equipo de MarTech / Marketing Automation**  
  - Integra las listas de lookalikes con las plataformas de email marketing, SMS, push y campañas pagadas.

---

## 📊 Datos utilizados

El motor se alimenta de información histórica de clientes de NovaRetail, incluyendo:

- **Información de cliente:**
  - `customer_id`, edad, nivel de ingresos estimado, región, canal de alta.
- **Comportamiento de compra:**
  - Número de órdenes históricas, frecuencia de compra, recencia,
  - Artículos promedio por pedido,
  - Gasto total histórico y distribución por categorías.
- **Preferencias de compra:**
  - Categoría favorita,
  - Horarios y días de compra más frecuentes.
- **Indicadores de valor:**
  - `vip_flag` para marcar clientes de alto valor (definidos por negocio),
  - `vip_score` interno para priorización.

> Nota: El archivo de trabajo se encuentra en `data/customer_lookalike_raw_100k.xlsx` y representa el histórico consolidado de clientes de NovaRetail extraído del CRM y del sistema de ventas omnicanal.

---

## 🧠 Enfoque analítico

![Arquitectura Lookalike](assets/diagram_lookalike_architecture.png)

1. **Exploración y calidad de datos**
   - Revisión de distribución de variables clave (ingresos, frecuencia, recencia, gasto total).
   - Detección y tratamiento de:
     - Formatos mixtos (porcentajes, montos con símbolos de moneda).
     - Valores especiales (por ejemplo, “never”, códigos especiales).
     - Inconsistencias en regiones y canales.

2. **Construcción de features por cliente**
   - Variables numéricas:
     - Edad, ingresos, gasto total histórico,
     - Número de órdenes, recencia, frecuencia,
     - Items promedio por compra,
     - Proporción de pedidos con reorden.
   - Variables categóricas:
     - Región limpia (`region_clean`),
     - Categoría favorita (`fav_category`),
     - Canal de alta (`signup_channel`).
   - Codificación mediante **One-Hot Encoding** para variables categóricas.

3. **Normalización**
   - Uso de `StandardScaler` para llevar todas las features a una escala comparable antes de aplicar kNN.

4. **Modelo de similitud (k-Nearest Neighbors)**
   - Uso de `NearestNeighbors` (scikit-learn) para encontrar, dado un cliente, sus vecinos más cercanos en el espacio de features.
   - El modelo se entrena sobre la matriz de features normalizadas (`X_scaled`).

5. **Función de negocio: `find_lookalikes(customer_id, k)`**
   - Entrada:
     - `customer_id`: cliente objetivo.
     - `k`: número de vecinos deseados.
   - Salida:
     - DataFrame con:
       - `customer_id` del lookalike,
       - indicadores de valor (`vip_flag`, gasto total, total de órdenes),
       - contexto de segmento (región, categoría, etc.),
       - `distance` en el espacio de features (medida de similitud).

6. **Visualizaciones clave**
   - Distribuciones de variables críticas (ingreso, gasto, frecuencia).
   - Proyección 2D con **PCA** para:
     - Visualizar la nube de clientes,
     - Resaltar la ubicación de los clientes VIP,
     - Entender cómo se agrupan segmentos de interés.

---

## 🔧 Stack tecnológico

- **Lenguaje:** Python 3.x  
- **Librerías principales:**
  - `pandas` – preparación y manipulación de datos
  - `numpy` – operaciones numéricas
  - `scikit-learn` – kNN, normalización, One-Hot Encoding, PCA
  - `matplotlib` – visualización
- **Ejecución:** GitHub Codespaces (entorno VS Code en la nube, sin instalar nada localmente)

---

## 🗂 Estructura del repositorio

```text
.
├── assets/
│   ├── header_lookalike.png               # Banner del proyecto
│   ├── diagram_lookalike_architecture.png # Diagrama de arquitectura (opcional)
│   └── pca_scatter_vip.png                # Ejemplo de gráfico PCA (opcional)
├── data/
│   └── customer_lookalike_raw_100k.xlsx   # Historial de clientes NovaRetail
├── knn_lookalike_project.py               # Script principal del motor de lookalikes
├── requirements.txt                       # Dependencias del proyecto
└── README.md

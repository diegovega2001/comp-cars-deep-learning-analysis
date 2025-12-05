# Scripts - Estructura del Proyecto

Este directorio contiene todos los scripts organizados por funcionalidad.

## 📁 Estructura

```
scripts/
├── extraction/          → Extracción de datos desde JSON a CSV
├── analysis/           → Análisis estadístico cuantitativo
├── visualization/      → Generación de gráficos y plots
├── pipeline/           → Pipelines de entrenamiento y embeddings
└── README.md          → Este archivo
```

---

## 🔍 **1. extraction/** - Extracción de Datos

### `extract_vision_results.py`
**Propósito:** Extrae resultados de experimentos ResNet50 y ViT-B/32 desde JSON a CSV.

**Entrada:**
- `results/models/resnet50/*/results_*.json`
- `results/models/vit_b_32/*/results_*.json`

**Salida:**
- `results/analysis/vision_models_results.csv` (24 configuraciones)
- `results/analysis/vision_finetuning_results.csv`
- `results/analysis/vision_embeddings_results.csv`

**Uso:**
```bash
uv run scripts/extraction/extract_vision_results.py
```

**Datos extraídos:**
- Configuración: modelo, objetivo, criterio, vistas
- Métricas: accuracy, recall@1/3/5, ARI, NMI, purity
- Clustering: n_clusters, pure_percentage, overlapping_classes
- Temporal: duration_minutes, epochs, best_epoch

---

### `extract_clip_layers_results.py`
**Propósito:** Extrae resultados del estudio CLIP layers (1-12 capas, vision/text).

**Entrada:**
- `results/models/clip-vit-base-patch32/CLIP/results_*.json`

**Salida:**
- `results/analysis/clip_layers_results.csv` (24 configuraciones)
- `results/analysis/clip_layers_finetuning_results.csv`
- `results/analysis/clip_layers_embeddings_results.csv`

**Uso:**
```bash
uv run scripts/extraction/extract_clip_layers_results.py
```

**Datos extraídos:**
- Configuración: component_type (vision/text), num_layers (1-12)
- Métricas: recall@1/3/5, ARI, purity, clusters puros/mixtos
- Progresión: correlación capas ↔ métricas

---

## 📊 **2. analysis/** - Análisis Estadístico

### `analyze_vision_models.py`
**Propósito:** Análisis estadístico completo de vision models (ResNet50/ViT-B/32).

**Entrada:**
- `results/analysis/vision_models_results.csv`

**Salida (6 CSVs en `results/analysis/statistics/`):**
1. `vision_model_analysis.csv` - Comparación ResNet50 vs ViT-B/32
2. `vision_objective_analysis.csv` - Classification vs Metric Learning
3. `vision_criterion_analysis.csv` - 6 funciones de pérdida
4. `vision_views_analysis.csv` - Front vs Front+Rear
5. `vision_clustering_analysis.csv` - Calidad de clustering
6. `vision_summary.csv` - Tabla resumen completa

**Uso:**
```bash
$env:PYTHONIOENCODING='utf-8'
uv run scripts/analysis/analyze_vision_models.py
```

**Análisis incluidos:**
- Comparación modelos, objetivos, criterios, vistas
- Calidad clustering: ARI, purity, % clusters puros
- Clases problemáticas (overlap)
- Baseline vs Finetuned
- Eficiencia temporal (ARI/hora)
- Rankings y top configuraciones

**Hallazgos clave:**
- ResNet50: 0.2665 ARI (128% mejor que ViT-B/32)
- Front+rear: 0.3254 ARI (510% mejor que front)
- Mejor: resnet50 + ntxent + front+rear (0.8806 ARI, 82% pure clusters)

---

### `analyze_clip_layers.py`
**Propósito:** Análisis estadístico del estudio CLIP layers (progresión 1-12 capas).

**Entrada:**
- `results/analysis/clip_layers_results.csv`

**Salida (3 CSVs en `results/analysis/statistics/`):**
1. `clip_component_analysis.csv` - Vision vs Text comparison
2. `clip_layers_analysis.csv` - Progresión por número de capas
3. `clip_summary.csv` - Resumen completo

**Uso:**
```bash
$env:PYTHONIOENCODING='utf-8'
uv run scripts/analysis/analyze_clip_layers.py
```

**Análisis incluidos:**
- Comparación vision vs text component
- Progresión por número de capas (1-12)
- Correlaciones capas ↔ métricas
- Calidad clustering por capas
- Clases problemáticas por componente
- Eficiencia temporal

**Hallazgos clave:**
- Vision: 0.4439 ARI (76% mejor que text: 0.2526)
- Optimal: 11 layers vision, 9 layers text
- Correlación vision capas→recall: 0.92 (fuerte)
- Correlación text capas→recall: 0.59 (moderada)

---

## 📈 **3. visualization/** - Visualizaciones

### `visualize_training_history.py`
**Propósito:** Genera gráficos de curvas de entrenamiento para vision models.

**Entrada:**
- `results/models/resnet50/*/results_*.json`
- `results/models/vit_b_32/*/results_*.json`

**Salida:**
- 24 gráficos PNG en `results/visualizations/training_history_plots/`
- Formato: `{views}_{model}_{objective}_{criterion}_training.png`

**Uso:**
```bash
$env:PYTHONIOENCODING='utf-8'
uv run scripts/visualization/visualize_training_history.py
```

**Métricas visualizadas:**
- Train/Val Loss
- Accuracy
- Recall@1, Recall@3, Recall@5

---

### `visualize_training_history_clip.py`
**Propósito:** Genera gráficos de curvas de entrenamiento para CLIP layers.

**Entrada:**
- `results/models/clip-vit-base-patch32/CLIP/results_*.json`

**Salida:**
- 24 gráficos PNG en `results/visualizations/training_history_plots/`
- Formato: `{layers}_layers_{component}_clip-vit-base-patch32_CLIP_training.png`

**Uso:**
```bash
$env:PYTHONIOENCODING='utf-8'
uv run scripts/visualization/visualize_training_history_clip.py
```

---

### `visualize_vision_models.py`
**Propósito:** Genera 12 visualizaciones comparativas avanzadas para vision models.

**Entrada:**
- `results/analysis/vision_models_results.csv`

**Salida (12 gráficos en `results/analysis/plots/vision_models/`):**
1. `01_model_comparison.png` - ResNet50 vs ViT-B/32
2. `02_objective_comparison.png` - Classification vs Metric Learning
3. `03_criterion_comparison.png` - 6 funciones de pérdida
4. `04_views_comparison.png` - Front vs Front+Rear
5. `05_heatmap_recall1.png` - Modelo × Criterio
6. `06_heatmap_ari.png` - Modelo × Criterio
7. `07_cluster_distribution.png` - Puros vs Mixtos
8. `08_clustering_quality_ranking.png` - Ranking 24 configs
9. `09_problematic_classes.png` - Análisis overlap
10. `10_finetuning_improvement.png` - Baseline → Finetuned
11. `11_temporal_efficiency.png` - ARI/hora
12. `12_top5_summary.png` - Top-5 por métrica

**Uso:**
```bash
$env:PYTHONIOENCODING='utf-8'
uv run scripts/visualization/visualize_vision_models.py
```

**Características:**
- Alta resolución (300 DPI)
- Color-coded por categorías
- Anotaciones con valores exactos
- Correlaciones y estadísticas

---

### `visualize_clip_layers.py`
**Propósito:** Genera 10 visualizaciones de progresión para CLIP layers.

**Entrada:**
- `results/analysis/clip_layers_results.csv`

**Salida (10 gráficos en `results/analysis/plots/clip_layers/`):**
1. `01_component_comparison.png` - Vision vs Text
2. `02_recall_progression.png` - Recall@1/3/5 por capas
3. `03_ari_progression.png` - ARI con máximos marcados
4. `04_vision_vs_text_by_layers.png` - Comparación cruzada
5. `05_layer_metric_correlation.png` - Correlaciones 0.92 vs 0.59
6. `06_heatmap_layers.png` - Componente × Capas
7. `07_clustering_quality_by_layers.png` - Puros/mixtos/purity
8. `08_temporal_efficiency.png` - Tiempo y ARI/hora
9. `09_configuration_ranking.png` - Top-10 por métrica
10. `10_complete_summary.png` - Panel integrado con estadísticas

**Uso:**
```bash
$env:PYTHONIOENCODING='utf-8'
uv run scripts/visualization/visualize_clip_layers.py
```

**Características:**
- Progresiones con líneas y markers
- Máximos destacados con estrellas
- Correlaciones calculadas y mostradas
- Resumen con estadísticas clave

---

## ⚙️ **4. pipeline/** - Pipelines de Entrenamiento

### `Finetuning.py`
**Propósito:** Pipeline completo de fine-tuning de modelos vision.

**Funcionalidades:**
- Carga de modelos: ResNet50, ViT-B/32, CLIP
- Fine-tuning con classification o metric learning
- Criterios: CrossEntropy, arcface, contrastive, multisimilarity, ntxent, triplet
- Evaluación: accuracy, recall@1/3/5
- Guardado de checkpoints y resultados JSON

**Uso típico:**
```python
from scripts.pipeline.Finetuning import FineTuningPipeline

pipeline = FineTuningPipeline(config_path='configs/resnet50_metric_learning.yaml')
results = pipeline.run()
```

---

### `Embeddings.py`
**Propósito:** Pipeline de generación de embeddings y clustering.

**Funcionalidades:**
- Extracción de embeddings con modelos entrenados
- Reducción dimensional: PCA, UMAP, t-SNE
- Clustering: DBSCAN, HDBSCAN, Agglomerative, OPTICS
- Métricas: ARI, NMI, purity, silhouette
- Análisis de clases problemáticas
- Visualizaciones: t-SNE, confusion matrix, cluster analysis

**Uso típico:**
```python
from scripts.pipeline.Embeddings import EmbeddingsPipeline

pipeline = EmbeddingsPipeline(config_path='configs/embeddings.yaml')
results = pipeline.run()
```

---

## 🚀 **Flujo de Trabajo Completo**

### 1. Entrenamiento (pipeline/)
```bash
# Fine-tuning vision models
uv run scripts/pipeline/Finetuning.py --config configs/resnet50_metric_learning.yaml

# Generación de embeddings y clustering
uv run scripts/pipeline/Embeddings.py --config configs/embeddings.yaml
```

### 2. Extracción de datos (extraction/)
```bash
# Extraer resultados vision models
uv run scripts/extraction/extract_vision_results.py

# Extraer resultados CLIP layers
uv run scripts/extraction/extract_clip_layers_results.py
```

### 3. Análisis estadístico (analysis/)
```bash
# Análisis vision models
$env:PYTHONIOENCODING='utf-8'
uv run scripts/analysis/analyze_vision_models.py

# Análisis CLIP layers
uv run scripts/analysis/analyze_clip_layers.py
```

### 4. Visualizaciones (visualization/)
```bash
# Gráficos de entrenamiento
uv run scripts/visualization/visualize_training_history.py
uv run scripts/visualization/visualize_training_history_clip.py

# Visualizaciones comparativas
uv run scripts/visualization/visualize_vision_models.py
uv run scripts/visualization/visualize_clip_layers.py
```

---

## 📊 **Resumen de Outputs**

| Fase | Scripts | Outputs | Total |
|------|---------|---------|-------|
| Extracción | 2 | 6 CSVs | 6 |
| Training Plots | 2 | 48 PNGs | 48 |
| Análisis | 2 | 9 CSVs | 9 |
| Visualizaciones | 2 | 22 PNGs | 22 |
| **TOTAL** | **8** | **85 archivos** | **85** |

---

## 🔧 **Requisitos**

### Dependencias principales:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- torch
- torchvision
- transformers

### Instalación:
```bash
uv pip install -r requirements.txt
```

---

## 📝 **Notas**

### Encoding UTF-8:
Para Windows PowerShell, usar siempre:
```powershell
$env:PYTHONIOENCODING='utf-8'
```

### Ejecución con uv:
Todos los scripts deben ejecutarse con `uv run` para usar el entorno correcto.

### Alta resolución:
Todos los gráficos se generan a 300 DPI para calidad de publicación.

---

**Última actualización:** Diciembre 4, 2025

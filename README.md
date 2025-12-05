# CompCars - Análisis de Vehículos con Deep Learning

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-package_manager-green.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Descripción

Proyecto de investigación para análisis y clasificación de vehículos utilizando el dataset **CompCars**. Basado en **PyTorch**, **scikit-learn** y **pytorch-metric-learning**, implementa modelos de deep learning (ResNet50, ViT, CLIP), análisis de embeddings, clustering avanzado y visualizaciones comparativas para estudios de fine-tuning multi-vista.

### Objetivos Principales

- **Fine-tuning** de modelos pre-entrenados (ResNet50, ViT-B/32, CLIP)
- **Análisis comparativo** de arquitecturas y funciones de pérdida
- **Estudio CLIP layers** (progresión 1-12 capas, vision/text components)
- **Clustering avanzado** con DBSCAN, HDBSCAN, Agglomerative
- **Análisis de embeddings** baseline vs fine-tuned
- **Soporte multi-vista** (front/rear) de vehículos
- **Visualizaciones** estadísticas y plots de entrenamiento

## 📂 Arquitectura del Proyecto

```
Memoria/
├── src/                          # Código fuente principal
│   ├── config/                   # Configuraciones
│   │   └── TransformConfig.py   # Transformaciones de imágenes
│   ├── data/                     # Procesamiento de datos      
│   │   ├── DataFrameMaker.py    # Generación de dataset CSV
│   │   └── MyDataset.py         # Dataset PyTorch personalizado
│   ├── models/                   # Arquitecturas de modelos
│   │   ├── Criterions.py        # Funciones de pérdida (metric learning)
│   │   ├── MyVisionModel.py     # ResNet50, ViT-B/32 multi-vista
│   │   └── MyCLIPModel.py       # CLIP (vision + text components)
│   ├── pipeline/                 # Pipelines de ML
│   │   ├── FineTuningPipeline.py   # Pipeline de fine-tuning
│   │   └── EmbeddingsPipeline.py   # Pipeline de análisis
│   └── utils/                    # Utilidades
│       ├── ClusteringAnalyzer.py  # DBSCAN, HDBSCAN, etc.
│       ├── DimensionalityReducer.py  # PCA, t-SNE, UMAP
│       └── ClusterVisualizer.py   # Visualizaciones
│
├── scripts/                      # Scripts de análisis
│   ├── extraction/               # Extracción de resultados JSON → CSV
│   │   ├── extract_vision_results.py
│   │   └── extract_clip_layers_results.py
│   ├── analysis/                 # Análisis estadístico
│   │   ├── analyze_vision_models.py
│   │   └── analyze_clip_layers.py
│   ├── visualization/            # Generación de gráficos
│   │   ├── visualize_training_history.py
│   │   ├── visualize_training_history_clip.py
│   │   ├── visualize_vision_models.py
│   │   └── visualize_clip_layers.py
│   ├── pipeline/                 # Scripts de entrenamiento
│   │   ├── Finetuning.py
│   │   └── Embeddings.py
│   └── README.md                 # Documentación de scripts
│
├── configs/                      # Configuraciones YAML
│   ├── resnet50_*.yaml          # Configs ResNet50
│   ├── vitb32_*.yaml            # Configs ViT-B/32
│   ├── CLIP.yaml                # Config CLIP layers
│   └── embeddings.yaml          # Config análisis embeddings
│
├── dataset.csv                   # Dataset generado (163 modelos)
├── requirements.txt              # Dependencias del proyecto
├── requirements-dev.txt          # Dependencias desarrollo
├── pyproject.toml                # Configuración uv + proyecto
└── README.md                    # Este archivo

# Carpetas ignoradas (no en git):
# results/                        → Modelos, embeddings, análisis
# dataset/image/                  → Imágenes CompCars (~214k)
# dataset/label/                  → Metadatos y anotaciones
# .venv/                          → Entorno virtual
```

## 🚀 Instalación y Configuración

### 1. Prerrequisitos

- **Python 3.10+** (proyecto usa Python 3.10.19)
- **uv** - Gestor de paquetes rápido ([Instalación](https://github.com/astral-sh/uv))
- **CUDA GPU** (opcional, recomendado para entrenamiento)
- **Git** para clonar el repositorio

### 2. Instalar uv (si no lo tienes)

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verificar instalación
uv --version
```

### 3. Clonar el repositorio

```bash
git clone https://github.com/diegovega2001/Memoria.git
cd Memoria
```

### 4. Configurar entorno con uv

```bash
# uv creará automáticamente el entorno virtual y sincronizará dependencias
uv sync

# Verificar instalación
uv run python --version  # Debe mostrar Python 3.10.19
```

### 5. Verificar instalación

```bash
# Verificar PyTorch
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Verificar scikit-learn
uv run python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"

# Verificar CUDA (si disponible)
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Nota:** `uv` gestiona automáticamente el entorno virtual, no es necesario activarlo manualmente.

## 💡 Uso del Proyecto

### Flujo Completo de Análisis

Este proyecto está organizado en **4 fases** secuenciales:

#### **Fase 1: Fine-tuning de modelos**
Entrenar modelos con diferentes configuraciones.

```bash
# Ejemplo: Fine-tuning ResNet50 con metric learning
uv run scripts/pipeline/Finetuning.py --config configs/resnet50_metric_learning.yaml
```

#### **Fase 2: Generación de embeddings y clustering**
Extraer embeddings y aplicar clustering.

```bash
# Generar embeddings y análisis de clustering
uv run scripts/pipeline/Embeddings.py --config configs/embeddings.yaml
```

#### **Fase 3: Extracción de resultados a CSV**
Consolidar resultados JSON en CSVs estructurados.

```bash
# Extraer resultados de vision models (ResNet50/ViT-B/32)
uv run scripts/extraction/extract_vision_results.py

# Extraer resultados de CLIP layers study
uv run scripts/extraction/extract_clip_layers_results.py
```

#### **Fase 4: Análisis estadístico y visualizaciones**
Generar análisis comparativos y gráficos.

```bash
# Análisis estadístico
$env:PYTHONIOENCODING='utf-8'  # Windows PowerShell
uv run scripts/analysis/analyze_vision_models.py
uv run scripts/analysis/analyze_clip_layers.py

# Visualizaciones
uv run scripts/visualization/visualize_training_history.py
uv run scripts/visualization/visualize_vision_models.py
uv run scripts/visualization/visualize_clip_layers.py
```

### Configuraciones Disponibles

El proyecto incluye múltiples configuraciones experimentales en `configs/`:

**Vision Models:**
- `resnet50_classification.yaml` - ResNet50 con CrossEntropy
- `resnet50_metric_learning.yaml` - ResNet50 con arcface/contrastive/ntxent/triplet/multisimilarity
- `vitb32_classification.yaml` - ViT-B/32 con CrossEntropy
- `vitb32_metric_learning.yaml` - ViT-B/32 con metric learning

**CLIP:**
- `CLIP.yaml` - Estudio de capas CLIP (1-12 layers, vision/text components)

**Embeddings:**
- `embeddings.yaml` - Configuración de análisis y clustering

## 📊 Resultados del Proyecto

El análisis completo generó **85 archivos** de resultados:

### Vision Models (ResNet50/ViT-B/32)
- **24 configuraciones** analizadas
- **6 CSVs** de datos extraídos
- **24 gráficos** de curvas de entrenamiento
- **12 visualizaciones** comparativas
- **6 CSVs** de análisis estadístico

**Hallazgos principales:**
- ✅ ResNet50: **0.2665 ARI** (128% mejor que ViT-B/32)
- ✅ Metric Learning: **0.2058 ARI** (71% mejor que Classification)
- ✅ Front+Rear: **0.3254 ARI** (510% mejor que Front solo)
- ✅ Mejor config: **resnet50 + ntxent + front+rear** → 0.8806 ARI, 82% clusters puros
- ✅ 22/24 configuraciones mejoraron con finetuning (+53% ARI promedio)

### CLIP Layers Study
- **24 configuraciones** (12 vision + 12 text, 1-12 layers)
- **3 CSVs** de datos extraídos
- **24 gráficos** de curvas de entrenamiento
- **10 visualizaciones** de progresión
- **3 CSVs** de análisis estadístico

**Hallazgos principales:**
- ✅ Vision component: **0.4439 ARI** (76% mejor que Text: 0.2526)
- ✅ Optimal layers: **11 layers** (vision), **9 layers** (text)
- ✅ Correlación vision capas→recall: **0.92** (muy fuerte)
- ✅ Correlación text capas→recall: **0.59** (moderada)
- ✅ Mejor config: **vision 11 layers** → 0.4740 ARI, 56% clusters puros

### Estructura de Resultados

```
results/
├── analysis/
│   ├── plots/
│   │   ├── vision_models/          # 12 gráficos comparativos
│   │   └── clip_layers/            # 10 gráficos de progresión
│   ├── statistics/                 # 9 CSVs de análisis estadístico
│   ├── vision_models_results.csv   # 24 configs vision
│   └── clip_layers_results.csv     # 24 configs CLIP
└── visualizations/
    └── training_history_plots/     # 48 gráficos de entrenamiento
```

Ver `scripts/README.md` para documentación detallada de cada fase.

## 🔬 Características Técnicas

### **Modelos Soportados**
- **ResNet50** - Arquitectura CNN clásica (2048-dim embeddings)
- **ViT-B/32** - Vision Transformer (768-dim embeddings)
- **CLIP** - Modelo multimodal vision + text (512-dim embeddings)

### **Objetivos de Entrenamiento**
- **Classification** - CrossEntropy loss para clasificación directa
- **Metric Learning** - Aprendizaje de espacio métrico con **pytorch-metric-learning**:
  - ArcFace Loss
  - Contrastive Loss
  - MultiSimilarity Loss
  - NTXent Loss (NT-Xent)
  - Triplet Loss

### **Clustering & Análisis**
- **Algoritmos (scikit-learn):** DBSCAN, HDBSCAN, Agglomerative, OPTICS
- **Reducción dimensional:** PCA, t-SNE, UMAP
- **Métricas (scikit-learn):** ARI, NMI, Purity, Silhouette, % clusters puros
- **Visualizaciones:** t-SNE plots, heatmaps, rankings, confusion matrices

### **Multi-Vista**
- Soporte front/rear simultáneo
- Fusión de características por concatenación
- Análisis comparativo front vs front+rear

### **Reproducibilidad**
- Seeds fijadas (Python, NumPy, PyTorch, CUDA)
- Configuraciones YAML versionadas
- Resultados JSON con timestamp
- Logging detallado de experimentos

## Dataset CompCars

El proyecto utiliza el dataset **CompCars** que contiene:

- **163 marcas de vehículos**
- **1,716 modelos diferentes**
- **~214,000 imágenes**
- **Múltiples viewpoints** (front, rear, side)
- **Bounding boxes** para cada vehículo
- **Metadatos** (año, tipo, modelo)

### Estructura del CSV generado:

```csv
image_name,image_path,released_year,viewpoint,bbox,make,model,type
826a5fd082682c,dataset/image/135/947/unknown/826a5fd082682c.jpg,unknown,rear,"[96.0, 53.0, 817.0, 596.0]",Saab,SAAB 9X,Unknown
```

## 🛠️ Dependencias Principales

El proyecto utiliza `uv` para gestión rápida de dependencias:

**Core ML:**
- `torch` >= 2.2.0 - Deep learning framework
- `torchvision` >= 0.17.0 - Modelos pre-entrenados y transformaciones
- `pytorch-metric-learning` >= 2.5.0 - Funciones de pérdida y minería (ArcFace, NTXent, Triplet, etc.)
- `transformers` >= 4.38.0 - CLIP y otros modelos
- `scikit-learn` >= 1.4.0 - Clustering (DBSCAN, HDBSCAN) y métricas (ARI, NMI)

**Análisis & Visualización:**
- `pandas` >= 2.2.0 - Manipulación de datos
- `numpy` >= 1.26.0 - Operaciones numéricas
- `matplotlib` >= 3.8.0 - Gráficos
- `seaborn` >= 0.13.0 - Visualizaciones estadísticas

**Utilidades:**
- `pyyaml` - Configuraciones YAML
- `tqdm` - Progress bars
- `pillow` - Procesamiento de imágenes

Ver `requirements.txt` para lista completa.

## 🧪 Testing

```bash
# Ejecutar tests
uv run pytest tests/

# Tests con coverage
uv run pytest tests/ --cov=src --cov-report=html
```

## 📝 Scripts Útiles

```bash
# Ver versiones de dependencias
uv pip list

# Actualizar dependencias
uv sync --upgrade

# Ejecutar script específico
uv run scripts/analysis/analyze_vision_models.py

# Verificar instalación de PyTorch
uv run python -c "import torch; print(torch.__version__)"
```

## Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## Autor

**Diego Vega** - [diegovega2001](https://github.com/diegovega2001)
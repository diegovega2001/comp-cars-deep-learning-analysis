#!/usr/bin/env python3
"""
Script de visualización para análisis de Vision Models (ResNet50/ViT-B/32)
Genera gráficos comparativos de rendimiento y clustering
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
RESULTS_DIR = Path('results/analysis')
PLOTS_DIR = Path('results/analysis/plots/vision_models')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Cargar datos
print("="*80)
print("GENERANDO VISUALIZACIONES - VISION MODELS")
print("="*80)
print()

df = pd.read_csv(RESULTS_DIR / 'vision_models_results.csv')
print(f"Datos cargados: {len(df)} configuraciones\n")

# ============================================================================
# 1. COMPARACIÓN POR MODELO
# ============================================================================
print("[1/12] Comparación por modelo...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparación ResNet50 vs ViT-B/32', fontsize=16, fontweight='bold')

# Recall@1
model_stats = df.groupby('model_name')[['best_val_recall@1', 'best_val_recall@3', 'best_val_recall@5']].mean()
ax = axes[0, 0]
model_stats[['best_val_recall@1', 'best_val_recall@3', 'best_val_recall@5']].plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#9b59b6'])
ax.set_title('Recall Promedio por Modelo', fontweight='bold')
ax.set_ylabel('Recall (%)')
ax.set_xlabel('Modelo')
ax.legend(['Recall@1', 'Recall@3', 'Recall@5'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)

# ARI
ax = axes[0, 1]
ari_stats = df.groupby('model_name')[['finetuned_ari']].mean()
bars = ax.bar(ari_stats.index, ari_stats['finetuned_ari'], color=['#e74c3c', '#3498db'])
ax.set_title('ARI Promedio por Modelo', fontweight='bold')
ax.set_ylabel('ARI')
ax.set_xlabel('Modelo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom')

# Clusters puros
ax = axes[1, 0]
cluster_stats = df.groupby('model_name')[['finetuned_pure_percentage']].mean()
bars = ax.bar(cluster_stats.index, cluster_stats['finetuned_pure_percentage'], color=['#f39c12', '#16a085'])
ax.set_title('% Clusters Puros Promedio por Modelo', fontweight='bold')
ax.set_ylabel('% Clusters Puros')
ax.set_xlabel('Modelo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom')

# Tiempo
ax = axes[1, 1]
time_stats = df.groupby('model_name')[['duration_minutes']].mean()
bars = ax.bar(time_stats.index, time_stats['duration_minutes'], color=['#8e44ad', '#27ae60'])
ax.set_title('Tiempo de Entrenamiento Promedio', fontweight='bold')
ax.set_ylabel('Duración (minutos)')
ax.set_xlabel('Modelo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f} min', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. COMPARACIÓN POR OBJETIVO
# ============================================================================
print("[2/12] Comparación por objetivo...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparación Classification vs Metric Learning', fontsize=16, fontweight='bold')

# Recall promedio
obj_stats = df.groupby('objective')[['best_val_recall@1', 'best_val_recall@3', 'best_val_recall@5']].mean()
ax = axes[0, 0]
obj_stats.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#9b59b6'])
ax.set_title('Recall Promedio por Objetivo', fontweight='bold')
ax.set_ylabel('Recall (%)')
ax.set_xlabel('Objetivo')
ax.legend(['Recall@1', 'Recall@3', 'Recall@5'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', alpha=0.3)

# ARI
ax = axes[0, 1]
ari_obj = df.groupby('objective')[['finetuned_ari']].mean()
bars = ax.bar(ari_obj.index, ari_obj['finetuned_ari'], color=['#e67e22', '#1abc9c'])
ax.set_title('ARI Promedio por Objetivo', fontweight='bold')
ax.set_ylabel('ARI')
ax.set_xlabel('Objetivo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom')

# Clusters puros
ax = axes[1, 0]
pure_obj = df.groupby('objective')[['finetuned_pure_percentage']].mean()
bars = ax.bar(pure_obj.index, pure_obj['finetuned_pure_percentage'], color=['#c0392b', '#2980b9'])
ax.set_title('% Clusters Puros por Objetivo', fontweight='bold')
ax.set_ylabel('% Clusters Puros')
ax.set_xlabel('Objetivo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom')

# Número de configuraciones
ax = axes[1, 1]
config_count = df.groupby('objective').size()
bars = ax.bar(config_count.index, config_count.values, color=['#d35400', '#16a085'])
ax.set_title('Número de Configuraciones', fontweight='bold')
ax.set_ylabel('Cantidad')
ax.set_xlabel('Objetivo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_objective_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. COMPARACIÓN POR FUNCIÓN DE PÉRDIDA (CRITERION)
# ============================================================================
print("[3/12] Comparación por función de pérdida...")

# Filtrar solo metric learning
df_ml = df[df['objective'] == 'metric_learning'].copy()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Comparación de Funciones de Pérdida (Metric Learning)', fontsize=16, fontweight='bold')

# Recall@1
ax = axes[0, 0]
criterion_stats = df_ml.groupby('criterion')['best_val_recall@1'].mean().sort_values(ascending=False)
bars = ax.barh(range(len(criterion_stats)), criterion_stats.values, color=sns.color_palette("husl", len(criterion_stats)))
ax.set_yticks(range(len(criterion_stats)))
ax.set_yticklabels(criterion_stats.index)
ax.set_xlabel('Recall@1 (%)')
ax.set_title('Recall@1 Promedio por Criterio', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(criterion_stats.values):
    ax.text(v + 0.5, i, f'{v:.2f}', va='center')

# ARI
ax = axes[0, 1]
ari_crit = df_ml.groupby('criterion')['finetuned_ari'].mean().sort_values(ascending=False)
bars = ax.barh(range(len(ari_crit)), ari_crit.values, color=sns.color_palette("husl", len(ari_crit)))
ax.set_yticks(range(len(ari_crit)))
ax.set_yticklabels(ari_crit.index)
ax.set_xlabel('ARI')
ax.set_title('ARI Promedio por Criterio', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(ari_crit.values):
    ax.text(v + 0.01, i, f'{v:.4f}', va='center')

# Recall@5
ax = axes[1, 0]
recall5_crit = df_ml.groupby('criterion')['best_val_recall@5'].mean().sort_values(ascending=False)
bars = ax.barh(range(len(recall5_crit)), recall5_crit.values, color=sns.color_palette("husl", len(recall5_crit)))
ax.set_yticks(range(len(recall5_crit)))
ax.set_yticklabels(recall5_crit.index)
ax.set_xlabel('Recall@5 (%)')
ax.set_title('Recall@5 Promedio por Criterio', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(recall5_crit.values):
    ax.text(v + 0.5, i, f'{v:.2f}', va='center')

# Clusters puros
ax = axes[1, 1]
pure_crit = df_ml.groupby('criterion')['finetuned_pure_percentage'].mean().sort_values(ascending=False)
bars = ax.barh(range(len(pure_crit)), pure_crit.values, color=sns.color_palette("husl", len(pure_crit)))
ax.set_yticks(range(len(pure_crit)))
ax.set_yticklabels(pure_crit.index)
ax.set_xlabel('% Clusters Puros')
ax.set_title('% Clusters Puros por Criterio', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(pure_crit.values):
    ax.text(v + 1, i, f'{v:.1f}%', va='center')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_criterion_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. COMPARACIÓN FRONT vs FRONT+REAR
# ============================================================================
print("[4/12] Comparación de vistas...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparación Front vs Front+Rear', fontsize=16, fontweight='bold')

# Recall promedio
views_stats = df.groupby('views')[['best_val_recall@1', 'best_val_recall@3', 'best_val_recall@5']].mean()
ax = axes[0, 0]
views_stats.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db', '#2ecc71'])
ax.set_title('Recall Promedio por Vista', fontweight='bold')
ax.set_ylabel('Recall (%)')
ax.set_xlabel('Vista')
ax.legend(['Recall@1', 'Recall@3', 'Recall@5'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)

# ARI
ax = axes[0, 1]
ari_views = df.groupby('views')[['finetuned_ari']].mean()
bars = ax.bar(ari_views.index, ari_views['finetuned_ari'], color=['#f39c12', '#27ae60'])
ax.set_title('ARI Promedio por Vista', fontweight='bold')
ax.set_ylabel('ARI')
ax.set_xlabel('Vista')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom')

# Comparación visual con mejora porcentual
ax = axes[1, 0]
if 'front' in ari_views.index and 'front+rear' in ari_views.index:
    improvement = ((ari_views.loc['front+rear', 'finetuned_ari'] - ari_views.loc['front', 'finetuned_ari']) / 
                   ari_views.loc['front', 'finetuned_ari'] * 100)
    bars = ax.bar(['Front', 'Front+Rear'], 
                   [ari_views.loc['front', 'finetuned_ari'], ari_views.loc['front+rear', 'finetuned_ari']],
                   color=['#e67e22', '#16a085'])
    ax.set_title(f'ARI: Front+Rear es {improvement:.1f}% mejor', fontweight='bold')
    ax.set_ylabel('ARI')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Clusters puros
ax = axes[1, 1]
pure_views = df.groupby('views')[['finetuned_pure_percentage']].mean()
bars = ax.bar(pure_views.index, pure_views['finetuned_pure_percentage'], color=['#c0392b', '#2980b9'])
ax.set_title('% Clusters Puros por Vista', fontweight='bold')
ax.set_ylabel('% Clusters Puros')
ax.set_xlabel('Vista')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_views_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. HEATMAP MODELO × CRITERIO × RECALL@1
# ============================================================================
print("[5/12] Heatmap modelo × criterio × recall@1...")

# Crear matriz para heatmap
pivot_recall1 = df.pivot_table(values='best_val_recall@1', 
                                 index='criterion', 
                                 columns='model_name', 
                                 aggfunc='mean')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pivot_recall1, annot=True, fmt='.2f', cmap='YlOrRd', 
            cbar_kws={'label': 'Recall@1 (%)'}, ax=ax)
ax.set_title('Recall@1: Modelo × Criterio', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Modelo', fontweight='bold')
ax.set_ylabel('Criterio', fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_heatmap_recall1.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. HEATMAP MODELO × CRITERIO × ARI
# ============================================================================
print("[6/12] Heatmap modelo × criterio × ARI...")

pivot_ari = df.pivot_table(values='finetuned_ari', 
                            index='criterion', 
                            columns='model_name', 
                            aggfunc='mean')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pivot_ari, annot=True, fmt='.4f', cmap='RdYlGn', 
            cbar_kws={'label': 'ARI'}, ax=ax, vmin=0, vmax=1)
ax.set_title('ARI: Modelo × Criterio', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Modelo', fontweight='bold')
ax.set_ylabel('Criterio', fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '06_heatmap_ari.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. DISTRIBUCIÓN DE CLUSTERS PUROS VS MIXTOS
# ============================================================================
print("[7/12] Distribución clusters puros vs mixtos...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Distribución de Clusters: Puros vs Mixtos', fontsize=16, fontweight='bold')

# Por modelo
ax = axes[0]
model_cluster_data = df.groupby('model_name')[['finetuned_n_pure_clusters', 'finetuned_n_mixed_clusters']].mean()
x = np.arange(len(model_cluster_data))
width = 0.35
bars1 = ax.bar(x - width/2, model_cluster_data['finetuned_n_pure_clusters'], width, 
               label='Clusters Puros', color='#2ecc71')
bars2 = ax.bar(x + width/2, model_cluster_data['finetuned_n_mixed_clusters'], width,
               label='Clusters Mixtos', color='#e74c3c')
ax.set_xlabel('Modelo', fontweight='bold')
ax.set_ylabel('Número de Clusters')
ax.set_title('Por Modelo', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_cluster_data.index, rotation=0)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Por objetivo
ax = axes[1]
obj_cluster_data = df.groupby('objective')[['finetuned_n_pure_clusters', 'finetuned_n_mixed_clusters']].mean()
x = np.arange(len(obj_cluster_data))
bars1 = ax.bar(x - width/2, obj_cluster_data['finetuned_n_pure_clusters'], width,
               label='Clusters Puros', color='#3498db')
bars2 = ax.bar(x + width/2, obj_cluster_data['finetuned_n_mixed_clusters'], width,
               label='Clusters Mixtos', color='#f39c12')
ax.set_xlabel('Objetivo', fontweight='bold')
ax.set_ylabel('Número de Clusters')
ax.set_title('Por Objetivo', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(obj_cluster_data.index, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '07_cluster_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. CALIDAD DE CLUSTERING: % CLUSTERS PUROS
# ============================================================================
print("[8/12] Calidad de clustering...")

fig, ax = plt.subplots(figsize=(14, 8))

# Ordenar por % clusters puros
df_sorted = df.sort_values('finetuned_pure_percentage', ascending=True)

# Crear etiquetas para cada configuración
labels = df_sorted.apply(lambda x: f"{x['model_name'][:7]}\n{x['objective'][:6]}\n{x['criterion'] if pd.notna(x['criterion']) else 'class'}\n{x['views']}", axis=1)

# Colorear por modelo_name
colors = ['#e74c3c' if m == 'resnet50' else '#3498db' for m in df_sorted['model_name']]

bars = ax.barh(range(len(df_sorted)), df_sorted['finetuned_pure_percentage'], color=colors)
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('% Clusters Puros', fontweight='bold')
ax.set_title('Calidad de Clustering: % Clusters Puros por Configuración', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Añadir valores
for i, (v, ari) in enumerate(zip(df_sorted['finetuned_pure_percentage'], df_sorted['finetuned_ari'])):
    ax.text(v + 1, i, f'{v:.1f}% (ARI: {ari:.3f})', va='center', fontsize=7)

# Leyenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='ResNet50'),
                   Patch(facecolor='#3498db', label='ViT-B/32')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '08_clustering_quality_ranking.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 9. CLASES PROBLEMÁTICAS
# ============================================================================
print("[9/12] Análisis de clases problemáticas...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis de Clases Problemáticas (Overlap)', fontsize=16, fontweight='bold')

# Por modelo
ax = axes[0, 0]
model_prob = df.groupby('model_name')['finetuned_n_overlapping_classes'].mean()
bars = ax.bar(model_prob.index, model_prob.values, color=['#c0392b', '#8e44ad'])
ax.set_title('Clases en Múltiples Clusters - Por Modelo', fontweight='bold')
ax.set_ylabel('Número de Clases Problemáticas')
ax.set_xlabel('Modelo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom')

# Por objetivo
ax = axes[0, 1]
obj_prob = df.groupby('objective')['finetuned_n_overlapping_classes'].mean()
bars = ax.bar(obj_prob.index, obj_prob.values, color=['#d35400', '#16a085'])
ax.set_title('Clases Problemáticas - Por Objetivo', fontweight='bold')
ax.set_ylabel('Número de Clases Problemáticas')
ax.set_xlabel('Objetivo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom')

# Por criterio (metric learning)
ax = axes[1, 0]
crit_prob = df_ml.groupby('criterion')['finetuned_n_overlapping_classes'].mean().sort_values()
bars = ax.barh(range(len(crit_prob)), crit_prob.values, color=sns.color_palette("Reds_r", len(crit_prob)))
ax.set_yticks(range(len(crit_prob)))
ax.set_yticklabels(crit_prob.index)
ax.set_xlabel('Número de Clases Problemáticas')
ax.set_title('Clases Problemáticas - Por Criterio', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(crit_prob.values):
    ax.text(v + 1, i, f'{v:.1f}', va='center')

# Relación ARI vs Clases problemáticas
ax = axes[1, 1]
scatter = ax.scatter(df['finetuned_n_overlapping_classes'], df['finetuned_ari'], 
                     c=df['finetuned_pure_percentage'], s=100, cmap='RdYlGn', 
                     alpha=0.6, edgecolors='black')
ax.set_xlabel('Número de Clases Problemáticas', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('Correlación: Clases Problemáticas vs ARI', fontweight='bold')
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('% Clusters Puros')

# Calcular correlación
corr = df[['finetuned_n_overlapping_classes', 'finetuned_ari']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'Correlación: {corr:.3f}', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '09_problematic_classes.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 10. MEJORA BASELINE → FINETUNED
# ============================================================================
print("[10/12] Análisis de mejoras baseline → finetuned...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Mejora con Finetuning: Baseline → Finetuned', fontsize=16, fontweight='bold')

# Mejora absoluta por modelo
ax = axes[0, 0]
model_improvement = df.groupby('model_name')['ari_improvement'].mean()
bars = ax.bar(model_improvement.index, model_improvement.values, color=['#27ae60', '#2980b9'])
ax.set_title('Mejora de ARI - Por Modelo', fontweight='bold')
ax.set_ylabel('Mejora en ARI')
ax.set_xlabel('Modelo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top')

# Por objetivo
ax = axes[0, 1]
obj_improvement = df.groupby('objective')['ari_improvement'].mean()
bars = ax.bar(obj_improvement.index, obj_improvement.values, color=['#e67e22', '#16a085'])
ax.set_title('Mejora de ARI - Por Objetivo', fontweight='bold')
ax.set_ylabel('Mejora en ARI')
ax.set_xlabel('Objetivo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top')

# Comparación baseline vs finetuned
ax = axes[1, 0]
comparison_data = df[['baseline_ari', 'finetuned_ari']].mean()
bars = ax.bar(['Baseline', 'Finetuned'], comparison_data.values, color=['#95a5a6', '#2ecc71'])
ax.set_title('ARI: Baseline vs Finetuned (Promedio)', fontweight='bold')
ax.set_ylabel('ARI')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, comparison_data.values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Mejora por vista
ax = axes[1, 1]
views_improvement = df.groupby('views')['ari_improvement'].mean()
bars = ax.bar(views_improvement.index, views_improvement.values, color=['#f39c12', '#1abc9c'])
ax.set_title('Mejora de ARI - Por Vista', fontweight='bold')
ax.set_ylabel('Mejora en ARI')
ax.set_xlabel('Vista')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '10_finetuning_improvement.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 11. EFICIENCIA TEMPORAL
# ============================================================================
print("[11/12] Análisis de eficiencia temporal...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis de Eficiencia Temporal', fontsize=16, fontweight='bold')

# Tiempo por modelo
ax = axes[0, 0]
model_time = df.groupby('model_name')['duration_minutes'].mean()
bars = ax.bar(model_time.index, model_time.values, color=['#9b59b6', '#34495e'])
ax.set_title('Tiempo de Entrenamiento - Por Modelo', fontweight='bold')
ax.set_ylabel('Duración (minutos)')
ax.set_xlabel('Modelo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f} min', ha='center', va='bottom')

# Tiempo por objetivo
ax = axes[0, 1]
obj_time = df.groupby('objective')['duration_minutes'].mean()
bars = ax.bar(obj_time.index, obj_time.values, color=['#e67e22', '#1abc9c'])
ax.set_title('Tiempo de Entrenamiento - Por Objetivo', fontweight='bold')
ax.set_ylabel('Duración (minutos)')
ax.set_xlabel('Objetivo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f} min', ha='center', va='bottom')

# Eficiencia (ARI / tiempo)
ax = axes[1, 0]
df['efficiency'] = df['finetuned_ari'] / (df['duration_minutes'] / 60)  # ARI por hora
model_eff = df.groupby('model_name')['efficiency'].mean()
bars = ax.bar(model_eff.index, model_eff.values, color=['#c0392b', '#2980b9'])
ax.set_title('Eficiencia (ARI / Hora)', fontweight='bold')
ax.set_ylabel('ARI por Hora')
ax.set_xlabel('Modelo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom')

# Relación tiempo vs ARI
ax = axes[1, 1]
scatter = ax.scatter(df['duration_minutes'], df['finetuned_ari'], 
                     c=df['finetuned_pure_percentage'], s=100, cmap='viridis',
                     alpha=0.6, edgecolors='black')
ax.set_xlabel('Duración (minutos)', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('Relación Tiempo vs ARI', fontweight='bold')
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('% Clusters Puros')

# Calcular correlación
corr = df[['duration_minutes', 'finetuned_ari']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'Correlación: {corr:.3f}', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '11_temporal_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 12. RESUMEN COMPARATIVO TOP-5
# ============================================================================
print("[12/12] Resumen comparativo top-5...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Top-5 Configuraciones por Métrica', fontsize=16, fontweight='bold')

# Top-5 por ARI
ax = axes[0, 0]
top_ari = df.nlargest(5, 'finetuned_ari')
labels = top_ari.apply(lambda x: f"{x['model_name'][:7]}\n{x['criterion'] if pd.notna(x['criterion']) else 'class'}\n{x['views']}", axis=1)
bars = ax.barh(range(5), top_ari['finetuned_ari'].values, color=sns.color_palette("RdYlGn", 5)[::-1])
ax.set_yticks(range(5))
ax.set_yticklabels(labels)
ax.set_xlabel('ARI')
ax.set_title('Top-5 por ARI', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_ari['finetuned_ari'].values):
    ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

# Top-5 por Recall@1
ax = axes[0, 1]
top_recall1 = df.nlargest(5, 'best_val_recall@1')
labels = top_recall1.apply(lambda x: f"{x['model_name'][:7]}\n{x['criterion'] if pd.notna(x['criterion']) else 'class'}\n{x['views']}", axis=1)
bars = ax.barh(range(5), top_recall1['best_val_recall@1'].values, color=sns.color_palette("Blues", 5)[::-1])
ax.set_yticks(range(5))
ax.set_yticklabels(labels)
ax.set_xlabel('Recall@1 (%)')
ax.set_title('Top-5 por Recall@1', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_recall1['best_val_recall@1'].values):
    ax.text(v + 0.5, i, f'{v:.2f}%', va='center', fontweight='bold')

# Top-5 por % Clusters Puros
ax = axes[1, 0]
top_pure = df.nlargest(5, 'finetuned_pure_percentage')
labels = top_pure.apply(lambda x: f"{x['model_name'][:7]}\n{x['criterion'] if pd.notna(x['criterion']) else 'class'}\n{x['views']}", axis=1)
bars = ax.barh(range(5), top_pure['finetuned_pure_percentage'].values, color=sns.color_palette("Greens", 5)[::-1])
ax.set_yticks(range(5))
ax.set_yticklabels(labels)
ax.set_xlabel('% Clusters Puros')
ax.set_title('Top-5 por % Clusters Puros', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_pure['finetuned_pure_percentage'].values):
    ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

# Top-5 por Eficiencia
ax = axes[1, 1]
top_eff = df.nlargest(5, 'efficiency')
labels = top_eff.apply(lambda x: f"{x['model_name'][:7]}\n{x['criterion'] if pd.notna(x['criterion']) else 'class'}\n{x['views']}", axis=1)
bars = ax.barh(range(5), top_eff['efficiency'].values, color=sns.color_palette("Oranges", 5)[::-1])
ax.set_yticks(range(5))
ax.set_yticklabels(labels)
ax.set_xlabel('ARI / Hora')
ax.set_title('Top-5 por Eficiencia', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_eff['efficiency'].values):
    ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '12_top5_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIN
# ============================================================================
print()
print("="*80)
print("VISUALIZACIONES COMPLETADAS")
print("="*80)
print(f"📊 12 gráficos guardados en: {PLOTS_DIR}")
print()
print("Archivos generados:")
for i in range(1, 13):
    png_file = list(PLOTS_DIR.glob(f'{i:02d}_*.png'))[0]
    print(f"  ✓ {png_file.name}")
print()


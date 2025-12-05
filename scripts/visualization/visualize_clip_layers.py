#!/usr/bin/env python3
"""
Script de visualización para análisis de CLIP Layers Study
Genera gráficos de progresión por capas y comparación vision vs text
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
PLOTS_DIR = Path('results/analysis/plots/clip_layers')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Cargar datos
print("="*80)
print("GENERANDO VISUALIZACIONES - CLIP LAYERS STUDY")
print("="*80)
print()

df = pd.read_csv(RESULTS_DIR / 'clip_layers_results.csv')
print(f"Datos cargados: {len(df)} configuraciones\n")

# ============================================================================
# 1. COMPARACIÓN VISION vs TEXT COMPONENTS
# ============================================================================
print("[1/10] Comparación vision vs text components...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CLIP: Comparación Vision vs Text Components', fontsize=16, fontweight='bold')

# Recall promedio
comp_stats = df.groupby('component_type')[['best_val_recall@1', 'best_val_recall@3', 'best_val_recall@5']].mean()
ax = axes[0, 0]
comp_stats.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db', '#2ecc71'])
ax.set_title('Recall Promedio por Componente', fontweight='bold')
ax.set_ylabel('Recall (%)')
ax.set_xlabel('Componente')
ax.legend(['Recall@1', 'Recall@3', 'Recall@5'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)

# ARI
ax = axes[0, 1]
ari_comp = df.groupby('component_type')[['finetuned_ari']].mean()
bars = ax.bar(ari_comp.index, ari_comp['finetuned_ari'], color=['#9b59b6', '#f39c12'])
ax.set_title('ARI Promedio por Componente', fontweight='bold')
ax.set_ylabel('ARI')
ax.set_xlabel('Componente')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

# Clusters puros
ax = axes[1, 0]
pure_comp = df.groupby('component_type')[['finetuned_pure_percentage']].mean()
bars = ax.bar(pure_comp.index, pure_comp['finetuned_pure_percentage'], color=['#16a085', '#e67e22'])
ax.set_title('% Clusters Puros por Componente', fontweight='bold')
ax.set_ylabel('% Clusters Puros')
ax.set_xlabel('Componente')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# Diferencia porcentual
ax = axes[1, 1]
if 'vision' in ari_comp.index and 'text' in ari_comp.index:
    improvement = ((ari_comp.loc['vision', 'finetuned_ari'] - ari_comp.loc['text', 'finetuned_ari']) / 
                   ari_comp.loc['text', 'finetuned_ari'] * 100)
    bars = ax.bar(['Text', 'Vision'], 
                   [ari_comp.loc['text', 'finetuned_ari'], ari_comp.loc['vision', 'finetuned_ari']],
                   color=['#e67e22', '#27ae60'])
    ax.set_title(f'ARI: Vision es {improvement:.1f}% mejor que Text', fontweight='bold')
    ax.set_ylabel('ARI')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_component_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. PROGRESIÓN POR NÚMERO DE CAPAS - RECALL
# ============================================================================
print("[2/10] Progresión recall por número de capas...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Progresión de Recall por Número de Capas', fontsize=16, fontweight='bold')

# Vision component
ax = axes[0]
df_vision = df[df['component_type'] == 'vision'].sort_values('num_layers')
ax.plot(df_vision['num_layers'], df_vision['best_val_recall@1'], 
        marker='o', linewidth=2, markersize=8, label='Recall@1', color='#e74c3c')
ax.plot(df_vision['num_layers'], df_vision['best_val_recall@3'], 
        marker='s', linewidth=2, markersize=8, label='Recall@3', color='#3498db')
ax.plot(df_vision['num_layers'], df_vision['best_val_recall@5'], 
        marker='^', linewidth=2, markersize=8, label='Recall@5', color='#2ecc71')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('Recall (%)', fontweight='bold')
ax.set_title('Vision Component', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Text component
ax = axes[1]
df_text = df[df['component_type'] == 'text'].sort_values('num_layers')
ax.plot(df_text['num_layers'], df_text['best_val_recall@1'], 
        marker='o', linewidth=2, markersize=8, label='Recall@1', color='#e74c3c')
ax.plot(df_text['num_layers'], df_text['best_val_recall@3'], 
        marker='s', linewidth=2, markersize=8, label='Recall@3', color='#3498db')
ax.plot(df_text['num_layers'], df_text['best_val_recall@5'], 
        marker='^', linewidth=2, markersize=8, label='Recall@5', color='#2ecc71')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('Recall (%)', fontweight='bold')
ax.set_title('Text Component', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_recall_progression.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. PROGRESIÓN POR NÚMERO DE CAPAS - ARI
# ============================================================================
print("[3/10] Progresión ARI por número de capas...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Progresión de ARI por Número de Capas', fontsize=16, fontweight='bold')

# Vision component
ax = axes[0]
ax.plot(df_vision['num_layers'], df_vision['finetuned_ari'], 
        marker='o', linewidth=3, markersize=10, color='#9b59b6', label='ARI')
ax.fill_between(df_vision['num_layers'], df_vision['finetuned_ari'], alpha=0.3, color='#9b59b6')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('Vision Component', fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Marcar el máximo
max_idx = df_vision['finetuned_ari'].idxmax()
max_layers = df_vision.loc[max_idx, 'num_layers']
max_ari = df_vision.loc[max_idx, 'finetuned_ari']
ax.scatter([max_layers], [max_ari], color='red', s=200, zorder=5, marker='*', 
           edgecolors='black', linewidths=2)
ax.annotate(f'Máximo: {max_ari:.4f}\n({int(max_layers)} capas)', 
            xy=(max_layers, max_ari), xytext=(max_layers+0.5, max_ari-0.02),
            fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Text component
ax = axes[1]
ax.plot(df_text['num_layers'], df_text['finetuned_ari'], 
        marker='o', linewidth=3, markersize=10, color='#f39c12', label='ARI')
ax.fill_between(df_text['num_layers'], df_text['finetuned_ari'], alpha=0.3, color='#f39c12')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('Text Component', fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Marcar el máximo
max_idx = df_text['finetuned_ari'].idxmax()
max_layers = df_text.loc[max_idx, 'num_layers']
max_ari = df_text.loc[max_idx, 'finetuned_ari']
ax.scatter([max_layers], [max_ari], color='red', s=200, zorder=5, marker='*',
           edgecolors='black', linewidths=2)
ax.annotate(f'Máximo: {max_ari:.4f}\n({int(max_layers)} capas)', 
            xy=(max_layers, max_ari), xytext=(max_layers+0.5, max_ari+0.005),
            fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_ari_progression.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. COMPARACIÓN VISION vs TEXT POR CAPAS
# ============================================================================
print("[4/10] Comparación vision vs text por capas...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Vision vs Text: Comparación por Número de Capas', fontsize=16, fontweight='bold')

# Recall@1
ax = axes[0, 0]
ax.plot(df_vision['num_layers'], df_vision['best_val_recall@1'], 
        marker='o', linewidth=2.5, markersize=8, label='Vision', color='#e74c3c')
ax.plot(df_text['num_layers'], df_text['best_val_recall@1'], 
        marker='s', linewidth=2.5, markersize=8, label='Text', color='#3498db')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('Recall@1 (%)', fontweight='bold')
ax.set_title('Recall@1 por Capas', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# ARI
ax = axes[0, 1]
ax.plot(df_vision['num_layers'], df_vision['finetuned_ari'], 
        marker='o', linewidth=2.5, markersize=8, label='Vision', color='#9b59b6')
ax.plot(df_text['num_layers'], df_text['finetuned_ari'], 
        marker='s', linewidth=2.5, markersize=8, label='Text', color='#f39c12')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('ARI por Capas', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# % Clusters puros
ax = axes[1, 0]
ax.plot(df_vision['num_layers'], df_vision['finetuned_pure_percentage'], 
        marker='o', linewidth=2.5, markersize=8, label='Vision', color='#16a085')
ax.plot(df_text['num_layers'], df_text['finetuned_pure_percentage'], 
        marker='s', linewidth=2.5, markersize=8, label='Text', color='#e67e22')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('% Clusters Puros', fontweight='bold')
ax.set_title('% Clusters Puros por Capas', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Diferencia ARI (Vision - Text)
ax = axes[1, 1]
diff_ari = df_vision.set_index('num_layers')['finetuned_ari'] - df_text.set_index('num_layers')['finetuned_ari']
colors = ['#27ae60' if d > 0 else '#c0392b' for d in diff_ari.values]
bars = ax.bar(diff_ari.index, diff_ari.values, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('Diferencia ARI (Vision - Text)', fontweight='bold')
ax.set_title('Ventaja de Vision sobre Text', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_vision_vs_text_by_layers.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. CORRELACIÓN CAPAS-MÉTRICAS
# ============================================================================
print("[5/10] Análisis de correlación capas-métricas...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Correlación: Número de Capas vs Métricas', fontsize=16, fontweight='bold')

# Vision
ax = axes[0]
corr_vision_recall = df_vision[['num_layers', 'best_val_recall@1']].corr().iloc[0, 1]
corr_vision_ari = df_vision[['num_layers', 'finetuned_ari']].corr().iloc[0, 1]

metrics = ['Recall@1', 'ARI']
correlations = [corr_vision_recall, corr_vision_ari]
colors = ['#2ecc71' if c > 0.5 else '#f39c12' if c > 0 else '#e74c3c' for c in correlations]

bars = ax.barh(metrics, correlations, color=colors, edgecolor='black', linewidth=2)
ax.set_xlabel('Coeficiente de Correlación', fontweight='bold')
ax.set_title('Vision Component', fontweight='bold')
ax.set_xlim([-1, 1])
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.3)

for i, (metric, corr) in enumerate(zip(metrics, correlations)):
    ax.text(corr + 0.05 if corr > 0 else corr - 0.05, i, 
            f'{corr:.3f}', va='center', ha='left' if corr > 0 else 'right',
            fontweight='bold', fontsize=12)

# Text
ax = axes[1]
corr_text_recall = df_text[['num_layers', 'best_val_recall@1']].corr().iloc[0, 1]
corr_text_ari = df_text[['num_layers', 'finetuned_ari']].corr().iloc[0, 1]

correlations = [corr_text_recall, corr_text_ari]
colors = ['#2ecc71' if c > 0.5 else '#f39c12' if c > 0 else '#e74c3c' for c in correlations]

bars = ax.barh(metrics, correlations, color=colors, edgecolor='black', linewidth=2)
ax.set_xlabel('Coeficiente de Correlación', fontweight='bold')
ax.set_title('Text Component', fontweight='bold')
ax.set_xlim([-1, 1])
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.3)

for i, (metric, corr) in enumerate(zip(metrics, correlations)):
    ax.text(corr + 0.05 if corr > 0 else corr - 0.05, i, 
            f'{corr:.3f}', va='center', ha='left' if corr > 0 else 'right',
            fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_layer_metric_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. HEATMAP COMPONENTE × CAPAS × MÉTRICAS
# ============================================================================
print("[6/10] Heatmap componente × capas...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Heatmap: Componente × Número de Capas', fontsize=16, fontweight='bold')

# Recall@1
pivot_recall = df.pivot_table(values='best_val_recall@1', 
                                index='num_layers', 
                                columns='component_type')
ax = axes[0]
sns.heatmap(pivot_recall, annot=True, fmt='.2f', cmap='YlOrRd', 
            cbar_kws={'label': 'Recall@1 (%)'}, ax=ax)
ax.set_title('Recall@1', fontweight='bold')
ax.set_xlabel('Componente', fontweight='bold')
ax.set_ylabel('Número de Capas', fontweight='bold')

# ARI
pivot_ari = df.pivot_table(values='finetuned_ari', 
                            index='num_layers', 
                            columns='component_type')
ax = axes[1]
sns.heatmap(pivot_ari, annot=True, fmt='.4f', cmap='RdYlGn', 
            cbar_kws={'label': 'ARI'}, ax=ax, vmin=0, vmax=0.5)
ax.set_title('ARI', fontweight='bold')
ax.set_xlabel('Componente', fontweight='bold')
ax.set_ylabel('Número de Capas', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '06_heatmap_layers.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. CALIDAD DE CLUSTERING POR CAPAS
# ============================================================================
print("[7/10] Calidad de clustering por capas...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Calidad de Clustering por Número de Capas', fontsize=16, fontweight='bold')

# Distribución clusters puros/mixtos - Vision
ax = axes[0, 0]
x = df_vision['num_layers']
width = 0.35
ax.bar(x - width/2, df_vision['finetuned_n_pure_clusters'], width,
       label='Clusters Puros', color='#2ecc71', edgecolor='black')
ax.bar(x + width/2, df_vision['finetuned_n_mixed_clusters'], width,
       label='Clusters Mixtos', color='#e74c3c', edgecolor='black')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('Número de Clusters')
ax.set_title('Vision Component: Clusters Puros vs Mixtos', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(range(1, 13))

# Distribución clusters puros/mixtos - Text
ax = axes[0, 1]
x = df_text['num_layers']
ax.bar(x - width/2, df_text['finetuned_n_pure_clusters'], width,
       label='Clusters Puros', color='#3498db', edgecolor='black')
ax.bar(x + width/2, df_text['finetuned_n_mixed_clusters'], width,
       label='Clusters Mixtos', color='#f39c12', edgecolor='black')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('Número de Clusters')
ax.set_title('Text Component: Clusters Puros vs Mixtos', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(range(1, 13))

# Purity por capas
ax = axes[1, 0]
ax.plot(df_vision['num_layers'], df_vision['finetuned_purity'], 
        marker='o', linewidth=2.5, markersize=8, label='Vision', color='#9b59b6')
ax.plot(df_text['num_layers'], df_text['finetuned_purity'], 
        marker='s', linewidth=2.5, markersize=8, label='Text', color='#e67e22')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('Purity', fontweight='bold')
ax.set_title('Purity Score por Capas', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Clases problemáticas
ax = axes[1, 1]
ax.plot(df_vision['num_layers'], df_vision['finetuned_n_overlapping_classes'], 
        marker='o', linewidth=2.5, markersize=8, label='Vision', color='#c0392b')
ax.plot(df_text['num_layers'], df_text['finetuned_n_overlapping_classes'], 
        marker='s', linewidth=2.5, markersize=8, label='Text', color='#d35400')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('Clases Problemáticas', fontweight='bold')
ax.set_title('Clases en Múltiples Clusters', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '07_clustering_quality_by_layers.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. EFICIENCIA TEMPORAL
# ============================================================================
print("[8/10] Análisis de eficiencia temporal...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Análisis de Eficiencia Temporal - CLIP', fontsize=16, fontweight='bold')

# Tiempo por componente
ax = axes[0, 0]
comp_time = df.groupby('component_type')['duration_minutes'].mean()
bars = ax.bar(comp_time.index, comp_time.values, color=['#9b59b6', '#f39c12'], edgecolor='black')
ax.set_title('Tiempo de Entrenamiento por Componente', fontweight='bold')
ax.set_ylabel('Duración (minutos)')
ax.set_xlabel('Componente')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f} min', ha='center', va='bottom', fontweight='bold')

# Tiempo por número de capas
ax = axes[0, 1]
layers_time = df.groupby('num_layers')['duration_minutes'].mean()
ax.plot(layers_time.index, layers_time.values, marker='o', linewidth=2.5, 
        markersize=8, color='#e74c3c')
ax.set_xlabel('Número de Capas', fontweight='bold')
ax.set_ylabel('Duración (minutos)', fontweight='bold')
ax.set_title('Tiempo por Número de Capas', fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Eficiencia (ARI / hora)
ax = axes[1, 0]
df['efficiency'] = df['finetuned_ari'] / (df['duration_minutes'] / 60)
comp_eff = df.groupby('component_type')['efficiency'].mean()
bars = ax.bar(comp_eff.index, comp_eff.values, color=['#27ae60', '#16a085'], edgecolor='black')
ax.set_title('Eficiencia (ARI / Hora)', fontweight='bold')
ax.set_ylabel('ARI por Hora')
ax.set_xlabel('Componente')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

# Relación tiempo vs ARI
ax = axes[1, 1]
scatter = ax.scatter(df[df['component_type']=='vision']['duration_minutes'], 
                     df[df['component_type']=='vision']['finetuned_ari'],
                     s=100, c=df[df['component_type']=='vision']['num_layers'], 
                     cmap='viridis', alpha=0.7, edgecolors='black', label='Vision')
scatter2 = ax.scatter(df[df['component_type']=='text']['duration_minutes'], 
                      df[df['component_type']=='text']['finetuned_ari'],
                      s=100, c=df[df['component_type']=='text']['num_layers'], 
                      cmap='plasma', alpha=0.7, edgecolors='black', marker='s', label='Text')
ax.set_xlabel('Duración (minutos)', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('Relación Tiempo vs ARI', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Número de Capas')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '08_temporal_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 9. RANKING MEJORES CONFIGURACIONES
# ============================================================================
print("[9/10] Ranking mejores configuraciones...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Ranking de Configuraciones - CLIP Layers', fontsize=16, fontweight='bold')

# Top-10 por ARI
ax = axes[0, 0]
top_ari = df.nlargest(10, 'finetuned_ari')
labels = top_ari.apply(lambda x: f"{x['component_type'][:4]} - {int(x['num_layers'])} capas", axis=1)
colors = ['#9b59b6' if ct == 'vision' else '#f39c12' for ct in top_ari['component_type']]
bars = ax.barh(range(10), top_ari['finetuned_ari'].values, color=colors, edgecolor='black')
ax.set_yticks(range(10))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('ARI')
ax.set_title('Top-10 por ARI', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_ari['finetuned_ari'].values):
    ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)

# Top-10 por Recall@1
ax = axes[0, 1]
top_recall = df.nlargest(10, 'best_val_recall@1')
labels = top_recall.apply(lambda x: f"{x['component_type'][:4]} - {int(x['num_layers'])} capas", axis=1)
colors = ['#9b59b6' if ct == 'vision' else '#f39c12' for ct in top_recall['component_type']]
bars = ax.barh(range(10), top_recall['best_val_recall@1'].values, color=colors, edgecolor='black')
ax.set_yticks(range(10))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Recall@1 (%)')
ax.set_title('Top-10 por Recall@1', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_recall['best_val_recall@1'].values):
    ax.text(v + 0.2, i, f'{v:.2f}', va='center', fontsize=8)

# Top-10 por % Clusters Puros
ax = axes[1, 0]
top_pure = df.nlargest(10, 'finetuned_pure_percentage')
labels = top_pure.apply(lambda x: f"{x['component_type'][:4]} - {int(x['num_layers'])} capas", axis=1)
colors = ['#9b59b6' if ct == 'vision' else '#f39c12' for ct in top_pure['component_type']]
bars = ax.barh(range(10), top_pure['finetuned_pure_percentage'].values, color=colors, edgecolor='black')
ax.set_yticks(range(10))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('% Clusters Puros')
ax.set_title('Top-10 por % Clusters Puros', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_pure['finetuned_pure_percentage'].values):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)

# Top-10 por Eficiencia
ax = axes[1, 1]
top_eff = df.nlargest(10, 'efficiency')
labels = top_eff.apply(lambda x: f"{x['component_type'][:4]} - {int(x['num_layers'])} capas", axis=1)
colors = ['#9b59b6' if ct == 'vision' else '#f39c12' for ct in top_eff['component_type']]
bars = ax.barh(range(10), top_eff['efficiency'].values, color=colors, edgecolor='black')
ax.set_yticks(range(10))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('ARI / Hora')
ax.set_title('Top-10 por Eficiencia', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_eff['efficiency'].values):
    ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '09_configuration_ranking.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 10. RESUMEN COMPARATIVO COMPLETO
# ============================================================================
print("[10/10] Resumen comparativo completo...")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Resumen Completo: CLIP Layers Study', fontsize=18, fontweight='bold')

# 1. Mejor por componente
ax = fig.add_subplot(gs[0, 0])
best_by_comp = df.loc[df.groupby('component_type')['finetuned_ari'].idxmax()]
bars = ax.bar(best_by_comp['component_type'], best_by_comp['finetuned_ari'], 
              color=['#9b59b6', '#f39c12'], edgecolor='black', linewidth=2)
ax.set_title('Mejor ARI por Componente', fontweight='bold', fontsize=11)
ax.set_ylabel('ARI', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, layers in zip(bars, best_by_comp['num_layers']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}\n({int(layers)} capas)', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Recall promedio
ax = fig.add_subplot(gs[0, 1])
recall_avg = df.groupby('component_type')[['best_val_recall@1', 'best_val_recall@5']].mean()
x = np.arange(len(recall_avg))
width = 0.35
ax.bar(x - width/2, recall_avg['best_val_recall@1'], width, label='Recall@1', color='#e74c3c', edgecolor='black')
ax.bar(x + width/2, recall_avg['best_val_recall@5'], width, label='Recall@5', color='#2ecc71', edgecolor='black')
ax.set_title('Recall Promedio', fontweight='bold', fontsize=11)
ax.set_ylabel('Recall (%)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(recall_avg.index)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 3. Clusters promedio
ax = fig.add_subplot(gs[0, 2])
cluster_avg = df.groupby('component_type')[['finetuned_n_pure_clusters', 'finetuned_n_mixed_clusters']].mean()
x = np.arange(len(cluster_avg))
ax.bar(x - width/2, cluster_avg['finetuned_n_pure_clusters'], width, 
       label='Puros', color='#27ae60', edgecolor='black')
ax.bar(x + width/2, cluster_avg['finetuned_n_mixed_clusters'], width, 
       label='Mixtos', color='#c0392b', edgecolor='black')
ax.set_title('Clusters Promedio', fontweight='bold', fontsize=11)
ax.set_ylabel('Número de Clusters', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cluster_avg.index)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 4-6. Progresión por capas (recall, ari, pure%)
ax = fig.add_subplot(gs[1, :])
ax2 = ax.twinx()
ax3 = ax.twinx()
ax3.spines['right'].set_position(('outward', 60))

# Recall@1
line1 = ax.plot(df_vision['num_layers'], df_vision['best_val_recall@1'], 
                marker='o', linewidth=2, markersize=6, label='Vision Recall@1', color='#e74c3c')
line2 = ax.plot(df_text['num_layers'], df_text['best_val_recall@1'], 
                marker='s', linewidth=2, markersize=6, label='Text Recall@1', color='#3498db')

# ARI
line3 = ax2.plot(df_vision['num_layers'], df_vision['finetuned_ari'], 
                 marker='D', linewidth=2, markersize=6, label='Vision ARI', color='#9b59b6', linestyle='--')
line4 = ax2.plot(df_text['num_layers'], df_text['finetuned_ari'], 
                 marker='^', linewidth=2, markersize=6, label='Text ARI', color='#f39c12', linestyle='--')

# Pure %
line5 = ax3.plot(df_vision['num_layers'], df_vision['finetuned_pure_percentage'], 
                 marker='*', linewidth=2, markersize=8, label='Vision Pure%', color='#27ae60', linestyle=':')
line6 = ax3.plot(df_text['num_layers'], df_text['finetuned_pure_percentage'], 
                 marker='P', linewidth=2, markersize=6, label='Text Pure%', color='#16a085', linestyle=':')

ax.set_xlabel('Número de Capas', fontweight='bold', fontsize=11)
ax.set_ylabel('Recall@1 (%)', fontweight='bold', fontsize=10, color='#e74c3c')
ax2.set_ylabel('ARI', fontweight='bold', fontsize=10, color='#9b59b6')
ax3.set_ylabel('% Clusters Puros', fontweight='bold', fontsize=10, color='#27ae60')
ax.set_title('Progresión Completa por Número de Capas', fontweight='bold', fontsize=12, pad=20)
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Combinar leyendas
lines = line1 + line2 + line3 + line4 + line5 + line6
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, -0.1), ncol=3, fontsize=9)

# 7. Estadísticas finales
ax = fig.add_subplot(gs[2, :])
ax.axis('off')

stats_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                                    ESTADÍSTICAS CLAVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  🏆 MEJOR CONFIGURACIÓN GLOBAL
     • Componente: Vision - 11 capas
     • ARI: {df_vision.loc[df_vision['finetuned_ari'].idxmax(), 'finetuned_ari']:.4f}
     • Recall@1: {df_vision.loc[df_vision['finetuned_ari'].idxmax(), 'best_val_recall@1']:.2f}%
     • % Clusters Puros: {df_vision.loc[df_vision['finetuned_ari'].idxmax(), 'finetuned_pure_percentage']:.2f}%

  📊 COMPARACIÓN VISION vs TEXT
     • Vision ARI medio: {df_vision['finetuned_ari'].mean():.4f}  |  Text ARI medio: {df_text['finetuned_ari'].mean():.4f}  →  Vision {((df_vision['finetuned_ari'].mean() / df_text['finetuned_ari'].mean() - 1) * 100):.1f}% mejor
     • Vision Recall@1: {df_vision['best_val_recall@1'].mean():.2f}%  |  Text Recall@1: {df_text['best_val_recall@1'].mean():.2f}%  →  Diferencia: {df_vision['best_val_recall@1'].mean() - df_text['best_val_recall@1'].mean():.2f}%
     • Vision Pure%: {df_vision['finetuned_pure_percentage'].mean():.2f}%  |  Text Pure%: {df_text['finetuned_pure_percentage'].mean():.2f}%

  🔬 CORRELACIONES
     • Vision: Capas ↔ Recall@1 = {corr_vision_recall:.3f}  |  Capas ↔ ARI = {corr_vision_ari:.3f}
     • Text:   Capas ↔ Recall@1 = {corr_text_recall:.3f}  |  Capas ↔ ARI = {corr_text_ari:.3f}

  ⏱️  EFICIENCIA
     • Vision: {(df_vision['finetuned_ari'] / (df_vision['duration_minutes'] / 60)).mean():.4f} ARI/hora  |  Text: {(df_text['finetuned_ari'] / (df_text['duration_minutes'] / 60)).mean():.4f} ARI/hora
     • Tiempo medio: Vision {df_vision['duration_minutes'].mean():.1f} min  |  Text {df_text['duration_minutes'].mean():.1f} min

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10, 
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

plt.savefig(PLOTS_DIR / '10_complete_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIN
# ============================================================================
print()
print("="*80)
print("VISUALIZACIONES COMPLETADAS")
print("="*80)
print(f"📊 10 gráficos guardados en: {PLOTS_DIR}")
print()
print("Archivos generados:")
for i in range(1, 11):
    png_file = list(PLOTS_DIR.glob(f'{i:02d}_*.png'))[0]
    print(f"  ✓ {png_file.name}")
print()

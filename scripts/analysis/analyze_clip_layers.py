"""
Script para análisis estadístico del estudio de capas de CLIP.
Analiza el impacto del número de capas (1-12) y componentes (vision vs text).
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


def load_data(csv_path: Path) -> pd.DataFrame:
    """Carga el CSV de resultados de CLIP layers."""
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def analyze_by_component(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis comparativo entre componentes vision y text."""
    print("\n" + "=" * 80)
    print("ANÁLISIS POR COMPONENTE (Vision vs Text)")
    print("=" * 80)
    
    grouped = df.groupby('component_type').agg({
        'config_key': 'count',
        
        # Métricas de recall
        'best_val_recall@1': ['mean', 'std', 'min', 'max'],
        'best_val_recall@3': ['mean', 'std', 'min', 'max'],
        'best_val_recall@5': ['mean', 'std', 'min', 'max'],
        
        # Clustering
        'finetuned_ari': ['mean', 'std', 'min', 'max'],
        'finetuned_nmi': ['mean', 'std', 'min', 'max'],
        'finetuned_purity': ['mean', 'std', 'min', 'max'],
        'ari_improvement': ['mean', 'std'],
        
        # Calidad de clusters
        'finetuned_pure_percentage': ['mean', 'std', 'min', 'max'],
        'finetuned_n_pure_clusters': ['mean', 'std'],
        'finetuned_n_mixed_clusters': ['mean', 'std'],
        
        # Tiempos
        'duration_minutes': ['mean', 'std'],
        'embeddings_duration_minutes': ['mean', 'std'],
    }).round(4)
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})
    
    print("\n", grouped.to_string())
    
    # Comparación directa
    print("\n" + "-" * 80)
    print("COMPARACIÓN VISION vs TEXT:")
    print("-" * 80)
    
    if 'vision' in grouped.index and 'text' in grouped.index:
        vision = grouped.loc['vision']
        text = grouped.loc['text']
        
        print(f"\nRecall@1 medio:")
        print(f"  Vision: {vision['best_val_recall@1_mean']:.4f}")
        print(f"  Text:   {text['best_val_recall@1_mean']:.4f}")
        print(f"  Diferencia: {(vision['best_val_recall@1_mean'] - text['best_val_recall@1_mean']):.4f}")
        
        print(f"\nARI medio:")
        print(f"  Vision: {vision['finetuned_ari_mean']:.4f}")
        print(f"  Text:   {text['finetuned_ari_mean']:.4f}")
        print(f"  Diferencia: {(vision['finetuned_ari_mean'] - text['finetuned_ari_mean']):.4f}")
        
        print(f"\nPure % medio:")
        print(f"  Vision: {vision['finetuned_pure_percentage_mean']:.2f}%")
        print(f"  Text:   {text['finetuned_pure_percentage_mean']:.2f}%")
        print(f"  Diferencia: {(vision['finetuned_pure_percentage_mean'] - text['finetuned_pure_percentage_mean']):.2f}%")
    
    return grouped


def analyze_by_num_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis del impacto del número de capas (1-12)."""
    print("\n" + "=" * 80)
    print("ANÁLISIS POR NÚMERO DE CAPAS")
    print("=" * 80)
    
    grouped = df.groupby('num_layers').agg({
        'config_key': 'count',
        'component_type': lambda x: ', '.join(x.unique()),
        
        # Métricas
        'best_val_recall@1': ['mean', 'std'],
        'best_val_recall@3': ['mean', 'std'],
        'best_val_recall@5': ['mean', 'std'],
        'finetuned_ari': ['mean', 'std'],
        'finetuned_pure_percentage': ['mean', 'std'],
        
        # Tiempos
        'duration_minutes': ['mean', 'std'],
    }).round(4)
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})
    
    print("\n", grouped.to_string())
    
    # Mejor número de capas por métrica
    print("\n" + "-" * 80)
    print("MEJOR NÚMERO DE CAPAS POR MÉTRICA:")
    print("-" * 80)
    
    metrics = {
        'Recall@1': 'best_val_recall@1_mean',
        'Recall@5': 'best_val_recall@5_mean',
    }
    
    for metric_name, col in metrics.items():
        if col in grouped.columns:
            best_layers = grouped[col].idxmax()
            if pd.notna(best_layers):
                value = grouped.loc[best_layers, col]
                print(f"  {metric_name:12s}: {best_layers} capas ({value:.4f})")
    
    return grouped


def analyze_component_x_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis cruzado: componente × número de capas."""
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPONENTE × CAPAS")
    print("=" * 80)
    
    # Vision
    print("\n--- COMPONENTE VISION ---")
    df_vision = df[df['component_type'] == 'vision']
    
    if not df_vision.empty:
        vision_grouped = df_vision.groupby('num_layers').agg({
            'best_val_recall@1': ['mean', 'max'],
            'finetuned_ari': ['mean', 'max'],
            'finetuned_pure_percentage': ['mean', 'max'],
            'duration_minutes': 'mean',
        }).round(4)
        
        vision_grouped.columns = ['_'.join(col).strip() for col in vision_grouped.columns.values]
        print("\n", vision_grouped.to_string())
        
        # Mejor configuración vision
        best_vision_idx = df_vision['finetuned_ari'].idxmax()
        if pd.notna(best_vision_idx):
            best = df_vision.loc[best_vision_idx]
            print(f"\nMejor configuración (ARI):")
            print(f"  Capas:    {best['num_layers']}")
            print(f"  Recall@1: {best['best_val_recall@1']:.4f}")
            print(f"  ARI:      {best['finetuned_ari']:.4f}")
            print(f"  Pure %:   {best['finetuned_pure_percentage']:.2f}%")
    
    # Text
    print("\n--- COMPONENTE TEXT ---")
    df_text = df[df['component_type'] == 'text']
    
    if not df_text.empty:
        text_grouped = df_text.groupby('num_layers').agg({
            'best_val_recall@1': ['mean', 'max'],
            'finetuned_ari': ['mean', 'max'],
            'finetuned_pure_percentage': ['mean', 'max'],
            'duration_minutes': 'mean',
        }).round(4)
        
        text_grouped.columns = ['_'.join(col).strip() for col in text_grouped.columns.values]
        print("\n", text_grouped.to_string())
        
        # Mejor configuración text
        best_text_idx = df_text['finetuned_ari'].idxmax()
        if pd.notna(best_text_idx):
            best = df_text.loc[best_text_idx]
            print(f"\nMejor configuración (ARI):")
            print(f"  Capas:    {best['num_layers']}")
            print(f"  Recall@1: {best['best_val_recall@1']:.4f}")
            print(f"  ARI:      {best['finetuned_ari']:.4f}")
            print(f"  Pure %:   {best['finetuned_pure_percentage']:.2f}%")
    
    return df_vision, df_text


def analyze_progression_by_layers(df: pd.DataFrame):
    """Analiza la progresión de métricas según aumentan las capas."""
    print("\n" + "=" * 80)
    print("ANÁLISIS DE PROGRESIÓN POR CAPAS")
    print("=" * 80)
    
    for component in ['vision', 'text']:
        df_comp = df[df['component_type'] == component]
        
        if df_comp.empty:
            continue
        
        print(f"\n--- Progresión {component.upper()} ---")
        
        # Ordenar por capas
        progression = df_comp.sort_values('num_layers')[
            ['num_layers', 'best_val_recall@1', 'finetuned_ari', 'finetuned_pure_percentage']
        ]
        
        print("\n", progression.to_string(index=False))
        
        # Calcular correlación
        corr_recall = df_comp[['num_layers', 'best_val_recall@1']].corr().iloc[0, 1]
        corr_ari = df_comp[['num_layers', 'finetuned_ari']].corr().iloc[0, 1]
        
        print(f"\nCorrelación capas-Recall@1: {corr_recall:.4f}")
        print(f"Correlación capas-ARI:      {corr_ari:.4f}")
        
        # Encontrar punto óptimo (donde empieza a saturar)
        recall_values = df_comp.sort_values('num_layers')['best_val_recall@1'].values
        if len(recall_values) > 1:
            diffs = np.diff(recall_values)
            if len(diffs) > 0:
                avg_diff = np.mean(diffs[diffs > 0]) if len(diffs[diffs > 0]) > 0 else 0
                print(f"Mejora promedio por capa:   {avg_diff:.4f}")


def analyze_clustering_quality(df: pd.DataFrame):
    """Análisis de calidad de clustering para CLIP."""
    print("\n" + "=" * 80)
    print("ANÁLISIS DE CALIDAD DE CLUSTERING - CLIP")
    print("=" * 80)
    
    print("\n--- Estadísticas Generales ---")
    
    stats = {
        'ARI medio': df['finetuned_ari'].mean(),
        'ARI máximo': df['finetuned_ari'].max(),
        'Purity media': df['finetuned_purity'].mean(),
        '% Clusters puros medio': df['finetuned_pure_percentage'].mean(),
        'Nº clusters puros medio': df['finetuned_n_pure_clusters'].mean(),
        'Nº clusters mixtos medio': df['finetuned_n_mixed_clusters'].mean(),
        'Modelos por cluster mixto': df['finetuned_avg_models_per_mixed'].mean(),
    }
    
    for key, value in stats.items():
        if pd.notna(value):
            print(f"  {key:30s}: {value:.4f}")
    
    # Mejor configuración
    print("\n--- Mejor Configuración Global (ARI) ---")
    best_idx = df['finetuned_ari'].idxmax()
    if pd.notna(best_idx):
        best = df.loc[best_idx]
        print(f"  Componente:      {best['component_type']}")
        print(f"  Capas:           {best['num_layers']}")
        print(f"  Recall@1:        {best['best_val_recall@1']:.4f}")
        print(f"  ARI:             {best['finetuned_ari']:.4f}")
        print(f"  Purity:          {best['finetuned_purity']:.4f}")
        print(f"  Pure %:          {best['finetuned_pure_percentage']:.2f}%")
        print(f"  Clusters puros:  {best['finetuned_n_pure_clusters']:.0f}")
        print(f"  Clusters mixtos: {best['finetuned_n_mixed_clusters']:.0f}")
    
    # Mejor por % clusters puros
    print("\n--- Mejor Configuración por % Clusters Puros ---")
    best_pure_idx = df['finetuned_pure_percentage'].idxmax()
    if pd.notna(best_pure_idx):
        best = df.loc[best_pure_idx]
        print(f"  Componente:      {best['component_type']}")
        print(f"  Capas:           {best['num_layers']}")
        print(f"  Pure %:          {best['finetuned_pure_percentage']:.2f}%")
        print(f"  ARI:             {best['finetuned_ari']:.4f}")
        print(f"  Clusters puros:  {best['finetuned_n_pure_clusters']:.0f}")
    
    # Análisis de métodos de clustering
    print("\n--- Métodos de Clustering ---")
    
    clusterer_stats = df.groupby('finetuned_clusterer').agg({
        'config_key': 'count',
        'finetuned_ari': ['mean', 'std', 'max'],
        'finetuned_purity': ['mean', 'max'],
        'finetuned_pure_percentage': ['mean', 'max'],
    }).round(4)
    
    clusterer_stats.columns = ['_'.join(col).strip() for col in clusterer_stats.columns.values]
    clusterer_stats = clusterer_stats.rename(columns={'config_key_count': 'count'})
    
    print("\n", clusterer_stats.to_string())


def analyze_problematic_classes(df: pd.DataFrame):
    """Análisis de clases problemáticas en CLIP."""
    print("\n" + "=" * 80)
    print("ANÁLISIS DE CLASES PROBLEMÁTICAS - CLIP")
    print("=" * 80)
    
    print("\n--- Estadísticas Generales ---")
    
    stats = {
        'Nº clases problemáticas (media)': df['finetuned_n_overlapping_classes'].mean(),
        'Nº clases problemáticas (max)': df['finetuned_n_overlapping_classes'].max(),
        'Clusters por clase problemática': df['finetuned_avg_clusters_per_class'].mean(),
        'Max clusters por clase': df['finetuned_max_clusters_per_class'].max(),
    }
    
    for key, value in stats.items():
        if pd.notna(value):
            print(f"  {key:40s}: {value:.2f}")
    
    # Configuración con menos clases problemáticas
    print("\n--- Configuración con Menos Clases Problemáticas ---")
    best_idx = df['finetuned_n_overlapping_classes'].idxmin()
    if pd.notna(best_idx):
        best = df.loc[best_idx]
        print(f"  Componente:                {best['component_type']}")
        print(f"  Capas:                     {best['num_layers']}")
        print(f"  Clases problemáticas:      {best['finetuned_n_overlapping_classes']:.0f}")
        print(f"  ARI:                       {best['finetuned_ari']:.4f}")
        print(f"  Pure %:                    {best['finetuned_pure_percentage']:.2f}%")
    
    # Por componente
    print("\n--- Clases Problemáticas por Componente ---")
    
    comp_stats = df.groupby('component_type').agg({
        'finetuned_n_overlapping_classes': ['mean', 'min', 'max'],
        'finetuned_avg_clusters_per_class': ['mean', 'max'],
    }).round(2)
    
    print("\n", comp_stats.to_string())


def analyze_baseline_vs_finetuned(df: pd.DataFrame):
    """Análisis de mejoras entre baseline y finetuned."""
    print("\n" + "=" * 80)
    print("ANÁLISIS BASELINE vs FINETUNED - CLIP")
    print("=" * 80)
    
    df['ari_improvement_pct'] = ((df['finetuned_ari'] - df['baseline_ari']) / df['baseline_ari'] * 100)
    
    print(f"\nMejora media en ARI:         {df['ari_improvement'].mean():.4f}")
    print(f"Mejora porcentual media:     {df['ari_improvement_pct'].mean():.2f}%")
    print(f"Mejora máxima en ARI:        {df['ari_improvement'].max():.4f}")
    print(f"Configuraciones que mejoran: {(df['ari_improvement'] > 0).sum()}/{len(df)}")
    
    # Mejor mejora
    print("\n--- Configuración con Mayor Mejora ---")
    best_imp_idx = df['ari_improvement'].idxmax()
    if pd.notna(best_imp_idx):
        best = df.loc[best_imp_idx]
        print(f"  Componente:      {best['component_type']}")
        print(f"  Capas:           {best['num_layers']}")
        print(f"  Baseline ARI:    {best['baseline_ari']:.4f}")
        print(f"  Finetuned ARI:   {best['finetuned_ari']:.4f}")
        print(f"  Mejora:          {best['ari_improvement']:.4f} ({best['ari_improvement_pct']:.2f}%)")
    
    # Por componente
    print("\n--- Mejoras por Componente ---")
    comp_imp = df.groupby('component_type').agg({
        'ari_improvement': ['mean', 'std', 'max'],
        'ari_improvement_pct': ['mean', 'std', 'max'],
    }).round(4)
    print("\n", comp_imp.to_string())
    
    # Por número de capas
    print("\n--- Mejoras por Número de Capas ---")
    layers_imp = df.groupby('num_layers').agg({
        'ari_improvement': ['mean', 'std'],
        'ari_improvement_pct': ['mean', 'std'],
    }).round(4)
    print("\n", layers_imp.to_string())


def analyze_efficiency(df: pd.DataFrame):
    """Análisis de eficiencia temporal."""
    print("\n" + "=" * 80)
    print("ANÁLISIS DE EFICIENCIA TEMPORAL - CLIP")
    print("=" * 80)
    
    print("\n--- Tiempos de Finetuning ---")
    print(f"Duración media:      {df['duration_minutes'].mean():.2f} minutos")
    print(f"Duración mínima:     {df['duration_minutes'].min():.2f} minutos")
    print(f"Duración máxima:     {df['duration_minutes'].max():.2f} minutos")
    
    # Por componente
    print("\n--- Tiempos por Componente ---")
    comp_time = df.groupby('component_type').agg({
        'duration_minutes': ['mean', 'std', 'min', 'max'],
        'embeddings_duration_minutes': ['mean', 'std'],
    }).round(2)
    print("\n", comp_time.to_string())
    
    # Por capas
    print("\n--- Tiempos por Número de Capas ---")
    layers_time = df.groupby('num_layers').agg({
        'duration_minutes': ['mean', 'std'],
    }).round(2)
    print("\n", layers_time.to_string())
    
    # Eficiencia
    print("\n--- Eficiencia (ARI / Tiempo) ---")
    df['efficiency'] = df['finetuned_ari'] / (df['duration_minutes'] / 60)
    
    print(f"\nEficiencia media: {df['efficiency'].mean():.4f} ARI/hora")
    
    best_eff_idx = df['efficiency'].idxmax()
    if pd.notna(best_eff_idx):
        best = df.loc[best_eff_idx]
        print(f"\nConfiguración más eficiente:")
        print(f"  Componente: {best['component_type']}")
        print(f"  Capas:      {best['num_layers']}")
        print(f"  ARI:        {best['finetuned_ari']:.4f}")
        print(f"  Tiempo:     {best['duration_minutes']:.2f} min")
        print(f"  Eficiencia: {best['efficiency']:.4f} ARI/hora")


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Genera tabla resumen de configuraciones CLIP."""
    print("\n" + "=" * 80)
    print("TABLA RESUMEN - CLIP LAYERS")
    print("=" * 80)
    
    summary_cols = [
        'component_type',
        'num_layers',
        'best_val_recall@1',
        'best_val_recall@5',
        'finetuned_ari',
        'finetuned_pure_percentage',
        'finetuned_n_pure_clusters',
        'duration_minutes',
    ]
    
    available_cols = [col for col in summary_cols if col in df.columns]
    summary = df[available_cols].round(4)
    
    # Ordenar por ARI descendente
    summary = summary.sort_values('finetuned_ari', ascending=False)
    
    print("\n", summary.to_string(index=False))
    
    return summary


def save_analysis_results(output_dir: Path, **dataframes):
    """Guarda los resultados del análisis en CSV."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for name, df in dataframes.items():
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            output_file = output_dir / f'{name}.csv'
            df.to_csv(output_file)
            print(f"[OK] {output_file}")


def main():
    """Función principal para análisis de CLIP layers."""
    csv_path = Path('results/analysis/clip_layers_results.csv')
    
    if not csv_path.exists():
        print(f"[ERROR] No se encontró el archivo: {csv_path}")
        print("Ejecuta primero: python scripts/extract_clip_layers_results.py")
        return
    
    print("=" * 80)
    print("ANÁLISIS ESTADÍSTICO - CLIP LAYERS STUDY")
    print("=" * 80)
    
    # Cargar datos
    df = load_data(csv_path)
    print(f"\nDatos cargados: {len(df)} configuraciones")
    
    # Realizar análisis
    component_analysis = analyze_by_component(df)
    layers_analysis = analyze_by_num_layers(df)
    df_vision, df_text = analyze_component_x_layers(df)
    analyze_progression_by_layers(df)
    analyze_clustering_quality(df)
    analyze_problematic_classes(df)
    analyze_baseline_vs_finetuned(df)
    analyze_efficiency(df)
    summary = generate_summary_table(df)
    
    # Guardar resultados
    output_dir = Path('results/analysis/statistics')
    print(f"\n{'='*80}")
    print("GUARDANDO RESULTADOS")
    print(f"{'='*80}")
    
    save_analysis_results(
        output_dir,
        clip_component_analysis=component_analysis,
        clip_layers_analysis=layers_analysis,
        clip_summary=summary,
    )
    
    print(f"\n{'='*80}")
    print("ANÁLISIS COMPLETADO")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

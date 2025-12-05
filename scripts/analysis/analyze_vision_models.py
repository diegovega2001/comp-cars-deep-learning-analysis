"""
Script para análisis estadístico detallado de modelos de visión (ResNet50 y ViT-B/32).
Genera análisis comparativos entre modelos, objetivos, criterios de pérdida y vistas.
"""

import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


def load_data(csv_path: Path) -> pd.DataFrame:
    """Carga el CSV de resultados de modelos de visión."""
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def analyze_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis comparativo entre ResNet50 y ViT-B/32."""
    print("\n" + "=" * 80)
    print("ANÁLISIS POR MODELO (ResNet50 vs ViT-B/32)")
    print("=" * 80)
    
    grouped = df.groupby('model_name').agg({
        'config_key': 'count',
        
        # Métricas de finetuning
        'best_val_accuracy': ['mean', 'std', 'min', 'max'],
        'best_val_recall@1': ['mean', 'std', 'min', 'max'],
        'best_val_recall@5': ['mean', 'std', 'min', 'max'],
        'duration_minutes': ['mean', 'std', 'min', 'max'],
        'total_epochs': ['mean', 'std'],
        
        # Métricas de clustering
        'finetuned_ari': ['mean', 'std', 'min', 'max'],
        'finetuned_nmi': ['mean', 'std', 'min', 'max'],
        'finetuned_purity': ['mean', 'std', 'min', 'max'],
        'ari_improvement': ['mean', 'std'],
        
        # Calidad de clusters
        'finetuned_pure_percentage': ['mean', 'std', 'min', 'max'],
        'finetuned_n_pure_clusters': ['mean', 'std'],
        'finetuned_n_mixed_clusters': ['mean', 'std'],
        'finetuned_avg_models_per_mixed': ['mean', 'std'],
        
        # Tiempos
        'embeddings_duration_minutes': ['mean', 'std'],
    }).round(4)
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})
    
    print("\n", grouped.to_string())
    
    return grouped


def analyze_by_objective(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis comparativo entre classification y metric_learning."""
    print("\n" + "=" * 80)
    print("ANÁLISIS POR OBJETIVO (Classification vs Metric Learning)")
    print("=" * 80)
    
    grouped = df.groupby('objective').agg({
        'config_key': 'count',
        
        # Classification usa accuracy, metric_learning usa recall
        'best_val_accuracy': ['mean', 'std', 'min', 'max'],
        'best_val_recall@1': ['mean', 'std', 'min', 'max'],
        'best_val_recall@5': ['mean', 'std', 'min', 'max'],
        
        # Clustering
        'finetuned_ari': ['mean', 'std', 'max'],
        'finetuned_pure_percentage': ['mean', 'std', 'max'],
        'ari_improvement': ['mean', 'std'],
        
        # Tiempos
        'duration_minutes': ['mean', 'std'],
        'embeddings_duration_minutes': ['mean', 'std'],
    }).round(4)
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})
    
    print("\n", grouped.to_string())
    
    return grouped


def analyze_by_criterion(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis comparativo entre funciones de pérdida."""
    print("\n" + "=" * 80)
    print("ANÁLISIS POR FUNCIÓN DE PÉRDIDA")
    print("=" * 80)
    
    grouped = df.groupby('criterion').agg({
        'config_key': 'count',
        'model_name': lambda x: ', '.join(x.unique()),
        
        # Métricas principales
        'best_val_accuracy': ['mean', 'std', 'max'],
        'best_val_recall@1': ['mean', 'std', 'max'],
        'best_val_recall@5': ['mean', 'std', 'max'],
        
        # Clustering
        'finetuned_ari': ['mean', 'std', 'max'],
        'finetuned_purity': ['mean', 'std', 'max'],
        'finetuned_pure_percentage': ['mean', 'std', 'max'],
        'ari_improvement': ['mean', 'std'],
        
        # Eficiencia
        'duration_minutes': ['mean', 'std'],
    }).round(4)
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})
    
    print("\n", grouped.to_string())
    
    # Encontrar mejor función de pérdida por métrica
    print("\n" + "-" * 80)
    print("MEJORES FUNCIONES DE PÉRDIDA POR MÉTRICA:")
    print("-" * 80)
    
    metrics = {
        'Recall@1': 'best_val_recall@1_max',
        'Recall@5': 'best_val_recall@5_max',
    }
    
    for metric_name, col in metrics.items():
        if col in grouped.columns:
            best = grouped[col].idxmax()
            if pd.notna(best):
                value = grouped.loc[best, col]
                print(f"  {metric_name:12s}: {best:20s} ({value:.4f})")
    
    return grouped


def analyze_by_views(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis comparativo entre front y front+rear."""
    print("\n" + "=" * 80)
    print("ANÁLISIS POR VISTAS (Front vs Front+Rear)")
    print("=" * 80)
    
    grouped = df.groupby('views').agg({
        'config_key': 'count',
        
        # Métricas
        'best_val_accuracy': ['mean', 'std'],
        'best_val_recall@1': ['mean', 'std'],
        'best_val_recall@5': ['mean', 'std'],
        'finetuned_ari': ['mean', 'std'],
        'finetuned_pure_percentage': ['mean', 'std'],
        
        # Tiempos
        'duration_minutes': ['mean', 'std'],
        'embeddings_duration_minutes': ['mean', 'std'],
    }).round(4)
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})
    
    print("\n", grouped.to_string())
    
    # Comparación directa
    print("\n" + "-" * 80)
    print("COMPARACIÓN FRONT vs FRONT+REAR:")
    print("-" * 80)
    
    if 'front' in grouped.index and 'front + rear' in grouped.index:
        front = grouped.loc['front']
        rear = grouped.loc['front + rear']
        
        print(f"\nARI medio:")
        print(f"  Front:       {front['finetuned_ari_mean']:.4f}")
        print(f"  Front+Rear:  {rear['finetuned_ari_mean']:.4f}")
        print(f"  Mejora:      {(rear['finetuned_ari_mean'] - front['finetuned_ari_mean']):.4f}")
        
        print(f"\nPure % medio:")
        print(f"  Front:       {front['finetuned_pure_percentage_mean']:.4f}%")
        print(f"  Front+Rear:  {rear['finetuned_pure_percentage_mean']:.4f}%")
        print(f"  Mejora:      {(rear['finetuned_pure_percentage_mean'] - front['finetuned_pure_percentage_mean']):.4f}%")
    
    return grouped


def analyze_clustering_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis detallado de calidad de clustering."""
    print("\n" + "=" * 80)
    print("ANÁLISIS DE CALIDAD DE CLUSTERING")
    print("=" * 80)
    
    print("\n--- Estadísticas Generales de Clustering ---")
    
    # Estadísticas generales
    stats = {
        'ARI medio': df['finetuned_ari'].mean(),
        'ARI máximo': df['finetuned_ari'].max(),
        'ARI mínimo': df['finetuned_ari'].min(),
        'Purity media': df['finetuned_purity'].mean(),
        '% Clusters puros medio': df['finetuned_pure_percentage'].mean(),
        'Nº clusters puros medio': df['finetuned_n_pure_clusters'].mean(),
        'Nº clusters mixtos medio': df['finetuned_n_mixed_clusters'].mean(),
        'Modelos por cluster mixto': df['finetuned_avg_models_per_mixed'].mean(),
    }
    
    for key, value in stats.items():
        if pd.notna(value):
            print(f"  {key:30s}: {value:.4f}")
    
    # Mejor configuración por ARI
    print("\n--- Mejor Configuración por ARI ---")
    best_ari_idx = df['finetuned_ari'].idxmax()
    if pd.notna(best_ari_idx):
        best = df.loc[best_ari_idx]
        print(f"  Modelo:          {best['model_name']}")
        print(f"  Objetivo:        {best['objective']}")
        print(f"  Criterio:        {best['criterion']}")
        print(f"  Vistas:          {best['views']}")
        print(f"  ARI:             {best['finetuned_ari']:.4f}")
        print(f"  Purity:          {best['finetuned_purity']:.4f}")
        print(f"  Pure %:          {best['finetuned_pure_percentage']:.2f}%")
        print(f"  Clusters puros:  {best['finetuned_n_pure_clusters']:.0f}")
        print(f"  Clusters mixtos: {best['finetuned_n_mixed_clusters']:.0f}")
    
    # Mejor configuración por % clusters puros
    print("\n--- Mejor Configuración por % Clusters Puros ---")
    best_pure_idx = df['finetuned_pure_percentage'].idxmax()
    if pd.notna(best_pure_idx):
        best = df.loc[best_pure_idx]
        print(f"  Modelo:          {best['model_name']}")
        print(f"  Objetivo:        {best['objective']}")
        print(f"  Criterio:        {best['criterion']}")
        print(f"  Vistas:          {best['views']}")
        print(f"  Pure %:          {best['finetuned_pure_percentage']:.2f}%")
        print(f"  ARI:             {best['finetuned_ari']:.4f}")
        print(f"  Clusters puros:  {best['finetuned_n_pure_clusters']:.0f}")
    
    # Análisis de métodos de clustering
    print("\n--- Mejores Métodos de Clustering ---")
    
    clusterer_stats = df.groupby('finetuned_clusterer').agg({
        'config_key': 'count',
        'finetuned_ari': ['mean', 'std', 'max'],
        'finetuned_purity': ['mean', 'max'],
        'finetuned_pure_percentage': ['mean', 'max'],
    }).round(4)
    
    clusterer_stats.columns = ['_'.join(col).strip() for col in clusterer_stats.columns.values]
    clusterer_stats = clusterer_stats.rename(columns={'config_key_count': 'count'})
    
    print("\n", clusterer_stats.to_string())
    
    return clusterer_stats


def analyze_problematic_classes(df: pd.DataFrame):
    """Análisis de clases problemáticas (overlapping)."""
    print("\n" + "=" * 80)
    print("ANÁLISIS DE CLASES PROBLEMÁTICAS (Overlapping)")
    print("=" * 80)
    
    # Estadísticas de overlapping
    print("\n--- Estadísticas Generales ---")
    
    stats = {
        'Nº clases problemáticas (media)': df['finetuned_n_overlapping_classes'].mean(),
        'Nº clases problemáticas (max)': df['finetuned_n_overlapping_classes'].max(),
        'Clusters por clase problemática (media)': df['finetuned_avg_clusters_per_class'].mean(),
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
        print(f"  Modelo:                    {best['model_name']}")
        print(f"  Objetivo:                  {best['objective']}")
        print(f"  Criterio:                  {best['criterion']}")
        print(f"  Vistas:                    {best['views']}")
        print(f"  Clases problemáticas:      {best['finetuned_n_overlapping_classes']:.0f}")
        print(f"  ARI:                       {best['finetuned_ari']:.4f}")
        print(f"  Pure %:                    {best['finetuned_pure_percentage']:.2f}%")
    
    # Top 5 clases más problemáticas por configuración
    print("\n--- Top 3 Configuraciones con Clases Más Problemáticas ---")
    
    # Filtrar y ordenar
    df_sorted = df.nlargest(3, 'finetuned_n_overlapping_classes')
    
    for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"\n{i}. {row['model_name']} - {row['criterion']} - {row['views']}")
        print(f"   Clases problemáticas: {row['finetuned_n_overlapping_classes']:.0f}")
        print(f"   Clusters por clase:   {row['finetuned_avg_clusters_per_class']:.2f}")
        if pd.notna(row['finetuned_top_overlapping']):
            top_classes = str(row['finetuned_top_overlapping']).split(';')[:5]
            print(f"   Top clases: {', '.join(top_classes)}")


def analyze_baseline_vs_finetuned(df: pd.DataFrame):
    """Análisis de mejoras entre baseline y finetuned."""
    print("\n" + "=" * 80)
    print("ANÁLISIS BASELINE vs FINETUNED")
    print("=" * 80)
    
    # Mejoras en ARI
    print("\n--- Mejoras en ARI ---")
    
    df['ari_improvement_pct'] = ((df['finetuned_ari'] - df['baseline_ari']) / df['baseline_ari'] * 100)
    
    print(f"\nMejora media en ARI:        {df['ari_improvement'].mean():.4f}")
    print(f"Mejora porcentual media:    {df['ari_improvement_pct'].mean():.2f}%")
    print(f"Mejora máxima en ARI:       {df['ari_improvement'].max():.4f}")
    print(f"Configuraciones que mejoran: {(df['ari_improvement'] > 0).sum()}/{len(df)}")
    
    # Mejor mejora
    print("\n--- Configuración con Mayor Mejora en ARI ---")
    best_imp_idx = df['ari_improvement'].idxmax()
    if pd.notna(best_imp_idx):
        best = df.loc[best_imp_idx]
        print(f"  Modelo:          {best['model_name']}")
        print(f"  Objetivo:        {best['objective']}")
        print(f"  Criterio:        {best['criterion']}")
        print(f"  Vistas:          {best['views']}")
        print(f"  Baseline ARI:    {best['baseline_ari']:.4f}")
        print(f"  Finetuned ARI:   {best['finetuned_ari']:.4f}")
        print(f"  Mejora:          {best['ari_improvement']:.4f} ({best['ari_improvement_pct']:.2f}%)")
    
    # Mejoras por modelo
    print("\n--- Mejoras por Modelo ---")
    model_imp = df.groupby('model_name').agg({
        'ari_improvement': ['mean', 'std'],
        'ari_improvement_pct': ['mean', 'std'],
    }).round(4)
    print("\n", model_imp.to_string())
    
    # Mejoras por objetivo
    print("\n--- Mejoras por Objetivo ---")
    obj_imp = df.groupby('objective').agg({
        'ari_improvement': ['mean', 'std'],
        'ari_improvement_pct': ['mean', 'std'],
    }).round(4)
    print("\n", obj_imp.to_string())


def analyze_efficiency(df: pd.DataFrame):
    """Análisis de eficiencia temporal."""
    print("\n" + "=" * 80)
    print("ANÁLISIS DE EFICIENCIA TEMPORAL")
    print("=" * 80)
    
    # Tiempos de entrenamiento
    print("\n--- Tiempos de Finetuning ---")
    print(f"Duración media:      {df['duration_minutes'].mean():.2f} minutos")
    print(f"Duración mínima:     {df['duration_minutes'].min():.2f} minutos")
    print(f"Duración máxima:     {df['duration_minutes'].max():.2f} minutos")
    print(f"Desviación estándar: {df['duration_minutes'].std():.2f} minutos")
    
    # Tiempos por modelo
    print("\n--- Tiempos por Modelo ---")
    model_time = df.groupby('model_name').agg({
        'duration_minutes': ['mean', 'std', 'min', 'max'],
        'embeddings_duration_minutes': ['mean', 'std'],
    }).round(2)
    print("\n", model_time.to_string())
    
    # Tiempos por objetivo
    print("\n--- Tiempos por Objetivo ---")
    obj_time = df.groupby('objective').agg({
        'duration_minutes': ['mean', 'std'],
        'embeddings_duration_minutes': ['mean', 'std'],
    }).round(2)
    print("\n", obj_time.to_string())
    
    # Eficiencia: ARI / tiempo
    print("\n--- Eficiencia (ARI / Tiempo de Entrenamiento) ---")
    df['efficiency'] = df['finetuned_ari'] / (df['duration_minutes'] / 60)  # ARI por hora
    
    print(f"\nEficiencia media: {df['efficiency'].mean():.4f} ARI/hora")
    
    best_eff_idx = df['efficiency'].idxmax()
    if pd.notna(best_eff_idx):
        best = df.loc[best_eff_idx]
        print(f"\nConfiguración más eficiente:")
        print(f"  Modelo:     {best['model_name']}")
        print(f"  Objetivo:   {best['objective']}")
        print(f"  Criterio:   {best['criterion']}")
        print(f"  Vistas:     {best['views']}")
        print(f"  ARI:        {best['finetuned_ari']:.4f}")
        print(f"  Tiempo:     {best['duration_minutes']:.2f} min")
        print(f"  Eficiencia: {best['efficiency']:.4f} ARI/hora")


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Genera una tabla resumen con las métricas clave de cada configuración."""
    print("\n" + "=" * 80)
    print("TABLA RESUMEN DE CONFIGURACIONES")
    print("=" * 80)
    
    summary_cols = [
        'model_name',
        'objective',
        'criterion',
        'views',
        'best_val_accuracy',
        'best_val_recall@1',
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
        if df is not None and not df.empty:
            output_file = output_dir / f'{name}.csv'
            df.to_csv(output_file)
            print(f"[OK] {output_file}")


def main():
    """Función principal para análisis de modelos de visión."""
    csv_path = Path('results/analysis/vision_models_results.csv')
    
    if not csv_path.exists():
        print(f"[ERROR] No se encontró el archivo: {csv_path}")
        print("Ejecuta primero: python scripts/extract_vision_results.py")
        return
    
    print("=" * 80)
    print("ANÁLISIS ESTADÍSTICO - MODELOS DE VISIÓN")
    print("=" * 80)
    
    # Cargar datos
    df = load_data(csv_path)
    print(f"\nDatos cargados: {len(df)} configuraciones")
    
    # Realizar análisis
    model_analysis = analyze_by_model(df)
    objective_analysis = analyze_by_objective(df)
    criterion_analysis = analyze_by_criterion(df)
    views_analysis = analyze_by_views(df)
    clustering_analysis = analyze_clustering_quality(df)
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
        vision_model_analysis=model_analysis,
        vision_objective_analysis=objective_analysis,
        vision_criterion_analysis=criterion_analysis,
        vision_views_analysis=views_analysis,
        vision_clustering_analysis=clustering_analysis,
        vision_summary=summary,
    )
    
    print(f"\n{'='*80}")
    print("ANÁLISIS COMPLETADO")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

"""
Script para extraer resultados de experimentos CLIP layers study.
Específico para estudios de fine-tuning por capas de visión y texto de CLIP.
Completamente separado del análisis de modelos de visión.
"""

import json
import hashlib
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

warnings.filterwarnings('ignore')


def extract_clip_layer_info(config: Dict) -> Dict[str, Any]:
    """
    Extrae información específica de capas CLIP desde la configuración.
    
    Returns:
        Dict con:
        - component_type: 'vision' o 'text'
        - num_layers: número de capas finetuneadas
        - phase_name: nombre de la fase
        - lr: learning rate
        - epochs_configured: épocas configuradas
    """
    phases = config.get('clip_finetuning_phases', {})
    
    if not phases:
        return {
            'component_type': None,
            'num_layers': None,
            'phase_name': None,
            'lr': None,
            'epochs_configured': None,
            'warmup_steps': None,
            'patience': None
        }
    
    # Obtener la primera (y única) fase
    phase_name = list(phases.keys())[0]
    phase_config = phases[phase_name]
    
    component_type = phase_config.get('type')  # 'vision' o 'text'
    
    if component_type == 'vision':
        num_layers = phase_config.get('num_vision_layers')
    elif component_type == 'text':
        num_layers = phase_config.get('num_text_layers')
    else:
        num_layers = None
    
    return {
        'component_type': component_type,
        'num_layers': num_layers,
        'phase_name': phase_name,
        'lr': phase_config.get('lr'),
        'epochs_configured': phase_config.get('epochs'),
        'warmup_steps': phase_config.get('warmup_steps'),
        'patience': phase_config.get('early_stopping', {}).get('patience')
    }


def create_clip_config_key(config: Dict) -> str:
    """
    Crea una clave única para configuración CLIP.
    Incluye el tipo de componente y número de capas para diferenciar experimentos.
    """
    layer_info = extract_clip_layer_info(config)
    
    key_params = [
        'views',
        'class_granularity',
        'min_images',
        'train_ratio',
        'val_ratio',
        'test_ratio',
        'seed',
        'image_size',
        'grayscale',
        'use_bbox',
        'augment',
        'model_name',
        'objective',
        'batch_size',
        'finetune_optimizer_type',
        'finetune_optimizer_weight_decay',
        'use_amp',
    ]
    
    key_values = []
    for param in key_params:
        value = config.get(param)
        if value is not None:
            if isinstance(value, list):
                value = '_'.join(map(str, value))
            key_values.append(f"{param}:{value}")
    
    # Agregar información de capas
    key_values.append(f"component:{layer_info['component_type']}")
    key_values.append(f"layers:{layer_info['num_layers']}")
    
    config_str = '|'.join(sorted(key_values))
    return hashlib.md5(config_str.encode()).hexdigest()


def calculate_duration_minutes(start_time_str: str, end_time_str: str) -> Optional[float]:
    """Calcula la duración en minutos entre dos timestamps ISO."""
    if not start_time_str or not end_time_str:
        return None
    try:
        start_dt = datetime.fromisoformat(start_time_str)
        end_dt = datetime.fromisoformat(end_time_str)
        return (end_dt - start_dt).total_seconds() / 60.0
    except:
        return None


def load_json_safe(filepath: Path) -> Optional[Dict]:
    """Carga un archivo JSON de forma segura."""
    try:
        if not filepath.exists():
            return None
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        
        if file_size_mb > 100:
            print(f"    [WARNING] Archivo muy grande ({file_size_mb:.1f} MB): {filepath.name}")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"    [ERROR] Error cargando {filepath}: {e}")
        return None


def extract_clip_finetuning_data(layers_study_path: Path) -> List[Dict]:
    """
    Extrae datos de experimentos de finetuning CLIP layers study.
    
    Args:
        layers_study_path: Path a results/models/clip/layers_study
    """
    finetune_data = []
    
    finetuning_path = layers_study_path / 'finetuning'
    if not finetuning_path.exists():
        print(f"[SKIP] No existe: {finetuning_path}")
        return []
    
    # Procesar vision y text
    for component_type in ['vision', 'text']:
        component_path = finetuning_path / component_type
        
        if not component_path.exists():
            print(f"[SKIP] No existe: {component_path}")
            continue
        
        print(f"\n--- CLIP {component_type.upper()} ---")
        
        # Buscar carpetas de capas (1 layer, 2 layers, ..., 12 layers)
        layer_folders = sorted([f for f in component_path.iterdir() if f.is_dir()])
        
        for layer_folder in layer_folders:
            results_file = layer_folder / 'results.json'
            
            if not results_file.exists():
                print(f"  [SKIP] No existe results.json en {layer_folder}")
                continue
            
            print(f"  Procesando: {component_type}/{layer_folder.name}")
            
            results = load_json_safe(results_file)
            if results is None:
                continue
            
            config = results.get('config', {})
            config_key = create_clip_config_key(config)
            layer_info = extract_clip_layer_info(config)
            
            # Información básica
            experiment_name = results.get('experiment_name', layer_folder.name)
            start_time = results.get('start_time')
            end_time = results.get('end_time')
            duration_minutes = calculate_duration_minutes(start_time, end_time)
            
            # Dataset stats
            dataset_stats = results.get('dataset_stats', {})
            
            # Baseline evaluation (CLIP usa recall, no accuracy)
            baseline = results.get('baseline', {}).get('evaluation', {})
            baseline_recall1 = baseline.get('recall@1', None)
            baseline_recall3 = baseline.get('recall@3', None)
            baseline_recall5 = baseline.get('recall@5', None)
            
            # Training history - puede estar en phases
            finetuning = results.get('finetuning', {})
            training_history = finetuning.get('training_history', {})
            
            # Si está en formato de fases, extraer la fase relevante
            phase_name = layer_info['phase_name']
            if phase_name and phase_name in training_history:
                history = training_history[phase_name]
            else:
                # Buscar cualquier fase o usar el historial completo
                phase_keys = [k for k in training_history.keys() if k.startswith('phase_')]
                if phase_keys:
                    history = training_history[phase_keys[0]]
                else:
                    history = training_history
            
            train_loss_list = history.get('train_loss', [])
            val_loss_list = history.get('val_loss', [])
            val_recall1_list = history.get('val_recall@1', [])
            val_recall3_list = history.get('val_recall@3', [])
            val_recall5_list = history.get('val_recall@5', [])
            
            total_epochs = len(train_loss_list)
            
            # Para CLIP, usar recall@1 como métrica principal
            if val_recall1_list:
                best_epoch = int(np.argmax(val_recall1_list))
                best_val_recall1 = float(np.max(val_recall1_list))
                best_val_recall3 = val_recall3_list[best_epoch] if val_recall3_list and best_epoch < len(val_recall3_list) else None
                best_val_recall5 = val_recall5_list[best_epoch] if val_recall5_list and best_epoch < len(val_recall5_list) else None
                best_val_loss = val_loss_list[best_epoch] if val_loss_list and best_epoch < len(val_loss_list) else None
            else:
                best_epoch = None
                best_val_recall1 = None
                best_val_recall3 = None
                best_val_recall5 = None
                best_val_loss = None
            
            # Final metrics
            final_train_loss = train_loss_list[-1] if train_loss_list else None
            final_val_loss = val_loss_list[-1] if val_loss_list else None
            
            # Finetuned evaluation
            finetuned = results.get('finetuned', {}).get('evaluation', {})
            finetuned_recall1 = finetuned.get('recall@1', None)
            finetuned_recall3 = finetuned.get('recall@3', None)
            finetuned_recall5 = finetuned.get('recall@5', None)
            
            # Summary
            summary = results.get('summary', {})
            recall1_improvement = summary.get('recall@1_improvement', None)
            recall3_improvement = summary.get('recall@3_improvement', None)
            recall5_improvement = summary.get('recall@5_improvement', None)
            
            # Calcular mejoras si no están en summary
            if recall1_improvement is None and baseline_recall1 is not None and finetuned_recall1 is not None:
                recall1_improvement = finetuned_recall1 - baseline_recall1
            
            if recall3_improvement is None and baseline_recall3 is not None and finetuned_recall3 is not None:
                recall3_improvement = finetuned_recall3 - baseline_recall3
            
            if recall5_improvement is None and baseline_recall5 is not None and finetuned_recall5 is not None:
                recall5_improvement = finetuned_recall5 - baseline_recall5
            
            # Construir registro
            record = {
                'config_key': config_key,
                'experiment_name': experiment_name,
                'timestamp': start_time,
                
                # Configuración CLIP específica
                'model_name': config.get('model_name', 'clip-vit-base-patch32'),
                'objective': 'CLIP',
                'component_type': layer_info['component_type'],
                'num_layers': layer_info['num_layers'],
                'phase_name': layer_info['phase_name'],
                'views': config.get('views', []),
                'views_str': ' + '.join(config.get('views', [])),
                'num_views': len(config.get('views', [])),
                
                # Hiperparámetros
                'batch_size': config.get('batch_size'),
                'lr': layer_info['lr'],
                'weight_decay': config.get('finetune_optimizer_weight_decay'),
                'optimizer': config.get('finetune_optimizer_type'),
                'use_amp': config.get('use_amp'),
                'warmup_steps': layer_info['warmup_steps'],
                'patience': layer_info['patience'],
                'epochs_configured': layer_info['epochs_configured'],
                'seed': config.get('seed'),
                
                # Dataset
                'num_training_classes': dataset_stats.get('num_training_classes'),
                'num_total_classes': dataset_stats.get('num_total_classes'),
                'num_oneshot_classes': dataset_stats.get('num_oneshot_classes'),
                'train_samples': dataset_stats.get('splits', {}).get('train', {}).get('samples'),
                'val_samples': dataset_stats.get('splits', {}).get('val', {}).get('samples'),
                'test_samples': dataset_stats.get('splits', {}).get('test', {}).get('samples'),
                
                # Tiempos
                'start_time': start_time,
                'end_time': end_time,
                'duration_minutes': duration_minutes,
                
                # Training
                'total_epochs': total_epochs,
                'best_epoch': best_epoch,
                
                # Baseline (recall solamente, CLIP no usa accuracy)
                'baseline_recall@1': baseline_recall1,
                'baseline_recall@3': baseline_recall3,
                'baseline_recall@5': baseline_recall5,
                
                # Best metrics
                'best_val_loss': best_val_loss,
                'best_val_recall@1': best_val_recall1,
                'best_val_recall@3': best_val_recall3,
                'best_val_recall@5': best_val_recall5,
                
                # Final metrics
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                
                # Finetuned
                'finetuned_recall@1': finetuned_recall1,
                'finetuned_recall@3': finetuned_recall3,
                'finetuned_recall@5': finetuned_recall5,
                
                # Improvements
                'recall@1_improvement': recall1_improvement,
                'recall@3_improvement': recall3_improvement,
                'recall@5_improvement': recall5_improvement,
            }
            
            finetune_data.append(record)
    
    return finetune_data


def analyze_cluster_quality(cluster_analysis: Dict) -> Dict[str, Any]:
    """Analiza la calidad de los clusters generados."""
    if not cluster_analysis:
        return {}
    
    total_clusters = len(cluster_analysis)
    pure_clusters = []
    mixed_clusters = []
    dominant_clusters = []
    
    cluster_sizes = []
    models_per_mixed_cluster = []
    
    for cluster_id, cluster_info in cluster_analysis.items():
        cluster_sizes.append(cluster_info['size'])
        
        if cluster_info.get('is_pure', False):
            pure_clusters.append(cluster_id)
        
        if cluster_info.get('is_mixed', False):
            mixed_clusters.append(cluster_id)
            models_per_mixed_cluster.append(cluster_info['n_unique_models'])
        
        if cluster_info.get('is_dominant', False):
            dominant_clusters.append(cluster_id)
    
    n_pure = len(pure_clusters)
    n_mixed = len(mixed_clusters)
    n_dominant = len(dominant_clusters)
    
    pure_percentage = (n_pure / total_clusters * 100) if total_clusters > 0 else 0
    mixed_percentage = (n_mixed / total_clusters * 100) if total_clusters > 0 else 0
    dominant_percentage = (n_dominant / total_clusters * 100) if total_clusters > 0 else 0
    
    avg_models_per_mixed = np.mean(models_per_mixed_cluster) if models_per_mixed_cluster else 0
    avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
    median_cluster_size = np.median(cluster_sizes) if cluster_sizes else 0
    
    return {
        'n_clusters': total_clusters,
        'n_pure_clusters': n_pure,
        'n_mixed_clusters': n_mixed,
        'n_dominant_clusters': n_dominant,
        'pure_percentage': pure_percentage,
        'mixed_percentage': mixed_percentage,
        'dominant_percentage': dominant_percentage,
        'avg_models_per_mixed_cluster': avg_models_per_mixed,
        'avg_cluster_size': avg_cluster_size,
        'median_cluster_size': median_cluster_size,
        'min_cluster_size': int(np.min(cluster_sizes)) if cluster_sizes else 0,
        'max_cluster_size': int(np.max(cluster_sizes)) if cluster_sizes else 0,
    }


def extract_class_overlap_info(class_cluster_overlap: List[Dict]) -> Dict[str, Any]:
    """Extrae información sobre clases problemáticas."""
    if not class_cluster_overlap:
        return {
            'n_overlapping_classes': 0,
            'avg_clusters_per_class': 0,
            'max_clusters_per_class': 0,
            'overlapping_class_names': []
        }
    
    n_overlapping = len(class_cluster_overlap)
    clusters_per_class = [item['n_clusters'] for item in class_cluster_overlap]
    
    avg_clusters = np.mean(clusters_per_class) if clusters_per_class else 0
    max_clusters = np.max(clusters_per_class) if clusters_per_class else 0
    
    # Top 10 clases más problemáticas
    sorted_classes = sorted(class_cluster_overlap, key=lambda x: x['n_clusters'], reverse=True)
    top_classes = [
        f"{item['class_name']}({item['n_clusters']})" 
        for item in sorted_classes[:10]
    ]
    
    return {
        'n_overlapping_classes': n_overlapping,
        'avg_clusters_per_overlapping_class': avg_clusters,
        'max_clusters_per_overlapping_class': int(max_clusters),
        'top_overlapping_classes': '; '.join(top_classes)
    }


def extract_clip_embeddings_data(layers_study_path: Path) -> List[Dict]:
    """
    Extrae datos de experimentos de embeddings CLIP layers study.
    
    Args:
        layers_study_path: Path a results/models/clip/layers_study
    """
    embeddings_data = []
    
    embeddings_path = layers_study_path / 'embeddings'
    if not embeddings_path.exists():
        print(f"[SKIP] No existe: {embeddings_path}")
        return []
    
    # Procesar vision y text
    for component_type in ['vision', 'text']:
        component_path = embeddings_path / component_type
        
        if not component_path.exists():
            print(f"[SKIP] No existe: {component_path}")
            continue
        
        print(f"\n--- CLIP {component_type.upper()} EMBEDDINGS ---")
        
        # Buscar carpetas de capas
        layer_folders = sorted([f for f in component_path.iterdir() if f.is_dir()])
        
        for layer_folder in layer_folders:
            results_file = layer_folder / 'results.json'
            
            if not results_file.exists():
                print(f"  [SKIP] No existe results.json en {layer_folder}")
                continue
            
            print(f"  Procesando: {component_type}/{layer_folder.name}/embeddings")
            
            results = load_json_safe(results_file)
            if results is None:
                continue
            
            config = results.get('config', {})
            config_key = create_clip_config_key(config)
            layer_info = extract_clip_layer_info(config)
            
            # Información básica
            experiment_name = results.get('experiment_name', layer_folder.name)
            start_time = results.get('start_time')
            end_time = results.get('end_time')
            duration_minutes = calculate_duration_minutes(start_time, end_time)
            
            # Embeddings info
            embeddings_info = results.get('embeddings_info', {})
            baseline_shape = embeddings_info.get('baseline', {}).get('shape', [])
            finetuned_shape = embeddings_info.get('finetuned', {}).get('shape', [])
            
            # Best methods
            baseline_analysis = results.get('baseline_analysis', {})
            baseline_best_reducer = baseline_analysis.get('reduction', {}).get('best_method')
            baseline_best_shape = baseline_analysis.get('reduction', {}).get('best_embeddings_shape')
            
            # Baseline clustering
            baseline_clustering = baseline_analysis.get('clustering', {})
            
            # Extraer mejor método baseline
            baseline_best_comparison = baseline_clustering.get('comparison_df', [{}])
            if baseline_best_comparison:
                baseline_best = baseline_best_comparison[0]
                baseline_clusterer = baseline_best.get('method')
                baseline_ari = baseline_best.get('adjusted_rand_score')
                baseline_nmi = baseline_best.get('normalized_mutual_info')
                baseline_purity = baseline_best.get('purity')
                baseline_silhouette = baseline_best.get('silhouette_score')
                baseline_n_clusters = baseline_best.get('n_clusters')
            else:
                baseline_clusterer = None
                baseline_ari = None
                baseline_nmi = None
                baseline_purity = None
                baseline_silhouette = None
                baseline_n_clusters = None
            
            # Baseline cluster analysis
            baseline_visualization = baseline_analysis.get('visualization', {})
            baseline_cluster_analysis = baseline_visualization.get('cluster_analysis', {})
            baseline_cluster_quality = analyze_cluster_quality(baseline_cluster_analysis)
            
            # Finetuned analysis
            finetuned_analysis = results.get('finetuned_analysis', {})
            finetuned_best_reducer = finetuned_analysis.get('reduction', {}).get('best_method')
            finetuned_best_shape = finetuned_analysis.get('reduction', {}).get('best_embeddings_shape')
            
            # Finetuned clustering
            finetuned_clustering = finetuned_analysis.get('clustering', {})
            
            # Extraer mejor método finetuned
            finetuned_best_comparison = finetuned_clustering.get('comparison_df', [{}])
            if finetuned_best_comparison:
                finetuned_best = finetuned_best_comparison[0]
                finetuned_clusterer = finetuned_best.get('method')
                finetuned_ari = finetuned_best.get('adjusted_rand_score')
                finetuned_nmi = finetuned_best.get('normalized_mutual_info')
                finetuned_purity = finetuned_best.get('purity')
                finetuned_silhouette = finetuned_best.get('silhouette_score')
                finetuned_n_clusters = finetuned_best.get('n_clusters')
            else:
                finetuned_clusterer = None
                finetuned_ari = None
                finetuned_nmi = None
                finetuned_purity = None
                finetuned_silhouette = None
                finetuned_n_clusters = None
            
            # Finetuned cluster analysis
            finetuned_visualization = finetuned_analysis.get('visualization', {})
            finetuned_cluster_analysis = finetuned_visualization.get('cluster_analysis', {})
            finetuned_cluster_quality = analyze_cluster_quality(finetuned_cluster_analysis)
            
            # Class overlap
            finetuned_overlap = finetuned_visualization.get('class_cluster_overlap', [])
            finetuned_overlap_info = extract_class_overlap_info(finetuned_overlap)
            
            baseline_overlap = baseline_visualization.get('class_cluster_overlap', [])
            baseline_overlap_info = extract_class_overlap_info(baseline_overlap)
            
            # Comparison
            comparison = results.get('comparison', {})
            performance = comparison.get('performance_improvement', {})
            
            ari_improvement = performance.get('ari_improvement')
            
            if ari_improvement is None and baseline_ari is not None and finetuned_ari is not None:
                ari_improvement = finetuned_ari - baseline_ari
            
            # Construir registro
            record = {
                'config_key': config_key,
                'experiment_name': experiment_name,
                'timestamp': start_time,
                
                # Configuración
                'model_name': 'clip-vit-base-patch32',
                'objective': 'CLIP',
                'component_type': layer_info['component_type'],
                'num_layers': layer_info['num_layers'],
                'views_str': ' + '.join(config.get('views', [])),
                
                # Tiempos
                'embeddings_start_time': start_time,
                'embeddings_end_time': end_time,
                'embeddings_duration_minutes': duration_minutes,
                
                # Embeddings shapes
                'baseline_embedding_shape': baseline_shape,
                'finetuned_embedding_shape': finetuned_shape,
                
                # Reduction methods
                'baseline_reducer': baseline_best_reducer,
                'baseline_reduced_shape': baseline_best_shape,
                'finetuned_reducer': finetuned_best_reducer,
                'finetuned_reduced_shape': finetuned_best_shape,
                
                # Clustering methods
                'baseline_clusterer': baseline_clusterer,
                'finetuned_clusterer': finetuned_clusterer,
                
                # Baseline clustering metrics
                'baseline_ari': baseline_ari,
                'baseline_nmi': baseline_nmi,
                'baseline_purity': baseline_purity,
                'baseline_silhouette': baseline_silhouette,
                'baseline_n_clusters': baseline_n_clusters,
                
                # Finetuned clustering metrics
                'finetuned_ari': finetuned_ari,
                'finetuned_nmi': finetuned_nmi,
                'finetuned_purity': finetuned_purity,
                'finetuned_silhouette': finetuned_silhouette,
                'finetuned_n_clusters': finetuned_n_clusters,
                
                # Improvements
                'ari_improvement': ari_improvement,
                'nmi_improvement': (finetuned_nmi - baseline_nmi) if (finetuned_nmi and baseline_nmi) else None,
                'purity_improvement': (finetuned_purity - baseline_purity) if (finetuned_purity and baseline_purity) else None,
                
                # Baseline cluster quality
                'baseline_n_pure_clusters': baseline_cluster_quality.get('n_pure_clusters'),
                'baseline_pure_percentage': baseline_cluster_quality.get('pure_percentage'),
                'baseline_n_mixed_clusters': baseline_cluster_quality.get('n_mixed_clusters'),
                'baseline_avg_models_per_mixed': baseline_cluster_quality.get('avg_models_per_mixed_cluster'),
                'baseline_avg_cluster_size': baseline_cluster_quality.get('avg_cluster_size'),
                
                # Finetuned cluster quality
                'finetuned_n_pure_clusters': finetuned_cluster_quality.get('n_pure_clusters'),
                'finetuned_pure_percentage': finetuned_cluster_quality.get('pure_percentage'),
                'finetuned_n_mixed_clusters': finetuned_cluster_quality.get('n_mixed_clusters'),
                'finetuned_avg_models_per_mixed': finetuned_cluster_quality.get('avg_models_per_mixed_cluster'),
                'finetuned_avg_cluster_size': finetuned_cluster_quality.get('avg_cluster_size'),
                'finetuned_median_cluster_size': finetuned_cluster_quality.get('median_cluster_size'),
                'finetuned_min_cluster_size': finetuned_cluster_quality.get('min_cluster_size'),
                'finetuned_max_cluster_size': finetuned_cluster_quality.get('max_cluster_size'),
                
                # Class overlap (baseline)
                'baseline_n_overlapping_classes': baseline_overlap_info.get('n_overlapping_classes'),
                'baseline_avg_clusters_per_class': baseline_overlap_info.get('avg_clusters_per_overlapping_class'),
                'baseline_max_clusters_per_class': baseline_overlap_info.get('max_clusters_per_overlapping_class'),
                'baseline_top_overlapping': baseline_overlap_info.get('top_overlapping_classes'),
                
                # Class overlap (finetuned)
                'finetuned_n_overlapping_classes': finetuned_overlap_info.get('n_overlapping_classes'),
                'finetuned_avg_clusters_per_class': finetuned_overlap_info.get('avg_clusters_per_overlapping_class'),
                'finetuned_max_clusters_per_class': finetuned_overlap_info.get('max_clusters_per_overlapping_class'),
                'finetuned_top_overlapping': finetuned_overlap_info.get('top_overlapping_classes'),
            }
            
            embeddings_data.append(record)
    
    return embeddings_data


def main():
    """Función principal para extraer todos los resultados CLIP layers study."""
    clip_path = Path('results/models/clip/layers_study')
    
    if not clip_path.exists():
        print(f"[ERROR] No existe el directorio: {clip_path}")
        return
    
    print("=" * 80)
    print("EXTRACCIÓN DE RESULTADOS - CLIP LAYERS STUDY")
    print("=" * 80)
    
    # Extraer datos
    finetuning_data = extract_clip_finetuning_data(clip_path)
    embeddings_data = extract_clip_embeddings_data(clip_path)
    
    # Convertir a DataFrames
    print(f"\n{'='*80}")
    print("CREANDO DATAFRAMES")
    print(f"{'='*80}")
    
    df_finetuning = pd.DataFrame(finetuning_data)
    df_embeddings = pd.DataFrame(embeddings_data)
    
    print(f"Finetuning: {len(df_finetuning)} experimentos")
    print(f"Embeddings: {len(df_embeddings)} experimentos")
    
    # Combinar por config_key
    print(f"\n{'='*80}")
    print("COMBINANDO RESULTADOS")
    print(f"{'='*80}")
    
    if df_finetuning.empty and df_embeddings.empty:
        print("[ERROR] No se encontraron datos para procesar")
        return
    
    # Merge
    if not df_finetuning.empty and not df_embeddings.empty:
        df_combined = pd.merge(
            df_finetuning,
            df_embeddings,
            on='config_key',
            how='outer',
            suffixes=('_ft', '_emb')
        )
        
        # Consolidar columnas duplicadas
        for col in ['model_name', 'objective', 'component_type', 'num_layers', 'views_str']:
            col_ft = f'{col}_ft'
            col_emb = f'{col}_emb'
            if col_ft in df_combined.columns and col_emb in df_combined.columns:
                df_combined[col] = df_combined[col_ft].combine_first(df_combined[col_emb])
                df_combined = df_combined.drop(columns=[col_ft, col_emb])
        
        print(f"Combinado: {len(df_combined)} configuraciones únicas")
        
    elif not df_finetuning.empty:
        df_combined = df_finetuning
        print("Solo finetuning disponible")
    else:
        df_combined = df_embeddings
        print("Solo embeddings disponible")
    
    # Ordenar por componente y capas
    if 'component_type' in df_combined.columns and 'num_layers' in df_combined.columns:
        df_combined = df_combined.sort_values(['component_type', 'num_layers'])
    
    # Guardar CSVs
    output_dir = Path('results/analysis')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("GUARDANDO RESULTADOS")
    print(f"{'='*80}")
    
    # CSV combinado
    combined_csv = output_dir / 'clip_layers_results.csv'
    df_combined.to_csv(combined_csv, index=False)
    print(f"\n[OK] {combined_csv}")
    print(f"     Total de configuraciones: {len(df_combined)}")
    
    # CSVs separados (opcional)
    if not df_finetuning.empty:
        ft_csv = output_dir / 'clip_layers_finetuning_results.csv'
        df_finetuning.to_csv(ft_csv, index=False)
        print(f"[OK] {ft_csv}")
    
    if not df_embeddings.empty:
        emb_csv = output_dir / 'clip_layers_embeddings_results.csv'
        df_embeddings.to_csv(emb_csv, index=False)
        print(f"[OK] {emb_csv}")
    
    # Resumen
    print(f"\n{'='*80}")
    print("RESUMEN")
    print(f"{'='*80}")
    
    if not df_combined.empty:
        print(f"\nComponentes procesados:")
        if 'component_type' in df_combined.columns:
            for comp in df_combined['component_type'].unique():
                if pd.notna(comp):
                    count = len(df_combined[df_combined['component_type'] == comp])
                    print(f"  - {comp}: {count} configuraciones")
        
        print(f"\nRango de capas:")
        if 'num_layers' in df_combined.columns:
            layers = df_combined['num_layers'].dropna().unique()
            if len(layers) > 0:
                print(f"  - De {int(min(layers))} a {int(max(layers))} capas")
                print(f"  - Total de configuraciones distintas: {len(layers)}")
    
    print(f"\n{'='*80}")
    print("EXTRACCIÓN COMPLETADA")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

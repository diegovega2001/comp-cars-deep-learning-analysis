"""
Script para visualizar el historial de entrenamiento (training history).
Genera gráficos de losses y métricas por época para cada experimento.
CORREGIDO: Detecta automáticamente si usar accuracy o recall según el objetivo.
"""

import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_training_history(experiment_dir: Path) -> Optional[Dict]:
    """Carga el historial de entrenamiento desde training_history.pt o results.json."""
    # Intentar cargar desde training_history.pt
    history_file = experiment_dir / 'training_history.pt'
    if history_file.exists():
        try:
            history = torch.load(history_file, map_location='cpu')
            return history
        except Exception as e:
            print(f"    [WARNING] Error cargando training_history.pt: {e}")
    
    # Intentar cargar desde results.json
    results_file = experiment_dir / 'results.json'
    if results_file.exists() and results_file.stat().st_size < 10 * 1024 * 1024:  # < 10MB
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                finetuning = results.get('finetuning', {})
                history = finetuning.get('training_history', {})
                if history:
                    return history
        except Exception as e:
            print(f"    [WARNING] Error cargando results.json: {e}")
    
    return None


def flatten_history_if_needed(history: Dict) -> Dict:
    """Aplana el historial si tiene fases (CLIP)."""
    if any(k.startswith('phase_') for k in history.keys()):
        combined_history = {}
        phases = sorted([k for k in history.keys() if k.startswith('phase_')])
        
        for phase in phases:
            phase_history = history[phase]
            for key, value in phase_history.items():
                if key not in combined_history:
                    combined_history[key] = []
                if isinstance(value, list):
                    combined_history[key].extend(value)
                else:
                    combined_history[key].append(value)
        return combined_history
    return history


def load_config(experiment_dir: Path) -> Optional[Dict]:
    """Carga la configuración del experimento."""
    config_file = experiment_dir / 'config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"    [WARNING] Error cargando config.json: {e}")
    return None


def plot_training_history(experiment_dir: Path, output_dir: Path):
    """Genera gráficos del historial de entrenamiento para un experimento."""
    experiment_name = experiment_dir.name
    
    # Cargar historial y configuración
    history = load_training_history(experiment_dir)
    config = load_config(experiment_dir)
    
    if history is None:
        print(f"  ⚠ No se encontró historial de entrenamiento para {experiment_name}")
        return False
    
    # Aplanar historial si es necesario (por fases)
    history = flatten_history_if_needed(history)
    
    # Extraer información de configuración
    model_name = config.get('model_name', 'unknown') if config else 'unknown'
    objective = config.get('objective', 'unknown') if config else 'unknown'
    criterion = config.get('finetune_criterion', 'unknown') if config else 'unknown'
    
    # Detectar qué métricas están disponibles
    has_loss = 'train_loss' in history and len(history['train_loss']) > 0
    has_val_loss = 'val_loss' in history and len(history['val_loss']) > 0
    has_accuracy = 'val_accuracy' in history and len(history['val_accuracy']) > 0
    has_recall1 = 'val_recall@1' in history and len(history['val_recall@1']) > 0
    has_recall3 = 'val_recall@3' in history and len(history['val_recall@3']) > 0
    has_recall5 = 'val_recall@5' in history and len(history['val_recall@5']) > 0
    
    # Determinar qué gráficos crear basándose en el OBJETIVO
    # REGLA PRINCIPAL: usar objetivo para decidir qué métrica graficar
    # - classification → accuracy
    # - metric_learning → recall
    # - CLIP → recall
    metrics_to_plot = []
    
    if has_loss:
        metrics_to_plot.append('loss')
    
    # Decidir basado en el objetivo
    if objective == 'classification':
        # Classification siempre usa accuracy
        if has_accuracy:
            metrics_to_plot.append('accuracy')
    else:
        # metric_learning y CLIP usan recall
        if has_recall1 or has_recall3 or has_recall5:
            metrics_to_plot.append('recall')
    
    if not metrics_to_plot:
        print(f"  ⚠ No hay métricas para graficar en {experiment_name}")
        return False
    
    # Determinar layout
    n_plots = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # 1. Gráfico de Loss
    if 'loss' in metrics_to_plot:
        ax = axes[plot_idx]
        plot_idx += 1
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3, linewidth=2, color='#FF6B6B')
        if has_val_loss:
            ax.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3, linewidth=2, color='#4ECDC4')
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Gráfico de Accuracy (solo para classification)
    if 'accuracy' in metrics_to_plot:
        ax = axes[plot_idx]
        plot_idx += 1
        
        epochs = list(range(1, len(history['val_accuracy']) + 1))
        ax.plot(epochs, history['val_accuracy'], label='Val Accuracy', marker='s', markersize=3, linewidth=2, color='#2E86AB')
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Gráfico de Recall (para metric_learning y CLIP)
    if 'recall' in metrics_to_plot:
        ax = axes[plot_idx]
        plot_idx += 1
        
        # Usar la métrica más larga disponible para epochs
        recall_lengths = []
        if has_recall1:
            recall_lengths.append(len(history['val_recall@1']))
        if has_recall3:
            recall_lengths.append(len(history['val_recall@3']))
        if has_recall5:
            recall_lengths.append(len(history['val_recall@5']))
        
        max_length = max(recall_lengths) if recall_lengths else 0
        epochs = list(range(1, max_length + 1))
        
        if has_recall1:
            ax.plot(epochs[:len(history['val_recall@1'])], history['val_recall@1'], 
                   label='Recall@1', marker='o', markersize=3, linewidth=2, color='#2E86AB')
        if has_recall3:
            ax.plot(epochs[:len(history['val_recall@3'])], history['val_recall@3'], 
                   label='Recall@3', marker='s', markersize=3, linewidth=2, color='#A23B72')
        if has_recall5:
            ax.plot(epochs[:len(history['val_recall@5'])], history['val_recall@5'], 
                   label='Recall@5', marker='^', markersize=3, linewidth=2, color='#F18F01')
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Recall', fontsize=11, fontweight='bold')
        ax.set_title('Validation Recall Metrics', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Título general con información útil
    title_parts = [model_name]
    if objective and objective != 'unknown':
        title_parts.append(objective)
    if criterion and criterion != 'unknown':
        title_parts.append(criterion)
    
    title = ' - '.join(title_parts)
    fig.suptitle(f'{title}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Crear nombre de archivo único y descriptivo
    # Extraer información del path del experimento para crear nombre único
    path_parts = experiment_dir.parts
    
    # Buscar componentes clave en el path
    views_str = 'unknown'
    loss_str = ''
    component_str = ''
    layers_str = ''
    
    # Determinar si es CLIP layers study
    if 'layers_study' in path_parts:
        # CLIP layers: extraer component (vision/text) y layers
        for i, part in enumerate(path_parts):
            if part in ['vision', 'text']:
                component_str = part
            if 'layer' in part.lower():
                layers_str = part.replace(' ', '_')
        views_str = experiment_dir.name
        if component_str and layers_str:
            filename = f"{layers_str}_{component_str}_{model_name}_CLIP_training.png"
        else:
            filename = f"{experiment_dir.name}_{model_name}_CLIP_training.png"
    else:
        # Modelos de visión: extraer loss function si existe
        if 'metric learning' in path_parts:
            for i, part in enumerate(path_parts):
                if part in ['arcface', 'contrastive', 'multisimilarity', 'ntxent', 'triplet']:
                    loss_str = part
        
        # Extraer vistas del nombre del directorio
        views_str = experiment_dir.name.replace(' ', '_').replace('+', 'plus')
        
        # Construir nombre
        name_parts = [views_str, model_name]
        if objective and objective != 'unknown':
            name_parts.append(objective)
        if loss_str:
            name_parts.append(loss_str)
        elif criterion and criterion != 'unknown' and criterion != 'CrossEntropyLoss':
            name_parts.append(criterion.lower())
        
        filename = '_'.join(name_parts) + '_training.png'
    
    # Limpiar caracteres especiales
    filename = filename.replace('␠', '').replace(':', '-').replace('/', '_')
    
    output_file = output_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {filename}")
    return True


def main():
    """Función principal."""
    results_dir = Path('results')
    
    if not results_dir.exists():
        print(f"Error: No existe el directorio {results_dir}")
        return
    
    print("=" * 80)
    print("VISUALIZACIÓN DE HISTORIAL DE ENTRENAMIENTO")
    print("=" * 80)
    
    # Crear directorio de salida
    output_dir = Path('results/visualizations/training_history_plots')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nGenerando gráficos en: {output_dir}")
    
    # Buscar experimentos de finetuning en toda la estructura
    print("\nBuscando experimentos de finetuning...")
    finetune_folders = []
    
    # Buscar en modelos de visión (resnet50, vitb32)
    models_dir = results_dir / 'models'
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and model_dir.name in ['resnet50', 'vitb32']:
                # Classification
                class_ft = model_dir / 'classification' / 'finetuning'
                if class_ft.exists():
                    for view_dir in class_ft.iterdir():
                        if view_dir.is_dir():
                            finetune_folders.append(view_dir)
                
                # Metric learning
                ml_dir = model_dir / 'metric learning'
                if ml_dir.exists():
                    for loss_dir in ml_dir.iterdir():
                        if loss_dir.is_dir():
                            loss_ft = loss_dir / 'finetuning'
                            if loss_ft.exists():
                                for view_dir in loss_ft.iterdir():
                                    if view_dir.is_dir():
                                        finetune_folders.append(view_dir)
            
            # CLIP layers study
            elif model_dir.name == 'clip':
                layers_study = model_dir / 'layers_study' / 'finetuning'
                if layers_study.exists():
                    for component_dir in layers_study.iterdir():
                        if component_dir.is_dir() and component_dir.name in ['vision', 'text']:
                            for layer_dir in component_dir.iterdir():
                                if layer_dir.is_dir():
                                    finetune_folders.append(layer_dir)
    
    print(f"\nEncontrados {len(finetune_folders)} experimentos de finetuning")
    print("-" * 80)
    
    # Generar gráficos individuales
    success_count = 0
    for folder in sorted(finetune_folders):
        result = plot_training_history(folder, output_dir)
        if result:
            success_count += 1
    
    print("-" * 80)
    print(f"\nGráficos generados exitosamente: {success_count}/{len(finetune_folders)}")
    
    print("\n" + "=" * 80)
    print("VISUALIZACIÓN COMPLETADA")
    print("=" * 80)
    print(f"\nArchivos guardados en: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

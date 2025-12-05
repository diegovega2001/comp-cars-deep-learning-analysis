"""
Script para visualizar el historial de entrenamiento CLIP por capas.
Genera gráficos de curvas de entrenamiento comparando diferentes configuraciones de capas.
"""

import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def extract_clip_layer_info(config: Dict) -> Dict:
    """Extrae información de capas CLIP desde configuración."""
    phases = config.get('clip_finetuning_phases', {})
    
    if not phases:
        return {'component_type': None, 'num_layers': None}
    
    phase_name = list(phases.keys())[0]
    phase_config = phases[phase_name]
    
    component_type = phase_config.get('type')
    
    if component_type == 'vision':
        num_layers = phase_config.get('num_vision_layers')
    elif component_type == 'text':
        num_layers = phase_config.get('num_text_layers')
    else:
        num_layers = None
    
    return {
        'component_type': component_type,
        'num_layers': num_layers,
        'phase_name': phase_name
    }


def load_training_history(experiment_dir: Path) -> Optional[Dict]:
    """Carga el historial de entrenamiento."""
    # Intentar cargar desde training_history.pt
    history_file = experiment_dir / 'training_history.pt'
    if history_file.exists():
        try:
            return torch.load(history_file, map_location='cpu')
        except Exception as e:
            print(f"    [!] Error cargando training_history.pt: {e}")
    
    # Intentar cargar desde results.json
    results_file = experiment_dir / 'results.json'
    if results_file.exists() and results_file.stat().st_size < 10 * 1024 * 1024:
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results.get('training_history')
        except Exception as e:
            print(f"    [!] Error cargando results.json: {e}")
    
    return None


def flatten_phase_history(history: Dict, phase_name: str) -> Dict:
    """Extrae y aplana el historial de una fase específica."""
    if phase_name in history:
        return history[phase_name]
    
    # Si no hay fases, retornar todo
    if not any(k.startswith('phase_') for k in history.keys()):
        return history
    
    return {}


def load_config(experiment_dir: Path) -> Optional[Dict]:
    """Carga la configuración del experimento."""
    config_file = experiment_dir / 'config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"    [!] Error cargando config.json: {e}")
    return None


def collect_all_histories(results_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Recopila historiales de entrenamiento de todos los experimentos CLIP."""
    vision_histories = []
    text_histories = []
    
    finetune_folders = sorted([f for f in results_dir.iterdir() 
                               if f.is_dir() and f.name.startswith('finetune_')])
    
    print(f"\nRecopilando historiales de {len(finetune_folders)} experimentos...")
    
    for folder in finetune_folders:
        config = load_config(folder)
        if not config:
            continue
        
        layer_info = extract_clip_layer_info(config)
        if not layer_info['component_type'] or not layer_info['num_layers']:
            continue
        
        history = load_training_history(folder)
        if not history:
            continue
        
        # Extraer historial de la fase específica
        phase_history = flatten_phase_history(history, layer_info['phase_name'])
        
        if not phase_history:
            continue
        
        record = {
            'experiment_name': folder.name,
            'component_type': layer_info['component_type'],
            'num_layers': layer_info['num_layers'],
            'history': phase_history
        }
        
        if layer_info['component_type'] == 'vision':
            vision_histories.append(record)
        elif layer_info['component_type'] == 'text':
            text_histories.append(record)
        
        print(f"  [OK] {folder.name}: {layer_info['component_type']} - {layer_info['num_layers']} capas")
    
    return vision_histories, text_histories


def plot_training_curves_by_component(histories: List[Dict], component: str, output_dir: Path):
    """Grafica curvas de entrenamiento para un componente (visión o texto)."""
    if not histories:
        print(f"  [!] No hay historiales para {component}")
        return
    
    # Ordenar por número de capas
    histories = sorted(histories, key=lambda x: x['num_layers'])
    
    # Determinar métricas disponibles
    sample_history = histories[0]['history']
    has_loss = 'train_loss' in sample_history and 'val_loss' in sample_history
    has_accuracy = 'val_accuracy' in sample_history
    has_recall1 = 'val_recall@1' in sample_history
    has_recall3 = 'val_recall@3' in sample_history
    has_recall5 = 'val_recall@5' in sample_history
    
    # Configurar subplots
    n_plots = sum([has_loss, has_accuracy, has_recall1, has_recall3, has_recall5])
    if n_plots == 0:
        print(f"  [!] No hay métricas para graficar en {component}")
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Colores para diferentes números de capas
    colors = plt.cm.viridis(np.linspace(0, 1, len(histories)))
    
    plot_idx = 0
    
    # 1. Loss
    if has_loss:
        ax = axes[plot_idx]
        plot_idx += 1
        
        for idx, record in enumerate(histories):
            history = record['history']
            num_layers = record['num_layers']
            epochs = list(range(1, len(history['train_loss']) + 1))
            
            ax.plot(epochs, history['val_loss'], 
                   label=f'{num_layers} capas', linewidth=2, color=colors[idx])
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title(f'Loss - {component.upper()}', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 2. Accuracy
    if has_accuracy:
        ax = axes[plot_idx]
        plot_idx += 1
        
        for idx, record in enumerate(histories):
            history = record['history']
            num_layers = record['num_layers']
            epochs = list(range(1, len(history['val_accuracy']) + 1))
            
            ax.plot(epochs, history['val_accuracy'], 
                   label=f'{num_layers} capas', linewidth=2, color=colors[idx])
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Accuracy - {component.upper()}', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 3. Recall@1
    if has_recall1:
        ax = axes[plot_idx]
        plot_idx += 1
        
        for idx, record in enumerate(histories):
            history = record['history']
            num_layers = record['num_layers']
            epochs = list(range(1, len(history['val_recall@1']) + 1))
            
            ax.plot(epochs, history['val_recall@1'], 
                   label=f'{num_layers} capas', linewidth=2, color=colors[idx])
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation Recall@1 (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Recall@1 - {component.upper()}', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 4. Recall@3
    if has_recall3:
        ax = axes[plot_idx]
        plot_idx += 1
        
        for idx, record in enumerate(histories):
            history = record['history']
            num_layers = record['num_layers']
            epochs = list(range(1, len(history['val_recall@3']) + 1))
            
            ax.plot(epochs, history['val_recall@3'], 
                   label=f'{num_layers} capas', linewidth=2, color=colors[idx])
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation Recall@3 (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Recall@3 - {component.upper()}', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 5. Recall@5
    if has_recall5:
        ax = axes[plot_idx]
        plot_idx += 1
        
        for idx, record in enumerate(histories):
            history = record['history']
            num_layers = record['num_layers']
            epochs = list(range(1, len(history['val_recall@5']) + 1))
            
            ax.plot(epochs, history['val_recall@5'], 
                   label=f'{num_layers} capas', linewidth=2, color=colors[idx])
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation Recall@5 (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Recall@5 - {component.upper()}', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_{component}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] training_curves_{component}.png")


def plot_final_metrics_comparison(vision_histories: List[Dict], text_histories: List[Dict], output_dir: Path):
    """Compara métricas finales entre configuraciones de visión y texto."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    def extract_final_metrics(histories):
        """Extrae métricas finales de historiales."""
        data = []
        for record in histories:
            history = record['history']
            metrics = {
                'num_layers': record['num_layers'],
                'component_type': record['component_type'],
            }
            
            if 'val_loss' in history and history['val_loss']:
                metrics['final_val_loss'] = history['val_loss'][-1]
                metrics['best_val_loss'] = min(history['val_loss'])
            
            if 'val_accuracy' in history and history['val_accuracy']:
                metrics['final_val_accuracy'] = history['val_accuracy'][-1]
                metrics['best_val_accuracy'] = max(history['val_accuracy'])
            
            if 'val_recall@1' in history and history['val_recall@1']:
                metrics['final_recall1'] = history['val_recall@1'][-1]
                metrics['best_recall1'] = max(history['val_recall@1'])
            
            if 'val_recall@3' in history and history['val_recall@3']:
                metrics['final_recall3'] = history['val_recall@3'][-1]
                metrics['best_recall3'] = max(history['val_recall@3'])
            
            if 'val_recall@5' in history and history['val_recall@5']:
                metrics['final_recall5'] = history['val_recall@5'][-1]
                metrics['best_recall5'] = max(history['val_recall@5'])
            
            data.append(metrics)
        
        return pd.DataFrame(data)
    
    vision_df = extract_final_metrics(vision_histories)
    text_df = extract_final_metrics(text_histories)
    
    # 1. Best Validation Loss
    ax = axes[0, 0]
    if 'best_val_loss' in vision_df.columns and 'best_val_loss' in text_df.columns:
        ax.plot(vision_df['num_layers'], vision_df['best_val_loss'], 
               marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Visión')
        ax.plot(text_df['num_layers'], text_df['best_val_loss'], 
               marker='s', linewidth=2, markersize=8, color='#A23B72', label='Texto')
        ax.set_xlabel('Número de Capas', fontsize=11, fontweight='bold')
        ax.set_ylabel('Best Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Mejor Loss por Capas', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Best Recall@3
    ax = axes[0, 1]
    if 'best_recall3' in vision_df.columns and 'best_recall3' in text_df.columns:
        ax.plot(vision_df['num_layers'], vision_df['best_recall3'], 
               marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Visión')
        ax.plot(text_df['num_layers'], text_df['best_recall3'], 
               marker='s', linewidth=2, markersize=8, color='#A23B72', label='Texto')
        ax.set_xlabel('Número de Capas', fontsize=11, fontweight='bold')
        ax.set_ylabel('Best Validation Recall@3 (%)', fontsize=11, fontweight='bold')
        ax.set_title('Mejor Recall@3 por Capas', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    elif 'best_val_accuracy' in vision_df.columns and 'best_val_accuracy' in text_df.columns:
        ax.plot(vision_df['num_layers'], vision_df['best_val_accuracy'], 
               marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Visión')
        ax.plot(text_df['num_layers'], text_df['best_val_accuracy'], 
               marker='s', linewidth=2, markersize=8, color='#A23B72', label='Texto')
        ax.set_xlabel('Número de Capas', fontsize=11, fontweight='bold')
        ax.set_ylabel('Best Validation Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title('Mejor Accuracy por Capas', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Best Recall@1
    ax = axes[1, 0]
    if 'best_recall1' in vision_df.columns and 'best_recall1' in text_df.columns:
        ax.plot(vision_df['num_layers'], vision_df['best_recall1'], 
               marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Visión')
        ax.plot(text_df['num_layers'], text_df['best_recall1'], 
               marker='s', linewidth=2, markersize=8, color='#A23B72', label='Texto')
        ax.set_xlabel('Número de Capas', fontsize=11, fontweight='bold')
        ax.set_ylabel('Best Validation Recall@1 (%)', fontsize=11, fontweight='bold')
        ax.set_title('Mejor Recall@1 por Capas', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Tabla resumen
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "RESUMEN DE MÉTRICAS FINALES\n" + "="*35 + "\n\n"
    
    if 'best_recall1' in vision_df.columns:
        v_best_r1 = vision_df.loc[vision_df['best_recall1'].idxmax()]
        t_best_r1 = text_df.loc[text_df['best_recall1'].idxmax()]
        
        summary_text += f"MEJOR RECALL@1:\n"
        summary_text += f"  Visión: {v_best_r1['num_layers']:.0f} capas\n"
        summary_text += f"          {v_best_r1['best_recall1']:.2f}%\n\n"
        summary_text += f"  Texto: {t_best_r1['num_layers']:.0f} capas\n"
        summary_text += f"         {t_best_r1['best_recall1']:.2f}%\n\n"
    
    if 'best_recall5' in vision_df.columns:
        v_best_r5 = vision_df.loc[vision_df['best_recall5'].idxmax()]
        t_best_r5 = text_df.loc[text_df['best_recall5'].idxmax()]
        
        summary_text += f"MEJOR RECALL@5:\n"
        summary_text += f"  Visión: {v_best_r5['num_layers']:.0f} capas\n"
        summary_text += f"          {v_best_r5['best_recall5']:.2f}%\n\n"
        summary_text += f"  Texto: {t_best_r5['num_layers']:.0f} capas\n"
        summary_text += f"         {t_best_r5['best_recall5']:.2f}%\n\n"
    
    if 'best_recall1' in vision_df.columns:
        summary_text += f"PROMEDIO RECALL@1:\n"
        summary_text += f"  Visión: {vision_df['best_recall1'].mean():.2f}%\n"
        summary_text += f"  Texto:  {text_df['best_recall1'].mean():.2f}%\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
           fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] final_metrics_comparison.png")


def plot_convergence_analysis(vision_histories: List[Dict], text_histories: List[Dict], output_dir: Path):
    """Analiza velocidad de convergencia por número de capas."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    def calculate_epochs_to_converge(histories, threshold_pct=0.95):
        """Calcula épocas necesarias para alcanzar el 95% del mejor Recall@1."""
        data = []
        for record in histories:
            history = record['history']
            
            # Try to use Recall@1 if available, fallback to accuracy
            metric_key = None
            if 'val_recall@1' in history and history['val_recall@1']:
                metric_key = 'val_recall@1'
            elif 'val_accuracy' in history and history['val_accuracy']:
                metric_key = 'val_accuracy'
            else:
                continue
            
            val_metric = history[metric_key]
            best_metric = max(val_metric)
            threshold = best_metric * threshold_pct
            
            epochs_to_threshold = None
            for epoch, metric_val in enumerate(val_metric, start=1):
                if metric_val >= threshold:
                    epochs_to_threshold = epoch
                    break
            
            if epochs_to_threshold:
                data.append({
                    'num_layers': record['num_layers'],
                    'component_type': record['component_type'],
                    'epochs_to_converge': epochs_to_threshold,
                    'total_epochs': len(val_metric),
                    'best_metric': best_metric
                })
        
        return pd.DataFrame(data)
    
    vision_conv = calculate_epochs_to_converge(vision_histories)
    text_conv = calculate_epochs_to_converge(text_histories)
    
    # 1. Épocas hasta convergencia
    ax = axes[0]
    if not vision_conv.empty and not text_conv.empty:
        ax.plot(vision_conv['num_layers'], vision_conv['epochs_to_converge'], 
               marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Visión')
        ax.plot(text_conv['num_layers'], text_conv['epochs_to_converge'], 
               marker='s', linewidth=2, markersize=8, color='#A23B72', label='Texto')
        ax.set_xlabel('Número de Capas', fontsize=11, fontweight='bold')
        ax.set_ylabel('Épocas hasta 95% del Mejor Recall@1', fontsize=11, fontweight='bold')
        ax.set_title('Velocidad de Convergencia', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Total de épocas ejecutadas
    ax = axes[1]
    if not vision_conv.empty and not text_conv.empty:
        ax.plot(vision_conv['num_layers'], vision_conv['total_epochs'], 
               marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Visión')
        ax.plot(text_conv['num_layers'], text_conv['total_epochs'], 
               marker='s', linewidth=2, markersize=8, color='#A23B72', label='Texto')
        ax.set_xlabel('Número de Capas', fontsize=11, fontweight='bold')
        ax.set_ylabel('Total de Épocas Ejecutadas', fontsize=11, fontweight='bold')
        ax.set_title('Duración de Entrenamiento (épocas)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] convergence_analysis.png")


def plot_learning_progression_heatmap(histories: List[Dict], component: str, output_dir: Path):
    """Heatmap de progresión de accuracy durante entrenamiento."""
    if not histories:
        print(f"  [!] No hay historiales para {component}")
        return
    
    # Preparar datos para heatmap
    max_epochs = max(len(h['history'].get('val_accuracy', [])) for h in histories)
    
    data_matrix = []
    layer_labels = []
    
    for record in sorted(histories, key=lambda x: x['num_layers']):
        if 'val_accuracy' not in record['history']:
            continue
        
        val_acc = record['history']['val_accuracy']
        # Rellenar con NaN si hay menos épocas
        padded_acc = val_acc + [np.nan] * (max_epochs - len(val_acc))
        data_matrix.append(padded_acc)
        layer_labels.append(f"{record['num_layers']:.0f} capas")
    
    if not data_matrix:
        print(f"  [!] No hay datos de accuracy para {component}")
        return
    
    data_matrix = np.array(data_matrix)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    ax.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuración', fontsize=12, fontweight='bold')
    ax.set_title(f'Progresión de Accuracy Durante Entrenamiento - {component.upper()}', 
                fontsize=14, fontweight='bold')
    
    ax.set_yticks(np.arange(len(layer_labels)))
    ax.set_yticklabels(layer_labels)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Validation Accuracy (%)', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'learning_progression_heatmap_{component}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] learning_progression_heatmap_{component}.png")


def main():
    """Función principal."""
    results_dir = Path('results')
    
    if not results_dir.exists():
        print(f"Error: No se encontró el directorio {results_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("VISUALIZACIÓN DE CURVAS DE ENTRENAMIENTO CLIP - POR CAPAS")
    print("=" * 80)
    
    # Recopilar historiales
    vision_histories, text_histories = collect_all_histories(results_dir)
    
    print(f"\n{'=' * 80}")
    print("RESUMEN DE DATOS")
    print(f"{'=' * 80}")
    print(f"Historiales de visión: {len(vision_histories)}")
    print(f"Historiales de texto:  {len(text_histories)}")
    
    if not vision_histories and not text_histories:
        print("\nNo se encontraron historiales de entrenamiento.")
        sys.exit(0)
    
    # Crear directorio de salida
    output_dir = Path('results/clip_training_history_plots')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerando visualizaciones en: {output_dir}")
    print("-" * 80)
    
    # Generar gráficos
    if vision_histories:
        plot_training_curves_by_component(vision_histories, 'vision', output_dir)
        plot_learning_progression_heatmap(vision_histories, 'vision', output_dir)
    
    if text_histories:
        plot_training_curves_by_component(text_histories, 'text', output_dir)
        plot_learning_progression_heatmap(text_histories, 'text', output_dir)
    
    if vision_histories and text_histories:
        plot_final_metrics_comparison(vision_histories, text_histories, output_dir)
        plot_convergence_analysis(vision_histories, text_histories, output_dir)
    
    print("-" * 80)
    print(f"\n{'=' * 80}")
    print("VISUALIZACIÓN COMPLETADA")
    print(f"{'=' * 80}")
    print(f"\nArchivos guardados en: {output_dir.absolute()}")


if __name__ == '__main__':
    main()


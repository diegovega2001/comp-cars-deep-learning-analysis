"""
Script de análisis de embeddings optimizado.

Este script ejecuta el análisis de embeddings de forma inteligente:
- Para modelos de clasificación: ejecuta análisis full (baseline + finetuned) como referencia
- Para modelos de metric learning: solo ejecuta análisis finetuned y compara con el baseline de clasificación
- Para CLIP: ejecuta análisis full ya que tiene su propio baseline
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from src.pipeline.EmbeddingsPipeline import create_embeddings_pipeline

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingsAnalysisRunner:
    """
    Runner para análisis de embeddings con lógica inteligente.
    
    Detecta automáticamente qué tipo de análisis ejecutar basándose en:
    - Si es clasificación o metric learning
    - Si ya existe un análisis baseline de referencia
    """
    
    def __init__(
        self,
        dataset_path: str = "dataset.csv",
        embeddings_config_path: str = "configs/embeddings.yaml",
        results_base_dir: str = "results/models"
    ):
        self.dataset_path = Path(dataset_path)
        self.embeddings_config_path = Path(embeddings_config_path)
        self.results_base_dir = Path(results_base_dir)
        
        # Cargar dataset y configuración
        self.dataset_df = self._load_dataset()
        self.embeddings_config = self._load_embeddings_config()
        
        # Tracking de resultados
        self.successful_analyses: List[str] = []
        self.failed_analyses: List[Tuple[str, str]] = []
        self.comparisons_made: List[Dict[str, Any]] = []
    
    def _load_dataset(self) -> pd.DataFrame:
        """Carga el dataset CSV."""
        logger.info(f"Cargando dataset desde: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        logger.info(f"Dataset cargado: {len(df)} registros")
        return df
    
    def _load_embeddings_config(self) -> Dict[str, Any]:
        """Carga la configuración de embeddings."""
        logger.info(f"Cargando configuración desde: {self.embeddings_config_path}")
        with open(self.embeddings_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Configuración de embeddings cargada")
        return config
    
    def _load_finetuning_config(self, finetuning_dir: Path) -> Dict[str, Any]:
        """Carga la configuración del fine-tuning."""
        config_path = finetuning_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No se encontró config.json en {finetuning_dir}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_merged_config(self, finetuning_dir: Path) -> Dict[str, Any]:
        """Fusiona la configuración del fine-tuning con la de embeddings."""
        finetuning_config = self._load_finetuning_config(finetuning_dir)
        # embeddings_config tiene prioridad
        return {**finetuning_config, **self.embeddings_config}
    
    def _save_results_to_embeddings_dir(
        self,
        pipeline,
        embeddings_dir: Path,
        analysis_type: str
    ) -> None:
        """
        Guarda los resultados del pipeline en el directorio de embeddings.
        
        Args:
            pipeline: Pipeline de embeddings con resultados
            embeddings_dir: Directorio donde guardar los resultados
            analysis_type: Tipo de análisis ('full', 'finetuned_only', 'baseline_only')
        """
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar configuración
        config_path = embeddings_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline.config, f, indent=2, default=str)
        
        # Guardar resultados
        results_path = embeddings_dir / "results.json"
        
        # Convertir resultados a formato serializable
        results_to_save = self._make_json_serializable(pipeline.results.copy())
        results_to_save['analysis_type'] = analysis_type
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        # Guardar embeddings como numpy
        if pipeline.baseline_embeddings is not None:
            np.save(embeddings_dir / "baseline_embeddings.npy", pipeline.baseline_embeddings.numpy())
        
        if pipeline.finetuned_embeddings is not None:
            np.save(embeddings_dir / "finetuned_embeddings.npy", pipeline.finetuned_embeddings.numpy())
        
        if pipeline.baseline_labels is not None:
            np.save(embeddings_dir / "labels.npy", pipeline.baseline_labels.numpy())
        
        # Guardar comparación si existe
        if 'comparison' in pipeline.results and 'performance_improvement' in pipeline.results['comparison']:
            comparison_df = pd.DataFrame([pipeline.results['comparison']['performance_improvement']])
            comparison_df.to_csv(embeddings_dir / "performance_comparison.csv", index=False)
        
        # Guardar visualizaciones
        vis_dir = embeddings_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar visualizaciones de baseline
        if hasattr(pipeline, 'baseline_visualizer') and pipeline.baseline_visualizer is not None:
            try:
                baseline_vis_dir = vis_dir / "baseline"
                baseline_vis_dir.mkdir(parents=True, exist_ok=True)
                pipeline.baseline_visualizer.save_all_visualizations(str(baseline_vis_dir))
            except Exception as e:
                logger.warning(f"Error guardando visualizaciones baseline: {e}")
        
        # Guardar visualizaciones de finetuned
        if hasattr(pipeline, 'finetuned_visualizer') and pipeline.finetuned_visualizer is not None:
            try:
                finetuned_vis_dir = vis_dir / "finetuned"
                finetuned_vis_dir.mkdir(parents=True, exist_ok=True)
                pipeline.finetuned_visualizer.save_all_visualizations(str(finetuned_vis_dir))
            except Exception as e:
                logger.warning(f"Error guardando visualizaciones finetuned: {e}")
        
        logger.info(f"Resultados guardados en: {embeddings_dir}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convierte objetos a formato JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def run_full_analysis(
        self,
        finetuning_dir: Path,
        embeddings_dir: Path,
        experiment_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Ejecuta análisis FULL (baseline + finetuned + comparación).
        
        Args:
            finetuning_dir: Directorio con los embeddings .pt del fine-tuning
            embeddings_dir: Directorio donde guardar los resultados
            experiment_name: Nombre del experimento
            
        Returns:
            Resultados del análisis o None si falla
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ANÁLISIS FULL: {experiment_name}")
        logger.info(f"{'='*80}")
        
        try:
            # Verificar que existen los embeddings
            baseline_path = finetuning_dir / "baseline_embeddings.pt"
            finetuned_path = finetuning_dir / "finetuned_embeddings.pt"
            
            if not baseline_path.exists():
                raise FileNotFoundError(f"No se encontró baseline_embeddings.pt en {finetuning_dir}")
            if not finetuned_path.exists():
                raise FileNotFoundError(f"No se encontró finetuned_embeddings.pt en {finetuning_dir}")
            
            # Crear pipeline
            merged_config = self._get_merged_config(finetuning_dir)
            pipeline = create_embeddings_pipeline(
                config=merged_config,
                df=self.dataset_df,
                experiment_name=experiment_name
            )
            
            # Ejecutar análisis completo
            pipeline.create_dataset_for_labels()
            pipeline.load_embeddings_from_files(baseline_path, finetuned_path)
            pipeline.analyze_baseline_embeddings()
            pipeline.analyze_finetuned_embeddings()
            pipeline.compare_results()
            
            # Guardar resultados
            self._save_results_to_embeddings_dir(pipeline, embeddings_dir, 'full')
            
            self.successful_analyses.append(experiment_name)
            logger.info(f"✓ Análisis FULL completado: {experiment_name}")
            
            return pipeline.results
            
        except Exception as e:
            self.failed_analyses.append((experiment_name, str(e)))
            logger.error(f"✗ Error en análisis FULL {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_finetuned_only_analysis(
        self,
        finetuning_dir: Path,
        embeddings_dir: Path,
        experiment_name: str,
        baseline_results_dir: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Ejecuta análisis solo de FINETUNED y compara con baseline existente.
        
        Args:
            finetuning_dir: Directorio con los embeddings .pt del fine-tuning
            embeddings_dir: Directorio donde guardar los resultados
            experiment_name: Nombre del experimento
            baseline_results_dir: Directorio con resultados baseline para comparar
            
        Returns:
            Resultados del análisis o None si falla
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ANÁLISIS FINETUNED-ONLY: {experiment_name}")
        logger.info(f"{'='*80}")
        
        try:
            # Verificar que existe el embedding finetuned
            finetuned_path = finetuning_dir / "finetuned_embeddings.pt"
            
            if not finetuned_path.exists():
                raise FileNotFoundError(f"No se encontró finetuned_embeddings.pt en {finetuning_dir}")
            
            # Crear pipeline
            merged_config = self._get_merged_config(finetuning_dir)
            pipeline = create_embeddings_pipeline(
                config=merged_config,
                df=self.dataset_df,
                experiment_name=experiment_name
            )
            
            # Crear dataset para labels y cargar solo embeddings finetuned
            pipeline.create_dataset_for_labels()
            
            # Cargar embeddings finetuned directamente
            pipeline.finetuned_embeddings = torch.load(finetuned_path, map_location='cpu')
            pipeline.finetuned_info = {
                'path': str(finetuned_path),
                'shape': list(pipeline.finetuned_embeddings.shape),
                'device': str(pipeline.finetuned_embeddings.device)
            }
            
            # Ejecutar solo análisis finetuned
            pipeline.analyze_finetuned_embeddings()
            
            # Si hay baseline de referencia, hacer comparación
            if baseline_results_dir and baseline_results_dir.exists():
                baseline_results_path = baseline_results_dir / "results.json"
                if baseline_results_path.exists():
                    logger.info(f"Comparando con baseline de: {baseline_results_dir}")
                    comparison = self._compare_with_baseline(
                        pipeline.results,
                        baseline_results_path,
                        experiment_name
                    )
                    if comparison:
                        pipeline.results['comparison'] = comparison
                        self.comparisons_made.append({
                            'experiment': experiment_name,
                            'baseline_source': str(baseline_results_dir),
                            'comparison': comparison
                        })
            
            # Guardar resultados
            self._save_results_to_embeddings_dir(pipeline, embeddings_dir, 'finetuned_only')
            
            self.successful_analyses.append(experiment_name)
            logger.info(f"✓ Análisis FINETUNED-ONLY completado: {experiment_name}")
            
            return pipeline.results
            
        except Exception as e:
            self.failed_analyses.append((experiment_name, str(e)))
            logger.error(f"✗ Error en análisis FINETUNED-ONLY {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compare_with_baseline(
        self,
        finetuned_results: Dict[str, Any],
        baseline_results_path: Path,
        experiment_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Compara resultados finetuned con baseline existente.
        
        Args:
            finetuned_results: Resultados del análisis finetuned
            baseline_results_path: Ruta al archivo results.json del baseline
            experiment_name: Nombre del experimento
            
        Returns:
            Diccionario con la comparación o None si falla
        """
        try:
            with open(baseline_results_path, 'r', encoding='utf-8') as f:
                baseline_results = json.load(f)
            
            if 'baseline_analysis' not in baseline_results:
                logger.warning(f"No se encontró baseline_analysis en {baseline_results_path}")
                return None
            
            if 'finetuned_analysis' not in finetuned_results:
                logger.warning("No se encontró finetuned_analysis en los resultados actuales")
                return None
            
            baseline_analysis = baseline_results['baseline_analysis']
            finetuned_analysis = finetuned_results['finetuned_analysis']
            
            # Construir comparación
            comparison = {
                'baseline_source': str(baseline_results_path),
                'reduction_methods': {
                    'baseline_best': baseline_analysis['reduction']['best_method'],
                    'finetuned_best': finetuned_analysis['reduction']['best_method']
                },
                'clustering_methods': {
                    'baseline_best': baseline_analysis['clustering']['best_method'],
                    'finetuned_best': finetuned_analysis['clustering']['best_method']
                }
            }
            
            # Comparar métricas de clustering
            baseline_clustering = baseline_analysis['clustering'].get('comparison_df', [])
            finetuned_clustering = finetuned_analysis['clustering'].get('comparison_df', [])
            
            if baseline_clustering and finetuned_clustering:
                baseline_best = max(baseline_clustering, key=lambda x: x.get('adjusted_rand_score', 0))
                finetuned_best = max(finetuned_clustering, key=lambda x: x.get('adjusted_rand_score', 0))
                
                comparison['performance_improvement'] = {
                    'baseline_ari': baseline_best.get('adjusted_rand_score', 0),
                    'finetuned_ari': finetuned_best.get('adjusted_rand_score', 0),
                    'ari_improvement': finetuned_best.get('adjusted_rand_score', 0) - baseline_best.get('adjusted_rand_score', 0),
                    'baseline_silhouette': baseline_best.get('silhouette_score', 0),
                    'finetuned_silhouette': finetuned_best.get('silhouette_score', 0),
                    'silhouette_improvement': finetuned_best.get('silhouette_score', 0) - baseline_best.get('silhouette_score', 0),
                    'baseline_nmi': baseline_best.get('normalized_mutual_info', 0),
                    'finetuned_nmi': finetuned_best.get('normalized_mutual_info', 0),
                    'nmi_improvement': finetuned_best.get('normalized_mutual_info', 0) - baseline_best.get('normalized_mutual_info', 0)
                }
                
                logger.info(f"Comparación completada para {experiment_name}")
                perf = comparison['performance_improvement']
                logger.info(f"  ARI: {perf['baseline_ari']:.4f} → {perf['finetuned_ari']:.4f} (Δ {perf['ari_improvement']:+.4f})")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparando con baseline: {e}")
            return None
    
    def process_vision_model(self, model_name: str) -> None:
        """
        Procesa un modelo de visión (resnet50 o vitb32).
        
        Para classification: ejecuta análisis FULL
        Para metric learning: ejecuta análisis FINETUNED-ONLY y compara con classification
        
        Args:
            model_name: Nombre del modelo (resnet50, vitb32)
        """
        model_dir = self.results_base_dir / model_name
        
        if not model_dir.exists():
            logger.warning(f"Directorio no encontrado: {model_dir}")
            return
        
        logger.info(f"\n{'#'*80}")
        logger.info(f"PROCESANDO MODELO: {model_name.upper()}")
        logger.info(f"{'#'*80}")
        
        # 1. Procesar CLASSIFICATION primero (será el baseline de referencia)
        classification_dir = model_dir / "classification"
        if classification_dir.exists():
            self._process_classification(model_name, classification_dir)
        
        # 2. Procesar METRIC LEARNING (usando baseline de classification)
        metric_learning_dir = model_dir / "metric learning"
        if metric_learning_dir.exists():
            self._process_metric_learning(model_name, metric_learning_dir, classification_dir)
    
    def _process_classification(self, model_name: str, classification_dir: Path) -> None:
        """Procesa los experimentos de clasificación (análisis FULL)."""
        logger.info(f"\n{'='*60}")
        logger.info(f"CLASSIFICATION - {model_name}")
        logger.info(f"{'='*60}")
        
        finetuning_dir = classification_dir / "finetuning"
        embeddings_base_dir = classification_dir / "embeddings"
        
        if not finetuning_dir.exists():
            logger.warning(f"No se encontró directorio de finetuning: {finetuning_dir}")
            return
        
        # Procesar cada vista (front, front + rear, etc.)
        for view_dir in finetuning_dir.iterdir():
            if view_dir.is_dir():
                view_name = view_dir.name
                experiment_name = f"{model_name}_classification_{view_name}"
                embeddings_dir = embeddings_base_dir / view_name
                
                self.run_full_analysis(view_dir, embeddings_dir, experiment_name)
    
    def _process_metric_learning(
        self,
        model_name: str,
        metric_learning_dir: Path,
        classification_dir: Path
    ) -> None:
        """Procesa los experimentos de metric learning (análisis FINETUNED-ONLY)."""
        logger.info(f"\n{'='*60}")
        logger.info(f"METRIC LEARNING - {model_name}")
        logger.info(f"{'='*60}")
        
        # Iterar por cada loss function (arcface, triplet, etc.)
        for loss_dir in metric_learning_dir.iterdir():
            if not loss_dir.is_dir():
                continue
            
            loss_name = loss_dir.name
            finetuning_dir = loss_dir / "finetuning"
            embeddings_base_dir = loss_dir / "embeddings"
            
            if not finetuning_dir.exists():
                logger.warning(f"No se encontró directorio de finetuning: {finetuning_dir}")
                continue
            
            # Procesar cada vista
            for view_dir in finetuning_dir.iterdir():
                if view_dir.is_dir():
                    view_name = view_dir.name
                    experiment_name = f"{model_name}_{loss_name}_{view_name}"
                    embeddings_dir = embeddings_base_dir / view_name
                    
                    # Buscar el baseline de classification para esta vista
                    baseline_results_dir = classification_dir / "embeddings" / view_name
                    
                    self.run_finetuned_only_analysis(
                        view_dir,
                        embeddings_dir,
                        experiment_name,
                        baseline_results_dir
                    )
    
    def process_clip(self) -> None:
        """
        Procesa el modelo CLIP.
        
        CLIP tiene su propio baseline, así que ejecuta análisis FULL.
        """
        clip_dir = self.results_base_dir / "clip"
        
        if not clip_dir.exists():
            logger.warning(f"Directorio CLIP no encontrado: {clip_dir}")
            return
        
        logger.info(f"\n{'#'*80}")
        logger.info("PROCESANDO MODELO: CLIP")
        logger.info(f"{'#'*80}")
        
        finetuning_dir = clip_dir / "finetuning"
        embeddings_base_dir = clip_dir / "embeddings"
        
        if not finetuning_dir.exists():
            logger.warning(f"No se encontró directorio de finetuning: {finetuning_dir}")
            return
        
        # Procesar cada vista
        for view_dir in finetuning_dir.iterdir():
            if view_dir.is_dir():
                view_name = view_dir.name
                experiment_name = f"clip_{view_name}"
                embeddings_dir = embeddings_base_dir / view_name
                
                self.run_full_analysis(view_dir, embeddings_dir, experiment_name)
    
    def run_all(self) -> Dict[str, Any]:
        """
        Ejecuta el análisis completo para todos los modelos.
        
        Returns:
            Resumen de la ejecución
        """
        logger.info("="*80)
        logger.info("INICIANDO ANÁLISIS DE EMBEDDINGS - MODO INTELIGENTE")
        logger.info("="*80)
        logger.info("- Classification: Análisis FULL (baseline + finetuned)")
        logger.info("- Metric Learning: Análisis FINETUNED-ONLY + comparación")
        logger.info("- CLIP: Análisis FULL")
        logger.info("="*80)
        
        # Procesar cada modelo
        self.process_vision_model("resnet50")
        self.process_vision_model("vitb32")
        self.process_clip()
        
        # Generar resumen
        summary = self._generate_summary()
        
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Genera un resumen de la ejecución."""
        summary = {
            'total_experiments': len(self.successful_analyses) + len(self.failed_analyses),
            'successful': len(self.successful_analyses),
            'failed': len(self.failed_analyses),
            'successful_experiments': self.successful_analyses,
            'failed_experiments': self.failed_analyses,
            'comparisons_made': len(self.comparisons_made)
        }
        
        logger.info(f"\n{'='*80}")
        logger.info("RESUMEN DE EJECUCIÓN")
        logger.info(f"{'='*80}")
        logger.info(f"Total de experimentos: {summary['total_experiments']}")
        logger.info(f"Exitosos: {summary['successful']}")
        logger.info(f"Fallidos: {summary['failed']}")
        logger.info(f"Comparaciones realizadas: {summary['comparisons_made']}")
        
        if self.failed_analyses:
            logger.warning("\nExperimentos fallidos:")
            for name, error in self.failed_analyses:
                logger.warning(f"  - {name}: {error}")
        
        logger.info(f"\n{'='*80}")
        logger.info("ANÁLISIS DE EMBEDDINGS COMPLETADO")
        logger.info(f"{'='*80}")
        
        return summary


if __name__ == "__main__":
    try:
        runner = EmbeddingsAnalysisRunner(
            dataset_path="dataset.csv",
            embeddings_config_path="configs/embeddings.yaml",
            results_base_dir="results/models"
        )
        
        summary = runner.run_all()
        
        # Guardar resumen
        summary_path = Path("results/analysis/embeddings_analysis_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Resumen guardado en: {summary_path}")
        
        if summary['failed'] > 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

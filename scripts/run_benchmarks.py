#!/usr/bin/env python3
"""
Script de benchmark automático para o sistema de pastagens brasileiras
Executa testes de performance e compara com benchmarks científicos
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import logging
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importações do projeto
try:
    from src.diffusion.pipeline_manager import PipelineManager
    from src.dataset.generator import DatasetGenerator, GenerationConfig
    from src.training.yolo_trainer import YOLOTrainer
    from src.training.evaluation import ModelEvaluator
    from src.training.benchmark import ScientificBenchmarkSuite
    from src.dataset.quality_metrics import QualityMetrics
except ImportError as e:
    logger.error(f"Erro ao importar módulos: {e}")
    logger.error("Certifique-se de que está executando do diretório raiz do projeto")
    sys.exit(1)

class BenchmarkRunner:
    """
    Runner para executar benchmarks completos do sistema
    """
    
    def __init__(self, output_dir: str = "/tmp/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'system_info': {},
            'generation_benchmark': {},
            'training_benchmark': {},
            'evaluation_benchmark': {},
            'scientific_comparison': {},
            'overall_score': 0.0
        }
        
        # Coleta informações do sistema
        self._collect_system_info()
    
    def _collect_system_info(self):
        """Coleta informações do sistema para o benchmark"""
        
        import platform
        import psutil
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'cuda_version': torch.version.cuda
            })
        
        # Versões de bibliotecas
        try:
            import diffusers
            import ultralytics
            import transformers
            
            system_info.update({
                'torch_version': torch.__version__,
                'diffusers_version': diffusers.__version__,
                'ultralytics_version': ultralytics.__version__,
                'transformers_version': transformers.__version__
            })
        except ImportError:
            pass
        
        self.results['system_info'] = system_info
        
        logger.info("Sistema de benchmark:")
        logger.info(f"  GPU: {system_info.get('gpu_name', 'CPU only')}")
        logger.info(f"  Memória: {system_info['memory_gb']:.1f}GB")
        logger.info(f"  PyTorch: {system_info.get('torch_version', 'N/A')}")
    
    def benchmark_generation_performance(self, num_images: int = 10) -> dict:
        """
        Benchmark de performance de geração de imagens
        
        Args:
            num_images: Número de imagens para gerar
            
        Returns:
            Resultados do benchmark de geração
        """
        
        logger.info(f"🎨 Benchmark de geração ({num_images} imagens)")
        
        try:
            # Configuração otimizada para benchmark
            config = GenerationConfig(
                num_images=num_images,
                resolution=(512, 512),
                output_dir=str(self.output_dir / "benchmark_images"),
                num_inference_steps=20,
                guidance_scale_range=(7.5, 7.5),  # Fixo para consistência
                quality_threshold=0.5,  # Mais permissivo
                save_metadata=True
            )
            
            # Inicializar gerador
            generator = DatasetGenerator()
            
            # Medir tempo de inicialização
            init_start = time.time()
            pipeline = PipelineManager()
            init_time = time.time() - init_start
            
            # Medir tempo de geração
            gen_start = time.time()
            output_path = generator.generate_dataset(config, save_intermediate=False)
            total_gen_time = time.time() - gen_start
            
            # Calcular métricas
            avg_time_per_image = total_gen_time / num_images
            images_per_hour = 3600 / avg_time_per_image
            
            # Coletar estatísticas de qualidade
            quality_stats = generator.get_generation_statistics()
            
            generation_results = {
                'num_images_generated': num_images,
                'total_time_seconds': total_gen_time,
                'avg_time_per_image': avg_time_per_image,
                'images_per_hour': images_per_hour,
                'initialization_time': init_time,
                'success_rate': quality_stats.get('successful_generations', 0) / quality_stats.get('total_attempts', 1),
                'avg_quality_score': quality_stats.get('avg_quality_score', 0),
                'output_path': str(output_path) if output_path else None
            }
            
            self.results['generation_benchmark'] = generation_results
            
            logger.info(f"  ⏱️ Tempo médio: {avg_time_per_image:.2f}s/imagem")
            logger.info(f"  📈 Throughput: {images_per_hour:.0f} imagens/hora")
            logger.info(f"  ✅ Taxa de sucesso: {generation_results['success_rate']*100:.1f}%")
            
            return generation_results
            
        except Exception as e:
            logger.error(f"Erro no benchmark de geração: {e}")
            return {'error': str(e)}
    
    def benchmark_training_performance(self, dataset_path: str = None) -> dict:
        """
        Benchmark de performance de treinamento YOLO
        
        Args:
            dataset_path: Caminho do dataset (usa gerado se None)
            
        Returns:
            Resultados do benchmark de treinamento
        """
        
        logger.info("🤖 Benchmark de treinamento YOLO")
        
        try:
            # Usar dataset gerado no benchmark anterior se disponível
            if not dataset_path:
                dataset_path = self.results.get('generation_benchmark', {}).get('output_path')
            
            if not dataset_path or not Path(dataset_path).exists():
                logger.error("Dataset não disponível para benchmark de treinamento")
                return {'error': 'No dataset available'}
            
            # Configuração de treinamento para benchmark (rápido)
            trainer = YOLOTrainer()
            trainer.config.model_size = "yolov8n"  # Modelo mais rápido
            trainer.config.epochs = 10  # Poucas épocas para benchmark
            trainer.config.batch_size = 4
            trainer.config.patience = 5
            trainer.config.verbose = False
            
            # Otimizar para hardware atual
            trainer.optimize_config_for_hardware()
            
            # Medir tempo de treinamento
            train_start = time.time()
            
            # Criar dataset YOLO válido se necessário
            from src.dataset.yolo_formatter import YOLOFormatter
            formatter = YOLOFormatter()
            
            yolo_dataset_path = formatter.format_dataset(
                dataset_dir=dataset_path,
                output_dir=str(self.output_dir / "yolo_benchmark"),
                task_type="detection",
                train_split=0.8,
                val_split=0.2,
                generate_annotations=True
            )
            
            if not yolo_dataset_path:
                return {'error': 'Failed to create YOLO dataset'}
            
            # Executar treinamento
            model_path = trainer.train(
                dataset_path=str(yolo_dataset_path / "data.yaml"),
                resume=False,
                pretrained=True
            )
            
            training_time = time.time() - train_start
            
            training_results = {
                'training_time_seconds': training_time,
                'training_time_minutes': training_time / 60,
                'epochs_trained': trainer.config.epochs,
                'batch_size_used': trainer.config.batch_size,
                'model_path': model_path,
                'dataset_path': str(yolo_dataset_path),
                'hardware_config': {
                    'device': trainer.config.device,
                    'workers': trainer.config.workers
                }
            }
            
            self.results['training_benchmark'] = training_results
            
            logger.info(f"  ⏱️ Tempo de treinamento: {training_time/60:.1f} minutos")
            logger.info(f"  🏋️ Configuração: {trainer.config.model_size} - {trainer.config.epochs} épocas")
            logger.info(f"  💾 Modelo salvo: {model_path}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Erro no benchmark de treinamento: {e}")
            return {'error': str(e)}
    
    def benchmark_evaluation_performance(self, model_path: str = None, dataset_path: str = None) -> dict:
        """
        Benchmark de performance de avaliação
        
        Args:
            model_path: Caminho do modelo treinado
            dataset_path: Caminho do dataset de teste
            
        Returns:
            Resultados do benchmark de avaliação
        """
        
        logger.info("📊 Benchmark de avaliação")
        
        try:
            # Usar resultados dos benchmarks anteriores se disponíveis
            if not model_path:
                model_path = self.results.get('training_benchmark', {}).get('model_path')
            
            if not dataset_path:
                dataset_path = self.results.get('training_benchmark', {}).get('dataset_path')
            
            if not model_path or not dataset_path:
                logger.error("Modelo ou dataset não disponível para avaliação")
                return {'error': 'Model or dataset not available'}
            
            # Executar avaliação
            evaluator = ModelEvaluator()
            
            eval_start = time.time()
            results = evaluator.evaluate_model(model_path, dataset_path)
            eval_time = time.time() - eval_start
            
            evaluation_results = {
                'evaluation_time_seconds': eval_time,
                'map50': results.map50,
                'map50_95': results.map50_95,
                'precision': results.precision,
                'recall': results.recall,
                'f1_score': results.f1_score,
                'per_class_metrics': results.per_class_metrics,
                'model_path': model_path,
                'dataset_path': dataset_path
            }
            
            self.results['evaluation_benchmark'] = evaluation_results
            
            logger.info(f"  ⏱️ Tempo de avaliação: {eval_time:.1f}s")
            logger.info(f"  📈 mAP@0.5: {results.map50:.3f}")
            logger.info(f"  🎯 Precisão: {results.precision:.3f}")
            logger.info(f"  🔍 Recall: {results.recall:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Erro no benchmark de avaliação: {e}")
            return {'error': str(e)}
    
    def benchmark_scientific_comparison(self) -> dict:
        """
        Compara resultados com benchmarks científicos
        
        Returns:
            Resultados da comparação científica
        """
        
        logger.info("🔬 Comparação com benchmarks científicos")
        
        try:
            eval_results = self.results.get('evaluation_benchmark', {})
            
            if not eval_results or 'error' in eval_results:
                return {'error': 'No evaluation results available'}
            
            # Extrair métricas para comparação
            our_metrics = {
                'map50': eval_results['map50'],
                'precision': eval_results['precision'],
                'recall': eval_results['recall'],
                'f1_score': eval_results['f1_score']
            }
            
            # Comparar com benchmarks científicos
            benchmark_suite = ScientificBenchmarkSuite()
            comparison = benchmark_suite.compare_with_benchmarks(our_metrics)
            
            # Gerar relatório de comparação
            report_path = self.output_dir / "scientific_comparison_report"
            benchmark_suite.generate_benchmark_report(comparison, str(report_path))
            
            scientific_results = {
                'our_performance': comparison.our_performance,
                'rankings': comparison.rankings,
                'statistical_tests': comparison.statistical_tests,
                'improvement_suggestions': comparison.improvement_suggestions,
                'report_path': str(report_path)
            }
            
            self.results['scientific_comparison'] = scientific_results
            
            # Log resultados importantes
            if comparison.rankings:
                for metric, rank in comparison.rankings.items():
                    logger.info(f"  {metric}: #{rank} posição")
            
            if comparison.improvement_suggestions:
                logger.info(f"  💡 {len(comparison.improvement_suggestions)} sugestões de melhoria")
            
            return scientific_results
            
        except Exception as e:
            logger.error(f"Erro na comparação científica: {e}")
            return {'error': str(e)}
    
    def calculate_overall_score(self) -> float:
        """
        Calcula score geral baseado em todos os benchmarks
        
        Returns:
            Score geral (0-10)
        """
        
        scores = []
        weights = []
        
        # Score de geração (baseado em throughput e qualidade)
        gen_results = self.results.get('generation_benchmark', {})
        if gen_results and 'error' not in gen_results:
            # Normalizar throughput (0-10 baseado em 100 img/h = 5.0)
            throughput_score = min(10, gen_results.get('images_per_hour', 0) / 20)
            quality_score = gen_results.get('avg_quality_score', 0) * 10
            success_score = gen_results.get('success_rate', 0) * 10
            
            gen_score = (throughput_score + quality_score + success_score) / 3
            scores.append(gen_score)
            weights.append(0.3)
        
        # Score de treinamento (baseado em tempo e eficiência)
        train_results = self.results.get('training_benchmark', {})
        if train_results and 'error' not in train_results:
            # Score baseado em tempo (menos tempo = melhor score)
            train_time_minutes = train_results.get('training_time_minutes', 60)
            time_score = max(0, 10 - (train_time_minutes - 10) / 5)  # 10min = 10, 60min = 0
            
            scores.append(time_score)
            weights.append(0.2)
        
        # Score de avaliação (baseado em mAP)
        eval_results = self.results.get('evaluation_benchmark', {})
        if eval_results and 'error' not in eval_results:
            map_score = eval_results.get('map50', 0) * 10  # mAP 0.85 = 8.5 pontos
            scores.append(map_score)
            weights.append(0.4)
        
        # Score científico (baseado em rankings)
        sci_results = self.results.get('scientific_comparison', {})
        if sci_results and 'error' not in sci_results:
            # Score baseado na posição média nos rankings
            rankings = sci_results.get('rankings', {})
            if rankings:
                avg_rank = np.mean(list(rankings.values()))
                # Converter rank para score (rank 1 = 10 pontos, rank 5 = 6 pontos)
                rank_score = max(0, 11 - avg_rank)
                scores.append(rank_score)
                weights.append(0.1)
        
        # Calcular score ponderado
        if scores:
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalizar pesos
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = 0.0
        
        self.results['overall_score'] = float(overall_score)
        return overall_score
    
    def generate_report(self) -> str:
        """
        Gera relatório completo de benchmark
        
        Returns:
            Caminho do relatório gerado
        """
        
        # Calcular score geral
        overall_score = self.calculate_overall_score()
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar resultados JSON
        json_report = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_report, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Gerar relatório em texto
        text_report = self.output_dir / f"benchmark_report_{timestamp}.txt"
        
        with open(text_report, 'w', encoding='utf-8') as f:
            f.write("🌱 RELATÓRIO DE BENCHMARK - PASTAGENS BRASILEIRAS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Score Geral: {overall_score:.1f}/10\n\n")
            
            # Informações do sistema
            f.write("💻 SISTEMA:\n")
            f.write("-" * 20 + "\n")
            system_info = self.results['system_info']
            f.write(f"GPU: {system_info.get('gpu_name', 'CPU only')}\n")
            f.write(f"Memória: {system_info.get('memory_gb', 0):.1f}GB\n")
            f.write(f"Python: {system_info.get('python_version', 'N/A')}\n")
            f.write(f"PyTorch: {system_info.get('torch_version', 'N/A')}\n\n")
            
            # Resultados de geração
            gen_results = self.results.get('generation_benchmark', {})
            if gen_results and 'error' not in gen_results:
                f.write("🎨 GERAÇÃO DE IMAGENS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Throughput: {gen_results['images_per_hour']:.0f} imagens/hora\n")
                f.write(f"Tempo médio: {gen_results['avg_time_per_image']:.2f}s/imagem\n")
                f.write(f"Taxa de sucesso: {gen_results['success_rate']*100:.1f}%\n")
                f.write(f"Qualidade média: {gen_results['avg_quality_score']:.3f}\n\n")
            
            # Resultados de treinamento
            train_results = self.results.get('training_benchmark', {})
            if train_results and 'error' not in train_results:
                f.write("🤖 TREINAMENTO YOLO:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Tempo: {train_results['training_time_minutes']:.1f} minutos\n")
                f.write(f"Épocas: {train_results['epochs_trained']}\n")
                f.write(f"Batch size: {train_results['batch_size_used']}\n\n")
            
            # Resultados de avaliação
            eval_results = self.results.get('evaluation_benchmark', {})
            if eval_results and 'error' not in eval_results:
                f.write("📊 AVALIAÇÃO:\n")
                f.write("-" * 20 + "\n")
                f.write(f"mAP@0.5: {eval_results['map50']:.3f}\n")
                f.write(f"Precisão: {eval_results['precision']:.3f}\n")
                f.write(f"Recall: {eval_results['recall']:.3f}\n")
                f.write(f"F1-Score: {eval_results['f1_score']:.3f}\n\n")
            
            # Comparação científica
            sci_results = self.results.get('scientific_comparison', {})
            if sci_results and 'error' not in sci_results:
                f.write("🔬 COMPARAÇÃO CIENTÍFICA:\n")
                f.write("-" * 20 + "\n")
                rankings = sci_results.get('rankings', {})
                for metric, rank in rankings.items():
                    f.write(f"{metric}: #{rank} posição\n")
                f.write("\n")
            
            # Interpretação do score
            f.write("📈 INTERPRETAÇÃO:\n")
            f.write("-" * 20 + "\n")
            if overall_score >= 8.0:
                f.write("🟢 EXCELENTE: Performance superior, pronto para produção\n")
            elif overall_score >= 6.0:
                f.write("🟡 BOM: Performance aceitável, algumas otimizações recomendadas\n")
            elif overall_score >= 4.0:
                f.write("🟠 MODERADO: Performance básica, otimizações necessárias\n")
            else:
                f.write("🔴 BAIXO: Performance insuficiente, revisão completa necessária\n")
        
        logger.info(f"📄 Relatório gerado: {text_report}")
        logger.info(f"📊 Score geral: {overall_score:.1f}/10")
        
        return str(text_report)
    
    def run_full_benchmark(self, num_images: int = 10) -> str:
        """
        Executa benchmark completo
        
        Args:
            num_images: Número de imagens para gerar
            
        Returns:
            Caminho do relatório gerado
        """
        
        logger.info("🚀 Iniciando benchmark completo")
        start_time = time.time()
        
        try:
            # 1. Benchmark de geração
            self.benchmark_generation_performance(num_images)
            
            # 2. Benchmark de treinamento
            self.benchmark_training_performance()
            
            # 3. Benchmark de avaliação
            self.benchmark_evaluation_performance()
            
            # 4. Comparação científica
            self.benchmark_scientific_comparison()
            
            # 5. Gerar relatório
            report_path = self.generate_report()
            
            total_time = time.time() - start_time
            logger.info(f"✅ Benchmark completo em {total_time/60:.1f} minutos")
            
            return report_path
            
        except KeyboardInterrupt:
            logger.warning("⚠️ Benchmark interrompido pelo usuário")
            return self.generate_report()  # Gerar relatório parcial
            
        except Exception as e:
            logger.error(f"❌ Erro no benchmark: {e}")
            return self.generate_report()  # Gerar relatório com erros

def main():
    """Função principal para execução via linha de comando"""
    
    parser = argparse.ArgumentParser(
        description="Benchmark automático do sistema de pastagens brasileiras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python run_benchmarks.py --quick                    # Benchmark rápido (5 imagens)
  python run_benchmarks.py --full                     # Benchmark completo (50 imagens)
  python run_benchmarks.py --images 20 --output /tmp  # Personalizado
        """
    )
    
    parser.add_argument(
        '--images', '-n',
        type=int,
        default=10,
        help='Número de imagens para gerar (default: 10)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default="/tmp/pastagens_benchmark",
        help='Diretório de output (default: /tmp/pastagens_benchmark)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Benchmark rápido (5 imagens, configuração mínima)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Benchmark completo (50 imagens, todas as análises)'
    )
    
    parser.add_argument(
        '--generation-only',
        action='store_true',
        help='Apenas benchmark de geração'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Log detalhado'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ajustar parâmetros baseado nas flags
    if args.quick:
        num_images = 5
    elif args.full:
        num_images = 50
    else:
        num_images = args.images
    
    logger.info(f"Iniciando benchmark com {num_images} imagens")
    
    # Criar runner
    runner = BenchmarkRunner(output_dir=args.output)
    
    try:
        if args.generation_only:
            # Apenas geração
            runner.benchmark_generation_performance(num_images)
            report_path = runner.generate_report()
        else:
            # Benchmark completo
            report_path = runner.run_full_benchmark(num_images)
        
        print(f"\n🎉 Benchmark concluído!")
        print(f"📄 Relatório: {report_path}")
        print(f"📊 Score: {runner.results['overall_score']:.1f}/10")
        
        # Mostrar resumo rápido
        if runner.results['evaluation_benchmark']:
            eval_results = runner.results['evaluation_benchmark']
            print(f"🎯 mAP@0.5: {eval_results['map50']:.3f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrompido pelo usuário")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
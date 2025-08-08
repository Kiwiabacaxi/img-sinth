"""
Gerador de datasets sintéticos para pastagens brasileiras
Integra Stable Diffusion, ControlNet e sistema de prompts especializado
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass, asdict
from PIL import Image
from tqdm import tqdm
import yaml
from datetime import datetime

from ..diffusion.pipeline_manager import PipelineManager
from ..diffusion.prompt_engine import PromptEngine, PastureConfig, Biome, Season, PastureQuality
from ..diffusion.controlnet_adapter import ControlNetAdapter
from ..diffusion.image_postprocess import ImagePostProcessor, ProcessingConfig
from .quality_metrics import QualityMetrics

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuração para geração de dataset"""
    num_images: int = 1000
    resolution: Tuple[int, int] = (1024, 1024)
    output_dir: str = "outputs/generated_images"
    
    # Distribuições
    biome_distribution: Dict[str, float] = None
    season_distribution: Dict[str, float] = None  
    quality_distribution: Dict[str, float] = None
    
    # Parâmetros de geração
    guidance_scale_range: Tuple[float, float] = (7.0, 12.0)
    num_inference_steps: int = 30
    use_controlnet: bool = True
    controlnet_strength: float = 1.0
    
    # Controle de qualidade
    quality_threshold: float = 0.7
    max_retries: int = 3
    
    # Seeds para reprodutibilidade
    use_fixed_seeds: bool = False
    seed_range: Tuple[int, int] = (0, 999999)
    
    def __post_init__(self):
        if self.biome_distribution is None:
            self.biome_distribution = {"cerrado": 0.5, "mata_atlantica": 0.3, "pampa": 0.2}
        if self.season_distribution is None:
            self.season_distribution = {"seca": 0.45, "chuvas": 0.4, "transicao": 0.15}
        if self.quality_distribution is None:
            self.quality_distribution = {"boa": 0.3, "moderada": 0.4, "degradada": 0.3}

class DatasetGenerator:
    """
    Gerador principal de datasets sintéticos
    Coordena pipeline completo de geração, pós-processamento e validação
    """
    
    def __init__(
        self,
        pipeline_manager: Optional[PipelineManager] = None,
        prompt_engine: Optional[PromptEngine] = None,
        controlnet_adapter: Optional[ControlNetAdapter] = None,
        post_processor: Optional[ImagePostProcessor] = None,
        config_dir: str = "configs"
    ):
        self.pipeline_manager = pipeline_manager or PipelineManager()
        self.prompt_engine = prompt_engine or PromptEngine(config_dir=f"{config_dir}/prompts")
        self.controlnet_adapter = controlnet_adapter or ControlNetAdapter()
        self.post_processor = post_processor or ImagePostProcessor()
        
        # Métricas e estatísticas
        self.generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'quality_failures': 0,
            'technical_failures': 0,
            'avg_quality_score': 0.0
        }
        
        self.quality_checker = QualityMetrics()
        
        # Carregar configurações adicionais
        self.config_dir = Path(config_dir)
        self._load_generation_configs()
        
    def _load_generation_configs(self):
        """Carrega configurações específicas de geração"""
        config_file = self.config_dir / "generation" / "dataset_specs.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    self.dataset_specs = yaml.safe_load(f)
                logger.info(f"✅ Configurações carregadas: {config_file}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar configurações: {e}")
                self.dataset_specs = {}
        else:
            self.dataset_specs = {}
            
    def generate_dataset(
        self, 
        config: GenerationConfig,
        save_metadata: bool = True,
        save_intermediate: bool = False
    ) -> str:
        """
        Gera dataset completo com imagens e metadados
        
        Args:
            config: Configuração de geração
            save_metadata: Se deve salvar metadados detalhados
            save_intermediate: Se deve salvar passos intermediários
            
        Returns:
            Caminho do dataset gerado
        """
        
        logger.info(f"🌱 Iniciando geração de dataset: {config.num_images} imagens")
        
        # Criar diretórios
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        metadata_dir = output_path / "metadata" 
        if save_metadata:
            metadata_dir.mkdir(exist_ok=True)
            
        # Resetar estatísticas
        self.generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0, 
            'quality_failures': 0,
            'technical_failures': 0,
            'avg_quality_score': 0.0
        }
        
        # Gerar configurações de pastagens
        pasture_configs = self._generate_pasture_configurations(config)
        
        # Carregar pipeline base
        self.pipeline_manager.load_base_pipeline()
        if config.use_controlnet:
            self.pipeline_manager.load_controlnet_pipeline()
            
        # Gerar imagens
        generated_metadata = []
        quality_scores = []
        
        progress_bar = tqdm(total=config.num_images, desc="Gerando imagens")
        
        successful_count = 0
        attempt_count = 0
        
        while successful_count < config.num_images and attempt_count < config.num_images * 3:
            attempt_count += 1
            self.generation_stats['total_attempts'] = attempt_count
            
            try:
                # Selecionar configuração aleatória
                pasture_config = random.choice(pasture_configs)
                
                # Gerar imagem
                result = self._generate_single_image(
                    pasture_config, 
                    config,
                    image_id=successful_count,
                    save_intermediate=save_intermediate,
                    intermediate_dir=output_path / "intermediate" if save_intermediate else None
                )
                
                if result is not None:
                    image, metadata, quality_metrics = result
                    
                    # Verificar qualidade
                    if quality_metrics.overall_score >= config.quality_threshold:
                        # Salvar imagem
                        image_filename = f"pasture_{successful_count:06d}.jpg"
                        image_path = images_dir / image_filename
                        image.save(image_path, quality=95, optimize=True)
                        
                        # Preparar metadados
                        full_metadata = {
                            'image_id': successful_count,
                            'filename': image_filename,
                            'generation_timestamp': datetime.now().isoformat(),
                            'pasture_config': asdict(pasture_config),
                            'generation_params': metadata,
                            'quality_metrics': asdict(quality_metrics),
                            'dataset_config': asdict(config)
                        }
                        
                        generated_metadata.append(full_metadata)
                        quality_scores.append(quality_metrics.overall_score)
                        
                        # Salvar metadados individuais se solicitado
                        if save_metadata:
                            metadata_filename = f"pasture_{successful_count:06d}.json"
                            metadata_path = metadata_dir / metadata_filename
                            
                            with open(metadata_path, 'w') as f:
                                json.dump(full_metadata, f, indent=2, ensure_ascii=False)
                                
                        successful_count += 1
                        self.generation_stats['successful_generations'] = successful_count
                        progress_bar.update(1)
                        
                        logger.debug(f"✅ Imagem {successful_count} gerada (qualidade: {quality_metrics.overall_score:.3f})")
                        
                    else:
                        self.generation_stats['quality_failures'] += 1
                        logger.debug(f"❌ Qualidade insuficiente: {quality_metrics.overall_score:.3f} < {config.quality_threshold}")
                        
                else:
                    self.generation_stats['technical_failures'] += 1
                    
            except Exception as e:
                self.generation_stats['technical_failures'] += 1
                logger.error(f"❌ Erro na geração: {e}")
                continue
                
            # Limpeza de memória periódica
            if attempt_count % 20 == 0:
                self.pipeline_manager.clear_memory()
                
        progress_bar.close()
        
        # Calcular estatísticas finais
        if quality_scores:
            self.generation_stats['avg_quality_score'] = np.mean(quality_scores)
            
        # Salvar metadados do dataset
        dataset_metadata = {
            'dataset_info': {
                'total_images': successful_count,
                'generation_date': datetime.now().isoformat(),
                'config': asdict(config),
                'statistics': self.generation_stats
            },
            'images': generated_metadata
        }
        
        dataset_metadata_path = output_path / "dataset_metadata.json"
        with open(dataset_metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)
            
        # Salvar relatório de qualidade
        self._save_quality_report(quality_scores, output_path / "quality_report.json")
        
        logger.info(f"✅ Dataset gerado com sucesso: {successful_count} imagens")
        logger.info(f"📊 Taxa de sucesso: {successful_count/attempt_count*100:.1f}%")
        logger.info(f"📈 Qualidade média: {self.generation_stats['avg_quality_score']:.3f}")
        
        return str(output_path)
        
    def _generate_pasture_configurations(self, config: GenerationConfig) -> List[PastureConfig]:
        """Gera configurações variadas de pastagens"""
        
        configs = []
        
        # Calcular número de imagens por categoria
        biome_counts = {
            biome: int(config.num_images * weight) 
            for biome, weight in config.biome_distribution.items()
        }
        
        season_counts = {
            season: int(config.num_images * weight)
            for season, weight in config.season_distribution.items()
        }
        
        quality_counts = {
            quality: int(config.num_images * weight)
            for quality, weight in config.quality_distribution.items()
        }
        
        # Gerar configurações balanceadas
        for _ in range(config.num_images * 2):  # Gerar extras para seleção aleatória
            
            # Seleção ponderada
            biome = np.random.choice(
                list(config.biome_distribution.keys()),
                p=list(config.biome_distribution.values())
            )
            
            season = np.random.choice(
                list(config.season_distribution.keys()),
                p=list(config.season_distribution.values())
            )
            
            quality = np.random.choice(
                list(config.quality_distribution.keys()), 
                p=list(config.quality_distribution.values())
            )
            
            # Converter para enums
            biome_enum = Biome(biome)
            season_enum = Season(season)
            quality_enum = PastureQuality(quality)
            
            # Gerar espécies invasoras baseadas na qualidade
            invasive_species = self._generate_invasive_species(quality_enum, biome_enum)
            
            # Gerar coberturas realistas
            grass_coverage = self._generate_realistic_coverage(quality_enum, season_enum)
            soil_exposure = 100 - grass_coverage
            
            pasture_config = PastureConfig(
                biome=biome_enum,
                season=season_enum,
                quality=quality_enum,
                invasive_species=invasive_species,
                grass_coverage=grass_coverage,
                soil_exposure=soil_exposure
            )
            
            configs.append(pasture_config)
            
        return configs
        
    def _generate_invasive_species(self, quality: PastureQuality, biome: Biome) -> List[str]:
        """Gera lista de espécies invasoras baseada na qualidade e bioma"""
        
        if quality == PastureQuality.BOA:
            return []  # Pastagem boa sem invasoras
            
        # Espécies por bioma
        biome_invasive_map = {
            Biome.CERRADO: ["capim_gordura", "carqueja", "cupinzeiro"],
            Biome.MATA_ATLANTICA: ["samambaia", "carqueja", "outras_invasoras"],
            Biome.PAMPA: ["carqueja", "outras_invasoras", "cupinzeiro"]
        }
        
        available_species = biome_invasive_map.get(biome, ["carqueja", "outras_invasoras"])
        
        # Probabilidade baseada na qualidade
        if quality == PastureQuality.MODERADA:
            # 1-2 espécies invasoras
            num_species = np.random.choice([1, 2], p=[0.7, 0.3])
        else:  # DEGRADADA
            # 2-3 espécies invasoras
            num_species = np.random.choice([2, 3], p=[0.6, 0.4])
            
        return random.sample(available_species, min(num_species, len(available_species)))
        
    def _generate_realistic_coverage(self, quality: PastureQuality, season: Season) -> int:
        """Gera cobertura de gramíneas realista"""
        
        # Ranges base por qualidade
        base_ranges = {
            PastureQuality.BOA: (80, 95),
            PastureQuality.MODERADA: (50, 80), 
            PastureQuality.DEGRADADA: (20, 50)
        }
        
        base_min, base_max = base_ranges[quality]
        
        # Ajustes sazonais
        if season == Season.SECA:
            # Reduzir cobertura na seca
            base_min = max(base_min - 15, 10)
            base_max = max(base_max - 10, base_min + 10)
        elif season == Season.CHUVAS:
            # Aumentar cobertura nas chuvas
            base_min = min(base_min + 5, 90)
            base_max = min(base_max + 10, 95)
            
        return random.randint(base_min, base_max)
        
    def _generate_single_image(
        self,
        pasture_config: PastureConfig,
        gen_config: GenerationConfig,
        image_id: int,
        save_intermediate: bool = False,
        intermediate_dir: Optional[Path] = None
    ) -> Optional[Tuple[Image.Image, Dict, object]]:
        """Gera uma única imagem com pós-processamento"""
        
        try:
            # Gerar prompts
            positive_prompt, negative_prompt = self.prompt_engine.generate_prompt(
                pasture_config, variation=True
            )
            
            # Parâmetros de geração
            guidance_scale = random.uniform(*gen_config.guidance_scale_range)
            seed = random.randint(*gen_config.seed_range) if not gen_config.use_fixed_seeds else image_id
            
            # Gerar ControlNet condition se necessário
            controlnet_image = None
            if gen_config.use_controlnet:
                controlnet_image = self._generate_controlnet_condition(pasture_config)
                
            # Gerar imagem base
            generation_result = self.pipeline_manager.generate_image(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=gen_config.num_inference_steps,
                guidance_scale=guidance_scale,
                width=gen_config.resolution[0],
                height=gen_config.resolution[1],
                seed=seed,
                controlnet_image=controlnet_image,
                controlnet_conditioning_scale=gen_config.controlnet_strength
            )
            
            raw_image = generation_result['image']
            
            # Pós-processamento
            processing_config = ProcessingConfig(
                target_resolution=gen_config.resolution
            )
            
            processed_image, quality_metrics = self.post_processor.process_image(
                raw_image,
                config=processing_config,
                season=pasture_config.season.value,
                biome=pasture_config.biome.value,
                save_intermediate=save_intermediate,
                output_dir=str(intermediate_dir / f"image_{image_id}") if intermediate_dir else None
            )
            
            # Metadados de geração
            metadata = {
                'positive_prompt': positive_prompt,
                'negative_prompt': negative_prompt,
                'guidance_scale': guidance_scale,
                'num_inference_steps': gen_config.num_inference_steps,
                'seed': seed,
                'controlnet_used': controlnet_image is not None,
                'resolution': gen_config.resolution
            }
            
            return processed_image, metadata, quality_metrics
            
        except Exception as e:
            logger.error(f"Erro na geração da imagem {image_id}: {e}")
            return None
            
    def _generate_controlnet_condition(self, pasture_config: PastureConfig) -> Optional[Image.Image]:
        """Gera condição ControlNet baseada na configuração da pastagem"""
        
        try:
            # Escolher tipo de condição baseado na qualidade
            if pasture_config.quality == PastureQuality.BOA:
                # Layout uniforme para pastagens boas
                condition = self.controlnet_adapter.create_pasture_layout_mask(
                    layout_type="uniform"
                )
            elif pasture_config.quality == PastureQuality.MODERADA:
                # Layout irregular para pastagens moderadas
                condition = self.controlnet_adapter.create_pasture_layout_mask(
                    layout_type="patchy"
                )
            else:  # DEGRADADA
                # Layout degradado
                condition = self.controlnet_adapter.create_pasture_layout_mask(
                    layout_type="degraded"
                )
                
            # Adicionar variação de profundidade sazonal
            depth_map = self.controlnet_adapter.create_seasonal_depth_map(
                season=pasture_config.season.value,
                biome=pasture_config.biome.value
            )
            
            # Combinar condições (usar layout como principal)
            return condition
            
        except Exception as e:
            logger.warning(f"Erro ao gerar condição ControlNet: {e}")
            return None
            
    def _save_quality_report(self, quality_scores: List[float], output_path: Path):
        """Salva relatório detalhado de qualidade"""
        
        if not quality_scores:
            return
            
        report = {
            'summary': {
                'total_images': len(quality_scores),
                'average_score': float(np.mean(quality_scores)),
                'std_score': float(np.std(quality_scores)),
                'min_score': float(np.min(quality_scores)),
                'max_score': float(np.max(quality_scores))
            },
            'distribution': {
                'excellent (>0.8)': sum(1 for s in quality_scores if s > 0.8),
                'good (0.6-0.8)': sum(1 for s in quality_scores if 0.6 <= s <= 0.8),
                'moderate (0.4-0.6)': sum(1 for s in quality_scores if 0.4 <= s < 0.6),
                'poor (<0.4)': sum(1 for s in quality_scores if s < 0.4)
            },
            'percentiles': {
                'p10': float(np.percentile(quality_scores, 10)),
                'p25': float(np.percentile(quality_scores, 25)),
                'p50': float(np.percentile(quality_scores, 50)),
                'p75': float(np.percentile(quality_scores, 75)),
                'p90': float(np.percentile(quality_scores, 90))
            },
            'generation_stats': self.generation_stats
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"📊 Relatório de qualidade salvo: {output_path}")
        
    def generate_samples(
        self,
        num_samples: int = 10,
        output_dir: str = "outputs/samples",
        specific_configs: Optional[List[PastureConfig]] = None
    ) -> List[str]:
        """
        Gera amostras rápidas para teste
        
        Args:
            num_samples: Número de amostras
            output_dir: Diretório de output
            specific_configs: Configurações específicas (opcional)
            
        Returns:
            Lista de caminhos das imagens geradas
        """
        
        logger.info(f"🧪 Gerando {num_samples} amostras de teste")
        
        # Configuração simples para samples
        config = GenerationConfig(
            num_images=num_samples,
            output_dir=output_dir,
            quality_threshold=0.5,  # Threshold mais baixo para samples
            use_controlnet=False,   # Mais rápido sem ControlNet
            num_inference_steps=20  # Menos steps para velocidade
        )
        
        # Usar configurações específicas se fornecidas
        if specific_configs:
            pasture_configs = specific_configs[:num_samples]
        else:
            pasture_configs = self._generate_pasture_configurations(config)[:num_samples]
            
        # Gerar samples
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.pipeline_manager.load_base_pipeline()
        
        generated_paths = []
        
        for i, pasture_config in enumerate(tqdm(pasture_configs, desc="Gerando samples")):
            try:
                result = self._generate_single_image(
                    pasture_config, config, i, save_intermediate=False
                )
                
                if result is not None:
                    image, metadata, quality_metrics = result
                    
                    # Salvar sample
                    sample_path = output_path / f"sample_{i:03d}.jpg"
                    image.save(sample_path, quality=85)
                    generated_paths.append(str(sample_path))
                    
                    # Salvar metadados básicos
                    metadata_path = output_path / f"sample_{i:03d}.json"
                    with open(metadata_path, 'w') as f:
                        json.dump({
                            'pasture_config': asdict(pasture_config),
                            'quality_score': quality_metrics.overall_score,
                            'prompt': metadata['positive_prompt']
                        }, f, indent=2, ensure_ascii=False)
                        
            except Exception as e:
                logger.error(f"Erro na geração do sample {i}: {e}")
                continue
                
        logger.info(f"✅ {len(generated_paths)} samples gerados em {output_dir}")
        return generated_paths
        
    def get_generation_statistics(self) -> Dict:
        """Retorna estatísticas da última geração"""
        return self.generation_stats.copy()
        
    def cleanup(self):
        """Limpeza de recursos"""
        self.pipeline_manager.unload_models()
        logger.info("🧹 Recursos liberados")
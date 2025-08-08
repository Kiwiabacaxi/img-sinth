"""
Sistema avançado de Prompt Engineering para pastagens brasileiras
Especializado em cenários específicos do Brasil (Cerrado, Mata Atlântica, Pampa)
"""

import yaml
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class Biome(Enum):
    CERRADO = "cerrado"
    MATA_ATLANTICA = "mata_atlantica" 
    PAMPA = "pampa"

class Season(Enum):
    SECA = "seca"
    CHUVAS = "chuvas"
    TRANSICAO = "transicao"

class PastureQuality(Enum):
    BOA = "boa"
    MODERADA = "moderada"
    DEGRADADA = "degradada"

@dataclass
class PastureConfig:
    """Configuração de pastagem para geração de prompts"""
    biome: Biome
    season: Season
    quality: PastureQuality
    invasive_species: List[str]
    grass_coverage: int  # Percentual 0-100
    soil_exposure: int   # Percentual 0-100
    
class PromptEngine:
    """
    Engine de prompts especializado em pastagens brasileiras
    Gera prompts contextualmente ricos baseados em dados científicos
    """
    
    def __init__(self, config_dir: str = "configs/prompts"):
        self.config_dir = Path(config_dir)
        self.templates = {}
        self.variations = {}
        self.species_data = {}
        self.degradation_patterns = {}
        
        # Carregar configurações
        self._load_configurations()
        
        # Templates base específicos por bioma
        self.base_templates = {
            Biome.CERRADO: {
                "base": "{season} Brazilian cerrado pasture, {grass_species} grass {coverage}% coverage, red latosol soil {soil_visibility}, {lighting}, {invasive_description}, realistic drone aerial photography, agricultural field, high resolution, photorealistic",
                "soil_type": "red latosol soil",
                "topography": "0-25° gentle slopes",
                "characteristics": "scattered termite mounds, gallery forests in distance"
            },
            Biome.MATA_ATLANTICA: {
                "base": "{season} Brazilian mata atlântica pasture, {grass_species} grass {coverage}% coverage, argisol soil {soil_visibility}, {lighting}, {invasive_description}, humid subtropical climate, realistic drone photography, agricultural landscape",
                "soil_type": "argisol/cambisol soil", 
                "topography": "5-45° rolling hills",
                "characteristics": "forest fragments visible, high humidity conditions"
            },
            Biome.PAMPA: {
                "base": "{season} Brazilian pampa grassland, native {grass_species} {coverage}% coverage, planosol soil {soil_visibility}, {lighting}, {invasive_description}, windy conditions, realistic aerial photography, extensive grazing system",
                "soil_type": "planosol soil",
                "topography": "gentle undulating plains", 
                "characteristics": "constant wind patterns, extensive grazing areas"
            }
        }
        
        # Negative prompts padrão
        self.negative_prompts = {
            "basic": "cartoon, painting, illustration, unrealistic, blurry, low quality, distorted, artificial looking",
            "agricultural": "urban elements, buildings, roads, people, vehicles, power lines, industrial structures",
            "quality": "oversaturated, undersaturated, noise, artifacts, watermark, text, logo"
        }
        
    def _load_configurations(self):
        """Carrega configurações YAML se existirem"""
        config_files = {
            'base_prompts.yaml': 'templates',
            'seasonal_variations.yaml': 'variations', 
            'species_specific.yaml': 'species_data',
            'degradation_patterns.yaml': 'degradation_patterns'
        }
        
        for filename, attr in config_files.items():
            filepath = self.config_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        setattr(self, attr, data)
                        logger.info(f"✅ Carregado {filename}")
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao carregar {filename}: {e}")
                    
    def generate_prompt(self, config: PastureConfig, variation: bool = True) -> Tuple[str, str]:
        """
        Gera prompt completo baseado na configuração
        
        Args:
            config: Configuração da pastagem
            variation: Se deve aplicar variações aleatórias
            
        Returns:
            Tuple (positive_prompt, negative_prompt)
        """
        
        # Obter template base
        base_template = self.base_templates[config.biome]["base"]
        
        # Gerar componentes do prompt
        season_desc = self._get_seasonal_description(config.season, config.biome)
        grass_species = self._get_grass_species(config.biome, config.season)
        coverage_desc = self._get_coverage_description(config.grass_coverage)
        soil_desc = self._get_soil_description(config.soil_exposure, config.season)
        lighting_desc = self._get_lighting_description(config.season)
        invasive_desc = self._get_invasive_description(config.invasive_species, config.quality)
        
        # Construir prompt principal
        positive_prompt = base_template.format(
            season=season_desc,
            grass_species=grass_species,
            coverage=config.grass_coverage,
            soil_visibility=soil_desc,
            lighting=lighting_desc,
            invasive_description=invasive_desc
        )
        
        # Adicionar variações se solicitado
        if variation:
            positive_prompt = self._apply_variations(positive_prompt, config)
            
        # Gerar negative prompt
        negative_prompt = self._generate_negative_prompt(config)
        
        logger.debug(f"Gerado prompt para {config.biome.value} - {config.season.value}")
        
        return positive_prompt, negative_prompt
        
    def _get_seasonal_description(self, season: Season, biome: Biome) -> str:
        """Descrição sazonal específica por bioma"""
        descriptions = {
            Season.SECA: {
                Biome.CERRADO: "Dry season",
                Biome.MATA_ATLANTICA: "Dry winter season", 
                Biome.PAMPA: "Dry season with constant winds"
            },
            Season.CHUVAS: {
                Biome.CERRADO: "Rainy season",
                Biome.MATA_ATLANTICA: "Humid summer season",
                Biome.PAMPA: "Wet season"
            },
            Season.TRANSICAO: {
                Biome.CERRADO: "Transition season",
                Biome.MATA_ATLANTICA: "Transitional season",
                Biome.PAMPA: "Seasonal transition"
            }
        }
        
        return descriptions[season][biome]
        
    def _get_grass_species(self, biome: Biome, season: Season) -> str:
        """Espécies de gramíneas por bioma"""
        species = {
            Biome.CERRADO: ["Brachiaria brizantha", "Panicum maximum", "Andropogon gayanus"],
            Biome.MATA_ATLANTICA: ["Brachiaria decumbens", "Panicum maximum", "Pennisetum purpureum"],
            Biome.PAMPA: ["native grasses mix", "Brachiaria species", "Paspalum notatum"]
        }
        
        return random.choice(species[biome])
        
    def _get_coverage_description(self, coverage: int) -> str:
        """Descrição da cobertura de gramíneas"""
        if coverage >= 85:
            return "dense uniform"
        elif coverage >= 60:
            return "good"
        elif coverage >= 40:
            return "moderate patchy"
        else:
            return "sparse poor"
            
    def _get_soil_description(self, exposure: int, season: Season) -> str:
        """Descrição da exposição do solo"""
        if exposure <= 10:
            return "barely visible"
        elif exposure <= 30:
            return "partially visible"
        elif exposure <= 50:
            return "moderately exposed"
        else:
            return "heavily exposed"
            
    def _get_lighting_description(self, season: Season) -> str:
        """Condições de iluminação por estação"""
        lighting = {
            Season.SECA: ["intense midday sunlight", "harsh direct lighting", "strong shadows"],
            Season.CHUVAS: ["diffuse overcast light", "soft morning light", "cloudy conditions"],
            Season.TRANSICAO: ["variable lighting", "partly cloudy", "dynamic shadows"]
        }
        
        return random.choice(lighting[season])
        
    def _get_invasive_description(self, invasive_species: List[str], quality: PastureQuality) -> str:
        """Descrição de plantas invasoras"""
        if not invasive_species or quality == PastureQuality.BOA:
            return "clean pasture without invasive species"
            
        descriptions = {
            "capim_gordura": "golden patches of capim-gordura (Melinis minutiflora) invasive grass",
            "carqueja": "scattered carqueja shrubs (Baccharis trimera) woody invasive plants", 
            "samambaia": "clustered bracken fern (Pteridium aquilinum) invasive patches",
            "cupinzeiro": "termite mounds scattered throughout",
            "outras_invasoras": "mixed invasive plant species"
        }
        
        invasive_texts = []
        for species in invasive_species:
            if species in descriptions:
                coverage = random.randint(5, 30) if quality == PastureQuality.MODERADA else random.randint(20, 50)
                invasive_texts.append(f"{descriptions[species]} {coverage}% coverage")
                
        return ", ".join(invasive_texts) if invasive_texts else "minimal invasive vegetation"
        
    def _apply_variations(self, prompt: str, config: PastureConfig) -> str:
        """Aplica variações contextuais ao prompt"""
        variations = []
        
        # Variações de qualidade visual
        quality_terms = ["photorealistic", "high resolution", "detailed", "sharp focus", "professional photography"]
        variations.append(random.choice(quality_terms))
        
        # Variações específicas por bioma
        biome_variations = {
            Biome.CERRADO: ["gallery forests in background", "scattered trees", "typical cerrado landscape"],
            Biome.MATA_ATLANTICA: ["forest fragments visible", "mountainous backdrop", "subtropical vegetation"],
            Biome.PAMPA: ["endless grassland horizon", "gentle rolling landscape", "extensive grazing area"]
        }
        
        if config.biome in biome_variations:
            variations.append(random.choice(biome_variations[config.biome]))
            
        # Adicionar variações ao prompt
        variation_text = ", ".join(variations)
        return f"{prompt}, {variation_text}"
        
    def _generate_negative_prompt(self, config: PastureConfig) -> str:
        """Gera negative prompt contextual"""
        negative_components = [
            self.negative_prompts["basic"],
            self.negative_prompts["agricultural"],
            self.negative_prompts["quality"]
        ]
        
        # Adicionar negatives específicos por qualidade
        if config.quality == PastureQuality.BOA:
            negative_components.append("weeds, invasive plants, bare soil, erosion, degradation")
        elif config.quality == PastureQuality.DEGRADADA:
            negative_components.append("lush vegetation, perfect grass coverage")
            
        return ", ".join(negative_components)
        
    def generate_batch_prompts(
        self, 
        num_prompts: int,
        biomes: Optional[List[Biome]] = None,
        seasons: Optional[List[Season]] = None,
        qualities: Optional[List[PastureQuality]] = None
    ) -> List[Tuple[str, str, PastureConfig]]:
        """
        Gera batch de prompts com configurações variadas
        
        Args:
            num_prompts: Número de prompts a gerar
            biomes: Lista de biomas (None para todos)
            seasons: Lista de estações (None para todas)
            qualities: Lista de qualidades (None para todas)
            
        Returns:
            Lista de (positive_prompt, negative_prompt, config)
        """
        
        # Usar todos os valores se não especificado
        biomes = biomes or list(Biome)
        seasons = seasons or list(Season)  
        qualities = qualities or list(PastureQuality)
        
        prompts = []
        
        for i in range(num_prompts):
            # Configuração aleatória
            config = PastureConfig(
                biome=random.choice(biomes),
                season=random.choice(seasons),
                quality=random.choice(qualities),
                invasive_species=self._generate_random_invasive_species(),
                grass_coverage=self._generate_random_coverage(),
                soil_exposure=random.randint(5, 60)
            )
            
            # Gerar prompts
            pos_prompt, neg_prompt = self.generate_prompt(config, variation=True)
            prompts.append((pos_prompt, neg_prompt, config))
            
        logger.info(f"✅ Gerados {num_prompts} prompts")
        return prompts
        
    def _generate_random_invasive_species(self) -> List[str]:
        """Gera lista aleatória de espécies invasoras"""
        all_species = ["capim_gordura", "carqueja", "samambaia", "cupinzeiro", "outras_invasoras"]
        num_species = random.choices([0, 1, 2, 3], weights=[30, 40, 20, 10])[0]
        
        return random.sample(all_species, num_species) if num_species > 0 else []
        
    def _generate_random_coverage(self) -> int:
        """Gera cobertura aleatória com distribuição realista"""
        # Distribuição: 30% boa (80-95%), 40% moderada (50-80%), 30% degradada (20-50%)
        quality_weights = [30, 40, 30]
        quality = random.choices(["boa", "moderada", "degradada"], weights=quality_weights)[0]
        
        if quality == "boa":
            return random.randint(80, 95)
        elif quality == "moderada":
            return random.randint(50, 80)
        else:
            return random.randint(20, 50)
            
    def get_prompt_statistics(self, prompts: List[Tuple]) -> Dict:
        """Gera estatísticas sobre batch de prompts gerados"""
        if not prompts:
            return {}
            
        configs = [config for _, _, config in prompts]
        
        return {
            "total_prompts": len(prompts),
            "biomes": {biome.value: sum(1 for c in configs if c.biome == biome) for biome in Biome},
            "seasons": {season.value: sum(1 for c in configs if c.season == season) for season in Season},
            "qualities": {quality.value: sum(1 for c in configs if c.quality == quality) for quality in PastureQuality},
            "avg_coverage": sum(c.grass_coverage for c in configs) / len(configs),
            "avg_soil_exposure": sum(c.soil_exposure for c in configs) / len(configs)
        }
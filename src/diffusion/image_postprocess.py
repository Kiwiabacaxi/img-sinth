"""
Pós-processamento de imagens geradas especializadas para pastagens brasileiras
Inclui correções automáticas, enhancement e validação de qualidade
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from skimage import filters, measure, morphology
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuração de pós-processamento"""
    enhance_contrast: bool = True
    enhance_saturation: bool = True  
    apply_sharpening: bool = True
    correct_colors: bool = True
    remove_artifacts: bool = True
    target_resolution: Tuple[int, int] = (1024, 1024)

@dataclass 
class QualityMetrics:
    """Métricas de qualidade da imagem"""
    blur_score: float
    contrast_score: float
    saturation_score: float
    noise_level: float
    grass_coverage: float
    soil_visibility: float
    overall_score: float

class ImagePostProcessor:
    """
    Pós-processador especializado em imagens de pastagens brasileiras
    Aplica correções automáticas baseadas em características agronômicas
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        # Configurações para diferentes estações
        self.seasonal_configs = {
            'seca': {
                'hue_shift': 10,          # Mais amarelado
                'saturation_mult': 0.8,   # Menos saturado
                'brightness_mult': 1.1,   # Mais claro
                'contrast_mult': 1.2      # Maior contraste
            },
            'chuvas': {
                'hue_shift': -5,          # Mais verdejante
                'saturation_mult': 1.2,   # Mais saturado
                'brightness_mult': 0.95,  # Ligeiramente mais escuro
                'contrast_mult': 1.0      # Contraste natural
            },
            'transicao': {
                'hue_shift': 0,
                'saturation_mult': 1.0,
                'brightness_mult': 1.0,
                'contrast_mult': 1.1
            }
        }
        
        # Cores típicas de pastagens brasileiras (HSV)
        self.brazilian_pasture_colors = {
            'grass_healthy': [(60, 40, 40), (80, 255, 180)],    # Verde saudável
            'grass_dry': [(20, 50, 100), (40, 200, 200)],       # Amarelo/dourado seco
            'soil_red': [(0, 100, 80), (10, 255, 150)],         # Solo vermelho laterítico
            'soil_brown': [(10, 50, 60), (25, 150, 120)]        # Solo marrom
        }
        
    def process_image(
        self, 
        image: Union[Image.Image, np.ndarray],
        config: ProcessingConfig = None,
        season: str = "chuvas",
        biome: str = "cerrado",
        save_intermediate: bool = False,
        output_dir: Optional[str] = None
    ) -> Tuple[Image.Image, QualityMetrics]:
        """
        Processa imagem com pipeline completo de pós-processamento
        
        Args:
            image: Imagem de entrada
            config: Configuração de processamento
            season: Estação ('seca', 'chuvas', 'transicao')
            biome: Bioma ('cerrado', 'mata_atlantica', 'pampa')
            save_intermediate: Salvar passos intermediários
            output_dir: Diretório para salvar intermediários
            
        Returns:
            Tuple (imagem_processada, métricas_qualidade)
        """
        
        config = config or ProcessingConfig()
        
        # Converter para PIL se necessário
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Formato de imagem não suportado")
            
        original = image.copy()
        processed = image.copy()
        
        logger.info(f"Iniciando pós-processamento: {season} - {biome}")
        
        # Pipeline de processamento
        steps = []
        
        # 1. Redimensionamento se necessário
        if processed.size != config.target_resolution:
            processed = processed.resize(config.target_resolution, Image.LANCZOS)
            steps.append(("resize", processed.copy()))
            
        # 2. Remoção de artefatos
        if config.remove_artifacts:
            processed = self._remove_artifacts(processed)
            steps.append(("artifacts_removal", processed.copy()))
            
        # 3. Correção de cores específica da estação
        if config.correct_colors:
            processed = self._correct_seasonal_colors(processed, season, biome)
            steps.append(("color_correction", processed.copy()))
            
        # 4. Ajuste de contraste
        if config.enhance_contrast:
            processed = self._enhance_contrast(processed, season)
            steps.append(("contrast", processed.copy()))
            
        # 5. Ajuste de saturação
        if config.enhance_saturation:
            processed = self._enhance_saturation(processed, season)
            steps.append(("saturation", processed.copy()))
            
        # 6. Nitidez (sharpening)
        if config.apply_sharpening:
            processed = self._apply_sharpening(processed)
            steps.append(("sharpening", processed.copy()))
            
        # 7. Refinamento final
        processed = self._final_refinement(processed, season, biome)
        steps.append(("final", processed.copy()))
        
        # Salvar passos intermediários se solicitado
        if save_intermediate and output_dir:
            self._save_intermediate_steps(steps, output_dir)
            
        # Calcular métricas de qualidade
        metrics = self._calculate_quality_metrics(processed, original)
        
        logger.info(f"✅ Pós-processamento concluído (Score: {metrics.overall_score:.3f})")
        
        return processed, metrics
        
    def _remove_artifacts(self, image: Image.Image) -> Image.Image:
        """Remove artefatos comuns de Stable Diffusion"""
        
        # Converter para array numpy
        img_array = np.array(image)
        
        # Filtro bilateral para suavizar mantendo bordas
        filtered = cv2.bilateralFilter(img_array, 5, 50, 50)
        
        # Remoção de ruído pequeno
        kernel = np.ones((3,3), np.uint8)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        
        return Image.fromarray(filtered)
        
    def _correct_seasonal_colors(
        self, 
        image: Image.Image, 
        season: str, 
        biome: str
    ) -> Image.Image:
        """Corrige cores baseado na estação e bioma"""
        
        # Obter configuração sazonal
        season_config = self.seasonal_configs.get(season, self.seasonal_configs['chuvas'])
        
        # Converter para HSV para manipulação de cores
        img_array = np.array(image)
        hsv = rgb2hsv(img_array / 255.0)
        
        # Ajustar matiz (hue) para estação
        if season_config['hue_shift'] != 0:
            hsv[:,:,0] = (hsv[:,:,0] + season_config['hue_shift']/360) % 1.0
            
        # Ajustar saturação
        hsv[:,:,1] = np.clip(hsv[:,:,1] * season_config['saturation_mult'], 0, 1)
        
        # Ajustar valor (brightness)  
        hsv[:,:,2] = np.clip(hsv[:,:,2] * season_config['brightness_mult'], 0, 1)
        
        # Converter de volta para RGB
        rgb = hsv2rgb(hsv)
        corrected = Image.fromarray((rgb * 255).astype(np.uint8))
        
        # Ajustes específicos por bioma
        if biome == "cerrado":
            # Enfatizar tons de vermelho do solo laterítico
            corrected = self._enhance_red_soil(corrected)
        elif biome == "pampa": 
            # Tons mais neutros, menos saturados
            enhancer = ImageEnhance.Color(corrected)
            corrected = enhancer.enhance(0.9)
            
        return corrected
        
    def _enhance_red_soil(self, image: Image.Image) -> Image.Image:
        """Realça tons vermelhos do solo laterítico do cerrado"""
        
        img_array = np.array(image)
        hsv = rgb2hsv(img_array / 255.0)
        
        # Máscara para tons vermelhos/marrons (solo)
        red_mask = (hsv[:,:,0] < 0.08) | (hsv[:,:,0] > 0.9)  # Hue vermelho
        soil_mask = (hsv[:,:,2] > 0.3) & (hsv[:,:,2] < 0.7)  # Valor médio
        combined_mask = red_mask & soil_mask
        
        # Realçar saturação e ajustar matiz nas áreas de solo
        hsv[combined_mask, 1] = np.clip(hsv[combined_mask, 1] * 1.2, 0, 1)
        hsv[combined_mask, 0] = np.clip(hsv[combined_mask, 0] - 0.02, 0, 1)
        
        # Converter de volta
        rgb = hsv2rgb(hsv) 
        return Image.fromarray((rgb * 255).astype(np.uint8))
        
    def _enhance_contrast(self, image: Image.Image, season: str) -> Image.Image:
        """Ajusta contraste baseado na estação"""
        
        season_config = self.seasonal_configs.get(season, self.seasonal_configs['chuvas'])
        contrast_factor = season_config['contrast_mult']
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast_factor)
        
    def _enhance_saturation(self, image: Image.Image, season: str) -> Image.Image:
        """Ajusta saturação baseado na estação"""
        
        season_config = self.seasonal_configs.get(season, self.seasonal_configs['chuvas'])
        
        # Saturação já foi ajustada em _correct_seasonal_colors
        # Aqui aplicamos ajustes finos adicionais
        if season == "seca":
            # Reduzir saturação em áreas muito verdes (não realista na seca)
            img_array = np.array(image)
            hsv = rgb2hsv(img_array / 255.0)
            
            green_mask = (hsv[:,:,0] > 0.2) & (hsv[:,:,0] < 0.4)  # Tons verdes
            high_sat_mask = hsv[:,:,1] > 0.7  # Alta saturação
            
            # Reduzir saturação em verdes muito saturados na seca
            reduce_mask = green_mask & high_sat_mask
            hsv[reduce_mask, 1] = hsv[reduce_mask, 1] * 0.7
            
            rgb = hsv2rgb(hsv)
            return Image.fromarray((rgb * 255).astype(np.uint8))
            
        return image
        
    def _apply_sharpening(self, image: Image.Image) -> Image.Image:
        """Aplica sharpening sutil para melhorar detalhes"""
        
        # Unsharp mask suave
        gaussian = image.filter(ImageFilter.GaussianBlur(radius=1))
        unsharp = Image.blend(image, gaussian, -0.5)  # Negative blend for sharpening
        
        return unsharp
        
    def _final_refinement(
        self, 
        image: Image.Image, 
        season: str, 
        biome: str
    ) -> Image.Image:
        """Refinamentos finais específicos por contexto"""
        
        refined = image.copy()
        
        # Ajuste de gamma para diferentes condições de luz
        if season == "seca":
            # Gamma mais alto para simular luz intensa
            gamma = 1.1
        else:
            # Gamma normal para condições difusas
            gamma = 1.0
            
        if gamma != 1.0:
            # Aplicar correção gamma
            img_array = np.array(refined) / 255.0
            corrected = np.power(img_array, 1.0/gamma)
            refined = Image.fromarray((corrected * 255).astype(np.uint8))
            
        # Ajuste final de brilho
        if season == "chuvas":
            # Ligeiramente mais escuro para simular nebulosidade
            enhancer = ImageEnhance.Brightness(refined)
            refined = enhancer.enhance(0.95)
            
        return refined
        
    def _calculate_quality_metrics(
        self, 
        processed: Image.Image,
        original: Image.Image
    ) -> QualityMetrics:
        """Calcula métricas de qualidade da imagem processada"""
        
        # Converter para arrays
        proc_array = np.array(processed.convert('RGB'))
        orig_array = np.array(original.convert('RGB'))
        
        # 1. Blur score (Laplacian variance)
        gray = cv2.cvtColor(proc_array, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Contrast score (RMS contrast)
        contrast_score = np.std(gray)
        
        # 3. Saturation score (média da saturação HSV)
        hsv = rgb2hsv(proc_array / 255.0)
        saturation_score = np.mean(hsv[:,:,1]) * 100
        
        # 4. Noise level (estimativa)
        noise_level = self._estimate_noise_level(gray)
        
        # 5. Grass coverage (estimativa por cor)
        grass_coverage = self._estimate_grass_coverage(proc_array)
        
        # 6. Soil visibility
        soil_visibility = self._estimate_soil_visibility(proc_array)
        
        # 7. Overall score (combinação ponderada)
        overall_score = self._calculate_overall_score(
            blur_score, contrast_score, saturation_score, 
            noise_level, grass_coverage, soil_visibility
        )
        
        return QualityMetrics(
            blur_score=blur_score,
            contrast_score=contrast_score,
            saturation_score=saturation_score,
            noise_level=noise_level,
            grass_coverage=grass_coverage,
            soil_visibility=soil_visibility,
            overall_score=overall_score
        )
        
    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """Estima nível de ruído na imagem"""
        
        # Usar filtro Laplaciano para detectar ruído
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3)
        noise_variance = np.var(laplacian)
        
        # Normalizar para [0, 1]
        return min(noise_variance / 10000, 1.0)
        
    def _estimate_grass_coverage(self, rgb_image: np.ndarray) -> float:
        """Estima percentual de cobertura de gramíneas"""
        
        # Converter para HSV
        hsv = rgb2hsv(rgb_image / 255.0)
        
        # Máscara para tons verdes (gramíneas)
        green_mask = (hsv[:,:,0] > 0.2) & (hsv[:,:,0] < 0.4)
        moderate_sat = hsv[:,:,1] > 0.3
        moderate_val = hsv[:,:,2] > 0.2
        
        grass_mask = green_mask & moderate_sat & moderate_val
        coverage = np.sum(grass_mask) / grass_mask.size * 100
        
        return min(coverage, 100.0)
        
    def _estimate_soil_visibility(self, rgb_image: np.ndarray) -> float:
        """Estima percentual de solo exposto"""
        
        # Converter para HSV
        hsv = rgb2hsv(rgb_image / 255.0)
        
        # Máscara para tons de solo (vermelho, marrom, amarelo)
        red_soil = (hsv[:,:,0] < 0.08) | (hsv[:,:,0] > 0.9)
        brown_soil = (hsv[:,:,0] > 0.05) & (hsv[:,:,0] < 0.15)
        
        # Valores médios (não muito escuros nem muito claros)
        moderate_val = (hsv[:,:,2] > 0.3) & (hsv[:,:,2] < 0.8)
        low_sat = hsv[:,:,1] < 0.6  # Solo geralmente menos saturado
        
        soil_mask = (red_soil | brown_soil) & moderate_val & low_sat
        visibility = np.sum(soil_mask) / soil_mask.size * 100
        
        return min(visibility, 100.0)
        
    def _calculate_overall_score(
        self,
        blur_score: float,
        contrast_score: float, 
        saturation_score: float,
        noise_level: float,
        grass_coverage: float,
        soil_visibility: float
    ) -> float:
        """Calcula score geral de qualidade"""
        
        # Normalizar métricas
        blur_norm = min(blur_score / 1000, 1.0)          # Maior é melhor
        contrast_norm = min(contrast_score / 100, 1.0)   # Maior é melhor
        saturation_norm = min(saturation_score / 100, 1.0)  # Moderado é melhor
        noise_norm = 1 - noise_level                     # Menor é melhor
        
        # Balanceamento realista (pastagem ideal: 60-80% grama, 20-40% solo)
        grass_ideal = 1 - abs(grass_coverage - 70) / 70
        soil_ideal = 1 - abs(soil_visibility - 30) / 30
        
        # Média ponderada
        weights = [0.2, 0.15, 0.15, 0.15, 0.2, 0.15]  # [blur, contrast, sat, noise, grass, soil]
        scores = [blur_norm, contrast_norm, saturation_norm, noise_norm, grass_ideal, soil_ideal]
        
        overall = sum(w * s for w, s in zip(weights, scores))
        return max(0, min(overall, 1.0))
        
    def _save_intermediate_steps(self, steps: List, output_dir: str):
        """Salva passos intermediários do processamento"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for step_name, image in steps:
            filepath = output_path / f"step_{step_name}.jpg"
            image.save(filepath, quality=90)
            logger.info(f"💾 Passo salvo: {filepath}")
            
    def process_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        configs: Optional[List[ProcessingConfig]] = None,
        seasons: Optional[List[str]] = None,
        biomes: Optional[List[str]] = None
    ) -> List[Tuple[Image.Image, QualityMetrics]]:
        """Processa batch de imagens"""
        
        results = []
        
        for i, image in enumerate(images):
            config = configs[i] if configs else ProcessingConfig()
            season = seasons[i] if seasons else "chuvas"
            biome = biomes[i] if biomes else "cerrado"
            
            logger.info(f"Processando imagem {i+1}/{len(images)}")
            
            try:
                processed, metrics = self.process_image(image, config, season, biome)
                results.append((processed, metrics))
            except Exception as e:
                logger.error(f"Erro no processamento da imagem {i+1}: {e}")
                continue
                
        return results
        
    def save_quality_report(
        self, 
        metrics_list: List[QualityMetrics], 
        output_path: str
    ):
        """Salva relatório de qualidade do batch"""
        
        report = {
            'total_images': len(metrics_list),
            'average_scores': {
                'blur': np.mean([m.blur_score for m in metrics_list]),
                'contrast': np.mean([m.contrast_score for m in metrics_list]),
                'saturation': np.mean([m.saturation_score for m in metrics_list]),
                'noise': np.mean([m.noise_level for m in metrics_list]),
                'grass_coverage': np.mean([m.grass_coverage for m in metrics_list]),
                'soil_visibility': np.mean([m.soil_visibility for m in metrics_list]),
                'overall': np.mean([m.overall_score for m in metrics_list])
            },
            'quality_distribution': {
                'excellent': sum(1 for m in metrics_list if m.overall_score > 0.8),
                'good': sum(1 for m in metrics_list if 0.6 < m.overall_score <= 0.8),
                'moderate': sum(1 for m in metrics_list if 0.4 < m.overall_score <= 0.6),
                'poor': sum(1 for m in metrics_list if m.overall_score <= 0.4)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"✅ Relatório de qualidade salvo: {output_path}")
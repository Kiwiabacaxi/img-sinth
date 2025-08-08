"""
Sistema de métricas de qualidade para imagens sintéticas de pastagens
Avalia realismo, qualidade técnica e adequação agronômica
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from skimage import filters, measure, morphology, feature
from skimage.color import rgb2hsv, rgb2lab
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
from scipy import ndimage
import lpips

logger = logging.getLogger(__name__)

@dataclass
class QualityScore:
    """Score de qualidade individual"""
    metric_name: str
    score: float
    weight: float
    description: str

@dataclass
class QualityReport:
    """Relatório completo de qualidade"""
    overall_score: float
    technical_quality: float
    agricultural_realism: float
    seasonal_consistency: float
    individual_scores: List[QualityScore]
    recommendations: List[str]

class QualityMetrics:
    """
    Sistema de métricas de qualidade especializado em pastagens brasileiras
    Combina análise técnica e conhecimento agronômico
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Inicializar LPIPS para avaliação perceptual
        try:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
        except:
            logger.warning("LPIPS não disponível - algumas métricas serão desabilitadas")
            self.lpips_model = None
            
        # Parâmetros de qualidade por bioma
        self.biome_quality_params = {
            'cerrado': {
                'expected_soil_color_hsv': [(0, 40, 80), (15, 180, 150)],  # Vermelho laterítico
                'expected_grass_color_hsv': [(30, 40, 60), (80, 255, 200)],  # Verde a amarelo
                'soil_exposure_range': (0.1, 0.6),
                'termite_mound_density': (0, 8)  # por km²
            },
            'mata_atlantica': {
                'expected_soil_color_hsv': [(10, 30, 50), (25, 150, 120)],  # Marrom
                'expected_grass_color_hsv': [(40, 50, 80), (80, 255, 180)],  # Verde
                'soil_exposure_range': (0.05, 0.4),
                'forest_fragment_expected': True
            },
            'pampa': {
                'expected_soil_color_hsv': [(15, 20, 60), (30, 100, 110)],  # Planosol
                'expected_grass_color_hsv': [(35, 30, 70), (75, 200, 160)],  # Verde nativo
                'soil_exposure_range': (0.1, 0.5),
                'wind_effect_expected': True
            }
        }
        
        # Cores típicas de plantas invasoras (HSV)
        self.invasive_colors = {
            'capim_gordura': [(20, 80, 150), (35, 255, 255)],  # Dourado
            'carqueja': [(40, 30, 80), (60, 120, 140)],        # Verde-acinzentado
            'samambaia': [(50, 60, 100), (70, 180, 200)]       # Verde-brilhante
        }
        
    def evaluate_image_quality(
        self,
        image: Union[Image.Image, np.ndarray],
        metadata: Optional[Dict] = None,
        reference_image: Optional[Union[Image.Image, np.ndarray]] = None
    ) -> QualityReport:
        """
        Avalia qualidade completa da imagem
        
        Args:
            image: Imagem a ser avaliada
            metadata: Metadados da geração (bioma, estação, etc.)
            reference_image: Imagem de referência (opcional)
            
        Returns:
            Relatório completo de qualidade
        """
        
        # Converter para formato padrão
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            image = Image.fromarray(image) if image.dtype == np.uint8 else Image.fromarray((image * 255).astype(np.uint8))
            
        # Extrair metadados
        biome = metadata.get('biome', 'cerrado') if metadata else 'cerrado'
        season = metadata.get('season', 'chuvas') if metadata else 'chuvas'
        quality_level = metadata.get('quality', 'moderada') if metadata else 'moderada'
        
        individual_scores = []
        
        # 1. Qualidade Técnica
        tech_scores = self._evaluate_technical_quality(img_array)
        individual_scores.extend(tech_scores)
        
        # 2. Realismo Agronômico
        agro_scores = self._evaluate_agricultural_realism(img_array, biome, season)
        individual_scores.extend(agro_scores)
        
        # 3. Consistência Sazonal
        seasonal_scores = self._evaluate_seasonal_consistency(img_array, season, biome)
        individual_scores.extend(seasonal_scores)
        
        # 4. Análise de Cores Específicas
        color_scores = self._evaluate_color_consistency(img_array, biome, season)
        individual_scores.extend(color_scores)
        
        # 5. Análise de Textura e Padrões
        texture_scores = self._evaluate_texture_realism(img_array, biome)
        individual_scores.extend(texture_scores)
        
        # 6. Comparação Perceptual (se disponível)
        if reference_image is not None and self.lpips_model is not None:
            perceptual_scores = self._evaluate_perceptual_similarity(img_array, reference_image)
            individual_scores.extend(perceptual_scores)
            
        # Calcular scores agregados
        tech_score = np.mean([s.score for s in individual_scores if 'technical' in s.metric_name])
        agro_score = np.mean([s.score for s in individual_scores if 'agricultural' in s.metric_name])
        seasonal_score = np.mean([s.score for s in individual_scores if 'seasonal' in s.metric_name])
        
        # Score geral ponderado
        overall_score = (
            tech_score * 0.3 +
            agro_score * 0.4 +
            seasonal_score * 0.3
        )
        
        # Gerar recomendações
        recommendations = self._generate_recommendations(individual_scores, overall_score)
        
        report = QualityReport(
            overall_score=overall_score,
            technical_quality=tech_score,
            agricultural_realism=agro_score,
            seasonal_consistency=seasonal_score,
            individual_scores=individual_scores,
            recommendations=recommendations
        )
        
        logger.debug(f"Avaliação concluída - Score geral: {overall_score:.3f}")
        return report
        
    def _evaluate_technical_quality(self, image: np.ndarray) -> List[QualityScore]:
        """Avalia qualidade técnica da imagem"""
        
        scores = []
        
        # 1. Sharpness (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000, 1.0)  # Normalizar
        
        scores.append(QualityScore(
            metric_name="technical_sharpness",
            score=sharpness_score,
            weight=0.2,
            description="Image sharpness and focus quality"
        ))
        
        # 2. Contrast (RMS contrast)
        contrast = np.std(gray) / 128.0  # Normalizar para [0,1]
        contrast_score = min(contrast, 1.0)
        
        scores.append(QualityScore(
            metric_name="technical_contrast",
            score=contrast_score,
            weight=0.15,
            description="Image contrast and dynamic range"
        ))
        
        # 3. Noise level
        noise_level = self._estimate_noise_level(gray)
        noise_score = 1.0 - min(noise_level, 1.0)  # Inverter - menos ruído = melhor
        
        scores.append(QualityScore(
            metric_name="technical_noise",
            score=noise_score,
            weight=0.15,
            description="Noise level estimation"
        ))
        
        # 4. Color distribution
        color_score = self._evaluate_color_distribution(image)
        
        scores.append(QualityScore(
            metric_name="technical_color_distribution",
            score=color_score,
            weight=0.1,
            description="Color distribution and balance"
        ))
        
        # 5. Artifact detection
        artifact_score = self._detect_artifacts(image)
        
        scores.append(QualityScore(
            metric_name="technical_artifacts",
            score=artifact_score,
            weight=0.2,
            description="Presence of generation artifacts"
        ))
        
        return scores
        
    def _evaluate_agricultural_realism(
        self, 
        image: np.ndarray, 
        biome: str, 
        season: str
    ) -> List[QualityScore]:
        """Avalia realismo agronômico específico"""
        
        scores = []
        
        # 1. Cobertura de gramíneas
        grass_coverage = self._estimate_grass_coverage(image, biome)
        coverage_score = self._score_grass_coverage(grass_coverage, season)
        
        scores.append(QualityScore(
            metric_name="agricultural_grass_coverage",
            score=coverage_score,
            weight=0.25,
            description=f"Grass coverage realism ({grass_coverage:.1f}%)"
        ))
        
        # 2. Exposição do solo
        soil_exposure = self._estimate_soil_exposure(image, biome)
        soil_score = self._score_soil_exposure(soil_exposure, biome, season)
        
        scores.append(QualityScore(
            metric_name="agricultural_soil_exposure", 
            score=soil_score,
            weight=0.2,
            description=f"Soil exposure appropriateness ({soil_exposure:.1f}%)"
        ))
        
        # 3. Padrões de crescimento
        growth_pattern_score = self._evaluate_growth_patterns(image)
        
        scores.append(QualityScore(
            metric_name="agricultural_growth_patterns",
            score=growth_pattern_score,
            weight=0.2,
            description="Natural grass growth pattern realism"
        ))
        
        # 4. Características específicas do bioma
        biome_specific_score = self._evaluate_biome_characteristics(image, biome)
        
        scores.append(QualityScore(
            metric_name="agricultural_biome_characteristics",
            score=biome_specific_score,
            weight=0.15,
            description=f"Biome-specific characteristics ({biome})"
        ))
        
        # 5. Estrutura da vegetação
        vegetation_structure_score = self._evaluate_vegetation_structure(image, biome)
        
        scores.append(QualityScore(
            metric_name="agricultural_vegetation_structure",
            score=vegetation_structure_score,
            weight=0.2,
            description="Vegetation structure and organization"
        ))
        
        return scores
        
    def _evaluate_seasonal_consistency(
        self,
        image: np.ndarray,
        season: str,
        biome: str
    ) -> List[QualityScore]:
        """Avalia consistência sazonal"""
        
        scores = []
        
        # 1. Cores sazonais apropriadas
        seasonal_color_score = self._evaluate_seasonal_colors(image, season, biome)
        
        scores.append(QualityScore(
            metric_name="seasonal_color_consistency",
            score=seasonal_color_score,
            weight=0.4,
            description=f"Color consistency with {season} season"
        ))
        
        # 2. Condições de iluminação
        lighting_score = self._evaluate_seasonal_lighting(image, season)
        
        scores.append(QualityScore(
            metric_name="seasonal_lighting", 
            score=lighting_score,
            weight=0.3,
            description=f"Lighting consistency with {season}"
        ))
        
        # 3. Vigor da vegetação
        vegetation_vigor_score = self._evaluate_vegetation_vigor(image, season)
        
        scores.append(QualityScore(
            metric_name="seasonal_vegetation_vigor",
            score=vegetation_vigor_score,
            weight=0.3,
            description=f"Vegetation vigor for {season} season"
        ))
        
        return scores
        
    def _estimate_grass_coverage(self, image: np.ndarray, biome: str) -> float:
        """Estima percentual de cobertura de gramíneas"""
        
        # Converter para HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Obter parâmetros do bioma
        biome_params = self.biome_quality_params.get(biome, self.biome_quality_params['cerrado'])
        grass_color_range = biome_params['expected_grass_color_hsv']
        
        # Criar máscara para tons verdes/amarelos (gramíneas)
        lower_bound = np.array(grass_color_range[0])
        upper_bound = np.array(grass_color_range[1])
        
        grass_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Aplicar operações morfológicas para refinar
        kernel = np.ones((3,3), np.uint8)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel)
        
        # Calcular percentual
        coverage = (np.sum(grass_mask > 0) / grass_mask.size) * 100
        
        return min(coverage, 100.0)
        
    def _estimate_soil_exposure(self, image: np.ndarray, biome: str) -> float:
        """Estima percentual de solo exposto"""
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Parâmetros do solo por bioma
        biome_params = self.biome_quality_params.get(biome, self.biome_quality_params['cerrado'])
        soil_color_range = biome_params['expected_soil_color_hsv']
        
        # Criar máscara para tons de solo
        lower_bound = np.array(soil_color_range[0])
        upper_bound = np.array(soil_color_range[1])
        
        soil_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Refinar máscara
        kernel = np.ones((5,5), np.uint8)
        soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calcular percentual
        exposure = (np.sum(soil_mask > 0) / soil_mask.size) * 100
        
        return min(exposure, 100.0)
        
    def _score_grass_coverage(self, coverage: float, season: str) -> float:
        """Avalia se a cobertura é apropriada para a estação"""
        
        # Ranges esperados por estação
        expected_ranges = {
            'chuvas': (70, 95),    # Alta cobertura
            'seca': (45, 75),      # Cobertura reduzida
            'transicao': (55, 85)  # Cobertura intermediária
        }
        
        expected_min, expected_max = expected_ranges.get(season, expected_ranges['chuvas'])
        
        if expected_min <= coverage <= expected_max:
            return 1.0  # Perfeito
        elif coverage < expected_min:
            # Penalizar cobertura muito baixa
            return max(0, coverage / expected_min)
        else:
            # Penalizar cobertura muito alta (menos crítico)
            excess = coverage - expected_max
            penalty = min(excess / 20, 0.3)  # Máximo 30% de penalidade
            return max(0.7, 1.0 - penalty)
            
    def _score_soil_exposure(self, exposure: float, biome: str, season: str) -> float:
        """Avalia adequação da exposição do solo"""
        
        biome_params = self.biome_quality_params.get(biome, self.biome_quality_params['cerrado'])
        expected_min, expected_max = biome_params['soil_exposure_range']
        
        # Ajustar para estação
        if season == 'seca':
            expected_max *= 1.5  # Mais solo exposto na seca
        elif season == 'chuvas':
            expected_max *= 0.7  # Menos solo na chuva
            
        expected_max = min(expected_max * 100, 80)  # Converter para percentual e limitar
        expected_min = expected_min * 100
        
        if expected_min <= exposure <= expected_max:
            return 1.0
        elif exposure < expected_min:
            return max(0.5, exposure / expected_min)
        else:
            excess = exposure - expected_max
            penalty = min(excess / 30, 0.5)
            return max(0.5, 1.0 - penalty)
            
    def _evaluate_growth_patterns(self, image: np.ndarray) -> float:
        """Avalia naturalidade dos padrões de crescimento"""
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Analisar texturas usando LBP (Local Binary Patterns)
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        # Calcular entropia da textura (indica naturalidade)
        hist, _ = np.histogram(lbp.ravel(), bins=10, density=True)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Normalizar entropia (valores típicos: 2-4 para texturas naturais)
        entropy_score = min(entropy / 4.0, 1.0)
        
        # Analisar gradientes (transições suaves indicam naturalidade)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Score baseado na suavidade dos gradientes
        gradient_std = np.std(gradient_magnitude)
        gradient_score = 1.0 / (1.0 + gradient_std / 50.0)  # Normalizar
        
        # Combinar scores
        pattern_score = (entropy_score * 0.6 + gradient_score * 0.4)
        
        return pattern_score
        
    def _evaluate_biome_characteristics(self, image: np.ndarray, biome: str) -> float:
        """Avalia características específicas do bioma"""
        
        if biome == 'cerrado':
            return self._evaluate_cerrado_features(image)
        elif biome == 'mata_atlantica':
            return self._evaluate_mata_atlantica_features(image)
        elif biome == 'pampa':
            return self._evaluate_pampa_features(image)
        else:
            return 0.7  # Score neutro para biomas não específicos
            
    def _evaluate_cerrado_features(self, image: np.ndarray) -> float:
        """Avalia características do cerrado"""
        
        scores = []
        
        # 1. Presença de tons vermelhos do latossolo
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        red_soil_mask = cv2.inRange(hsv, np.array([0, 40, 80]), np.array([15, 180, 150]))
        red_soil_presence = np.sum(red_soil_mask > 0) / red_soil_mask.size
        
        # Score baseado na presença moderada de solo vermelho
        if 0.1 <= red_soil_presence <= 0.4:
            soil_score = 1.0
        elif red_soil_presence < 0.1:
            soil_score = red_soil_presence / 0.1
        else:
            soil_score = max(0.5, 1.0 - (red_soil_presence - 0.4) / 0.3)
            
        scores.append(soil_score)
        
        # 2. Padrão de vegetação esparsa (não muito densa)
        vegetation_density = self._calculate_vegetation_density(image)
        if 0.4 <= vegetation_density <= 0.8:
            density_score = 1.0
        else:
            density_score = max(0.5, 1.0 - abs(vegetation_density - 0.6) / 0.4)
            
        scores.append(density_score)
        
        return np.mean(scores)
        
    def _evaluate_mata_atlantica_features(self, image: np.ndarray) -> float:
        """Avalia características da mata atlântica"""
        
        scores = []
        
        # 1. Tons mais verdes e úmidos
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, np.array([40, 50, 80]), np.array([80, 255, 180]))
        green_presence = np.sum(green_mask > 0) / green_mask.size
        
        green_score = min(green_presence / 0.6, 1.0)  # Expect mais verde
        scores.append(green_score)
        
        # 2. Menor contraste (condições mais úmidas/nebulosas)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray)
        
        # Contraste moderado é esperado (não muito alto)
        if contrast < 40:
            contrast_score = 1.0
        else:
            contrast_score = max(0.5, 1.0 - (contrast - 40) / 60)
            
        scores.append(contrast_score)
        
        return np.mean(scores)
        
    def _evaluate_pampa_features(self, image: np.ndarray) -> float:
        """Avalia características do pampa"""
        
        scores = []
        
        # 1. Textura fina (gramíneas nativas pequenas)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Análise de textura em alta frequência
        high_freq = cv2.filter2D(gray, -1, np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]))
        high_freq_std = np.std(high_freq)
        
        # Textura fina tem alta variação em pequena escala
        texture_score = min(high_freq_std / 30, 1.0)
        scores.append(texture_score)
        
        # 2. Distribuição uniforme (menos patches) 
        uniformity_score = 1.0 - self._calculate_patchiness(image)
        scores.append(uniformity_score)
        
        return np.mean(scores)
        
    def _evaluate_seasonal_colors(self, image: np.ndarray, season: str, biome: str) -> float:
        """Avalia consistência das cores com a estação"""
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        if season == 'seca':
            # Expect mais tons amarelos/dourados
            golden_mask = cv2.inRange(hsv, np.array([15, 40, 100]), np.array([35, 255, 255]))
            golden_presence = np.sum(golden_mask > 0) / golden_mask.size
            
            # Penalizar excesso de verde na seca
            green_mask = cv2.inRange(hsv, np.array([45, 80, 80]), np.array([75, 255, 200]))
            green_presence = np.sum(green_mask > 0) / green_mask.size
            
            # Score baseado na proporção adequada
            color_score = min(golden_presence / 0.3, 1.0) * (1.0 - min(green_presence / 0.2, 0.5))
            
        elif season == 'chuvas':
            # Expect tons verdes vibrantes
            green_mask = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([80, 255, 200]))
            green_presence = np.sum(green_mask > 0) / green_mask.size
            
            color_score = min(green_presence / 0.6, 1.0)
            
        else:  # transicao
            # Mix de cores - avaliação mais flexível
            mixed_colors = self._calculate_color_diversity(hsv)
            color_score = min(mixed_colors / 0.5, 1.0)
            
        return color_score
        
    def _calculate_vegetation_density(self, image: np.ndarray) -> float:
        """Calcula densidade da vegetação"""
        
        # Usar NDVI aproximado (Green - Red) / (Green + Red)
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        
        # Evitar divisão por zero
        ndvi = np.divide(g.astype(float) - r.astype(float), 
                        g.astype(float) + r.astype(float) + 1e-10)
        
        # Densidade baseada em valores positivos de NDVI
        vegetation_pixels = np.sum(ndvi > 0.1)
        density = vegetation_pixels / ndvi.size
        
        return density
        
    def _calculate_patchiness(self, image: np.ndarray) -> float:
        """Calcula nível de irregularidade/patches na imagem"""
        
        # Converter para escala de cinza e segmentar
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Usar threshold adaptativo para segmentação
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Encontrar contornos (patches)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calcular métrica de patchiness baseada no número e tamanho dos contornos
        if len(contours) == 0:
            return 0.0
            
        areas = [cv2.contourArea(contour) for contour in contours]
        area_std = np.std(areas) if len(areas) > 1 else 0
        
        # Normalizar patchiness
        patchiness = min(len(contours) / 100 + area_std / 10000, 1.0)
        
        return patchiness
        
    def _calculate_color_diversity(self, hsv_image: np.ndarray) -> float:
        """Calcula diversidade de cores na imagem"""
        
        # Quantizar cores para reduzir ruído
        hue_bins = 18  # 20 graus por bin
        quantized_hue = (hsv_image[:,:,0] / 180 * hue_bins).astype(int)
        
        # Calcular entropia das cores
        unique, counts = np.unique(quantized_hue, return_counts=True)
        probabilities = counts / counts.sum()
        
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Normalizar entropia
        max_entropy = np.log2(hue_bins)
        diversity = entropy / max_entropy
        
        return diversity
        
    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """Estima nível de ruído na imagem"""
        
        # Usar filtro Laplaciano para detectar ruído
        laplacian = cv2.Laplacian(gray_image.astype(np.float32), cv2.CV_32F)
        noise_variance = np.var(laplacian)
        
        # Normalizar para [0, 1]
        normalized_noise = min(noise_variance / 1000, 1.0)
        
        return normalized_noise
        
    def _evaluate_color_distribution(self, image: np.ndarray) -> float:
        """Avalia distribuição e balance de cores"""
        
        # Converter para LAB para análise perceptual
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Analisar distribuição de luminância
        l_channel = lab[:,:,0]
        l_std = np.std(l_channel)
        
        # Score baseado na variação de luminância (contraste)
        luminance_score = min(l_std / 30, 1.0)
        
        # Analisar balance de cores A e B
        a_mean = np.mean(lab[:,:,1])
        b_mean = np.mean(lab[:,:,2])
        
        # Penalizar desvios extremos (cores não naturais)
        color_balance_score = 1.0 - min(abs(a_mean - 128) / 64, 0.5) - min(abs(b_mean - 128) / 64, 0.5)
        
        return (luminance_score * 0.6 + color_balance_score * 0.4)
        
    def _detect_artifacts(self, image: np.ndarray) -> float:
        """Detecta artefatos de geração"""
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Detectar bordas muito rígidas (não naturais)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Muitas bordas rígidas indicam artefatos
        edge_score = max(0, 1.0 - edge_density / 0.1)
        
        # 2. Detectar padrões repetitivos (indicam overfitting)
        # Usar autocorrelação para detectar repetições
        autocorr = cv2.matchTemplate(gray, gray[::2, ::2], cv2.TM_CCOEFF_NORMED)
        max_autocorr = np.max(autocorr)
        
        repetition_score = max(0, 1.0 - (max_autocorr - 0.5) / 0.4)
        
        # 3. Detectar saturação extrema
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        oversaturated = np.sum(hsv[:,:,1] > 240) / hsv[:,:,1].size
        
        saturation_score = max(0, 1.0 - oversaturated / 0.1)
        
        # Combinar scores
        artifact_score = (edge_score * 0.4 + repetition_score * 0.3 + saturation_score * 0.3)
        
        return artifact_score
        
    def _evaluate_vegetation_structure(self, image: np.ndarray, biome: str) -> float:
        """Avalia estrutura e organização da vegetação"""
        
        # Analisar padrões de crescimento usando segmentação
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Máscara de vegetação
        vegetation_mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([85, 255, 255]))
        
        # Aplicar operações morfológicas
        kernel = np.ones((5,5), np.uint8)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
        
        # Analisar conectividade e fragmentação
        num_labels, labels = cv2.connectedComponents(vegetation_mask)
        
        if num_labels <= 1:  # Sem vegetação detectada
            return 0.3
            
        # Calcular tamanhos dos fragmentos
        fragment_sizes = []
        for label in range(1, num_labels):
            size = np.sum(labels == label)
            fragment_sizes.append(size)
            
        if len(fragment_sizes) == 0:
            return 0.3
            
        # Analisar distribuição dos tamanhos
        fragment_sizes = np.array(fragment_sizes)
        
        # Estrutura natural tem poucos fragmentos grandes e muitos pequenos
        # Calcular coeficiente de variação
        cv = np.std(fragment_sizes) / np.mean(fragment_sizes)
        
        # Score baseado na variação natural
        structure_score = min(cv / 2.0, 1.0)
        
        return structure_score
        
    def _evaluate_seasonal_lighting(self, image: np.ndarray, season: str) -> float:
        """Avalia adequação da iluminação para a estação"""
        
        # Converter para LAB para análise de luminância
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        
        mean_luminance = np.mean(l_channel)
        luminance_std = np.std(l_channel)
        
        if season == 'seca':
            # Expect alta luminância e alto contraste (sol intenso)
            target_luminance = 140
            target_contrast = 35
        elif season == 'chuvas':
            # Expect luminância moderada e contraste suave (nublado)
            target_luminance = 110
            target_contrast = 25
        else:  # transicao
            # Valores intermediários
            target_luminance = 125
            target_contrast = 30
            
        # Score baseado na proximidade dos valores esperados
        luminance_score = 1.0 - min(abs(mean_luminance - target_luminance) / 50, 0.8)
        contrast_score = 1.0 - min(abs(luminance_std - target_contrast) / 20, 0.8)
        
        lighting_score = (luminance_score * 0.6 + contrast_score * 0.4)
        
        return lighting_score
        
    def _evaluate_vegetation_vigor(self, image: np.ndarray, season: str) -> float:
        """Avalia vigor da vegetação para a estação"""
        
        # Calcular "verdor" médio usando NDVI aproximado
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        
        ndvi = np.divide(g.astype(float) - r.astype(float), 
                        g.astype(float) + r.astype(float) + 1e-10)
        
        mean_ndvi = np.mean(ndvi)
        
        # Valores esperados por estação
        if season == 'chuvas':
            target_ndvi = 0.4  # Alto vigor
        elif season == 'seca':
            target_ndvi = 0.1  # Baixo vigor
        else:  # transicao
            target_ndvi = 0.25  # Vigor moderado
            
        # Score baseado na proximidade do valor esperado
        vigor_score = 1.0 - min(abs(mean_ndvi - target_ndvi) / 0.3, 1.0)
        
        return vigor_score
        
    def _evaluate_perceptual_similarity(
        self,
        image: np.ndarray,
        reference_image: Union[Image.Image, np.ndarray]
    ) -> List[QualityScore]:
        """Avalia similaridade perceptual usando LPIPS"""
        
        scores = []
        
        if self.lpips_model is None:
            return scores
            
        try:
            # Preparar imagens para LPIPS
            if isinstance(reference_image, Image.Image):
                ref_array = np.array(reference_image)
            else:
                ref_array = reference_image
                
            # Converter para tensores PyTorch
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            ref_tensor = torch.from_numpy(ref_array).permute(2, 0, 1).float() / 255.0
            
            # Redimensionar se necessário
            if img_tensor.shape[-2:] != ref_tensor.shape[-2:]:
                ref_tensor = F.interpolate(ref_tensor.unsqueeze(0), 
                                         size=img_tensor.shape[-2:], 
                                         mode='bilinear').squeeze(0)
                                         
            # Adicionar batch dimension
            img_batch = img_tensor.unsqueeze(0).to(self.device)
            ref_batch = ref_tensor.unsqueeze(0).to(self.device)
            
            # Calcular LPIPS
            lpips_distance = self.lpips_model(img_batch, ref_batch).item()
            
            # Converter distância para score (menor distância = maior score)
            lpips_score = max(0, 1.0 - lpips_distance / 0.5)
            
            scores.append(QualityScore(
                metric_name="perceptual_similarity",
                score=lpips_score,
                weight=0.3,
                description=f"Perceptual similarity to reference (LPIPS: {lpips_distance:.3f})"
            ))
            
        except Exception as e:
            logger.warning(f"Erro na avaliação perceptual: {e}")
            
        return scores
        
    def _generate_recommendations(
        self,
        individual_scores: List[QualityScore],
        overall_score: float
    ) -> List[str]:
        """Gera recomendações baseadas nos scores"""
        
        recommendations = []
        
        # Analisar scores baixos
        low_scores = [s for s in individual_scores if s.score < 0.6]
        
        for score in low_scores:
            if 'sharpness' in score.metric_name:
                recommendations.append("Increase sampling steps or adjust guidance scale for better sharpness")
            elif 'noise' in score.metric_name:
                recommendations.append("Enable or tune denoising parameters")
            elif 'grass_coverage' in score.metric_name:
                recommendations.append("Adjust prompt to better specify grass coverage for the season")
            elif 'soil_exposure' in score.metric_name:
                recommendations.append("Fine-tune soil visibility prompts for the specific biome")
            elif 'seasonal' in score.metric_name:
                recommendations.append("Review seasonal color consistency in prompts")
            elif 'artifacts' in score.metric_name:
                recommendations.append("Reduce guidance scale or increase image resolution")
                
        # Recomendações gerais baseadas no score geral
        if overall_score < 0.5:
            recommendations.append("Overall quality is low - consider reviewing prompt engineering")
        elif overall_score < 0.7:
            recommendations.append("Moderate quality - focus on improving lowest scoring metrics")
            
        if not recommendations:
            recommendations.append("Image quality is good - no specific improvements needed")
            
        return recommendations
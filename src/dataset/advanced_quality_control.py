"""
Sistema avançado de controle de qualidade para imagens sintéticas de pastagens
Integra métricas científicas, filtros automáticos e análise de bias
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.multimodal.clip_score import CLIPScore
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class AdvancedQualityMetrics:
    """Métricas avançadas de qualidade"""
    
    # Métricas técnicas básicas
    technical_score: float = 0.0
    sharpness_score: float = 0.0
    exposure_score: float = 0.0
    contrast_score: float = 0.0
    noise_level: float = 0.0
    
    # Métricas científicas avançadas
    fid_score: Optional[float] = None
    clip_score: Optional[float] = None
    inception_score: Optional[float] = None
    
    # Métricas específicas para agricultura
    vegetation_index: float = 0.0
    soil_visibility: float = 0.0
    seasonal_consistency: float = 0.0
    biome_authenticity: float = 0.0
    
    # Detecção de artefatos
    has_artifacts: bool = False
    artifact_types: List[str] = None
    artifact_severity: float = 0.0
    
    # Diversidade e representatividade
    diversity_score: float = 0.0
    uniqueness_score: float = 0.0
    
    # Score geral combinado
    overall_quality_score: float = 0.0

@dataclass
class QualityControlConfig:
    """Configuração do sistema de controle de qualidade"""
    
    # Thresholds de qualidade
    min_technical_score: float = 0.7
    min_fid_score: float = 50.0  # FID menor é melhor
    min_clip_score: float = 0.25
    min_vegetation_index: float = 0.3
    
    # Configurações de análise
    enable_fid_calculation: bool = True
    enable_clip_analysis: bool = True
    enable_artifact_detection: bool = True
    enable_diversity_analysis: bool = True
    
    # Filtros automáticos
    auto_reject_artifacts: bool = True
    auto_reject_low_quality: bool = True
    save_rejected_images: bool = False
    
    # Configurações de processamento
    batch_size: int = 32
    use_gpu: bool = True
    cache_features: bool = True

class AdvancedQualityController:
    """
    Sistema avançado de controle de qualidade para imagens sintéticas
    """
    
    def __init__(
        self,
        config: Optional[QualityControlConfig] = None,
        reference_dataset_path: Optional[str] = None
    ):
        self.config = config or QualityControlConfig()
        self.reference_dataset_path = reference_dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.use_gpu else 'cpu')
        
        # Inicializar métricas científicas
        self.fid_metric = None
        self.clip_metric = None
        self.clip_model = None
        
        if FID_AVAILABLE and self.config.enable_fid_calculation:
            self.fid_metric = FrechetInceptionDistance(normalize=True).to(self.device)
            
        if CLIP_AVAILABLE and self.config.enable_clip_analysis:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                self.clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(self.device)
            except Exception as e:
                logger.warning(f"Erro ao carregar CLIP: {e}")
                self.clip_model = None
        
        # Cache para features de referência
        self.reference_features_cache = {}
        self.analyzed_images_cache = {}
        
        # Padrões de artefatos conhecidos
        self.artifact_detectors = self._initialize_artifact_detectors()
        
        # Estatísticas de qualidade
        self.quality_statistics = {
            'total_analyzed': 0,
            'accepted': 0,
            'rejected': 0,
            'artifact_detections': {},
            'quality_distribution': []
        }
        
        logger.info(f"AdvancedQualityController inicializado - Device: {self.device}")
        
    def _initialize_artifact_detectors(self) -> Dict:
        """Inicializa detectores de artefatos específicos"""
        
        return {
            'blur_detection': self._detect_blur,
            'noise_detection': self._detect_noise,
            'compression_artifacts': self._detect_compression_artifacts,
            'color_artifacts': self._detect_color_artifacts,
            'geometric_distortions': self._detect_geometric_distortions,
            'unnatural_patterns': self._detect_unnatural_patterns,
            'diffusion_artifacts': self._detect_diffusion_specific_artifacts
        }
    
    def analyze_image_quality(
        self,
        image: Union[str, Image.Image, np.ndarray],
        metadata: Optional[Dict] = None,
        reference_images: Optional[List] = None
    ) -> AdvancedQualityMetrics:
        """
        Análise completa de qualidade de uma imagem
        
        Args:
            image: Imagem para análise
            metadata: Metadados da imagem (bioma, estação, etc.)
            reference_images: Imagens de referência para comparação
            
        Returns:
            Métricas avançadas de qualidade
        """
        
        # Converter imagem para formato padrão
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Converter para numpy para análises técnicas
        img_array = np.array(image)
        
        # Análises técnicas básicas
        technical_metrics = self._analyze_technical_quality(img_array)
        
        # Análises científicas avançadas
        scientific_metrics = {}
        if reference_images:
            scientific_metrics = self._analyze_scientific_quality(image, reference_images)
        
        # Análises específicas para agricultura
        agricultural_metrics = self._analyze_agricultural_quality(img_array, metadata)
        
        # Detecção de artefatos
        artifact_analysis = self._detect_all_artifacts(img_array)
        
        # Análise de diversidade
        diversity_metrics = self._analyze_diversity(image)
        
        # Combinar todas as métricas
        metrics = AdvancedQualityMetrics(
            # Técnicas
            technical_score=technical_metrics['overall'],
            sharpness_score=technical_metrics['sharpness'],
            exposure_score=technical_metrics['exposure'],
            contrast_score=technical_metrics['contrast'],
            noise_level=technical_metrics['noise'],
            
            # Científicas
            fid_score=scientific_metrics.get('fid_score'),
            clip_score=scientific_metrics.get('clip_score'),
            
            # Agricultura
            vegetation_index=agricultural_metrics['vegetation_index'],
            soil_visibility=agricultural_metrics['soil_visibility'],
            seasonal_consistency=agricultural_metrics['seasonal_consistency'],
            biome_authenticity=agricultural_metrics['biome_authenticity'],
            
            # Artefatos
            has_artifacts=artifact_analysis['has_artifacts'],
            artifact_types=artifact_analysis['types'],
            artifact_severity=artifact_analysis['severity'],
            
            # Diversidade
            diversity_score=diversity_metrics['diversity_score'],
            uniqueness_score=diversity_metrics['uniqueness_score']
        )
        
        # Calcular score geral
        metrics.overall_quality_score = self._calculate_overall_score(metrics)
        
        # Atualizar estatísticas
        self._update_statistics(metrics)
        
        return metrics
    
    def _analyze_technical_quality(self, img_array: np.ndarray) -> Dict[str, float]:
        """Análise técnica da qualidade da imagem"""
        
        # Converter para grayscale para algumas análises
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 1. Análise de nitidez (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 1000.0)  # Normalizar
        
        # 2. Análise de exposição
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Detectar sub/super-exposição
        underexposed = np.sum(hist[:20]) / total_pixels
        overexposed = np.sum(hist[235:]) / total_pixels
        exposure_score = 1.0 - max(underexposed, overexposed) * 2
        
        # 3. Análise de contraste (RMS contrast)
        contrast = gray.std()
        contrast_score = min(1.0, contrast / 64.0)  # Normalizar
        
        # 4. Análise de ruído (estimativa via wavelet)
        noise_level = self._estimate_noise_level(gray)
        
        # Score técnico geral
        technical_score = np.mean([
            sharpness_score * 0.3,
            exposure_score * 0.25,
            contrast_score * 0.25,
            max(0, 1.0 - noise_level) * 0.2  # Ruído baixo é bom
        ])
        
        return {
            'overall': technical_score,
            'sharpness': sharpness_score,
            'exposure': exposure_score,
            'contrast': contrast_score,
            'noise': noise_level
        }
    
    def _analyze_scientific_quality(self, image: Image.Image, reference_images: List) -> Dict[str, float]:
        """Análise usando métricas científicas (FID, CLIP)"""
        
        scientific_metrics = {}
        
        # FID Score
        if self.fid_metric is not None and reference_images:
            try:
                # Preparar imagem atual
                img_tensor = self._image_to_tensor(image).unsqueeze(0).to(self.device)
                
                # Preparar imagens de referência
                ref_tensors = []
                for ref_img in reference_images[:50]:  # Limitar para performance
                    if isinstance(ref_img, str):
                        ref_img = Image.open(ref_img).convert('RGB')
                    ref_tensor = self._image_to_tensor(ref_img).unsqueeze(0)
                    ref_tensors.append(ref_tensor)
                
                ref_batch = torch.cat(ref_tensors).to(self.device)
                
                # Calcular FID
                self.fid_metric.update(ref_batch, real=True)
                self.fid_metric.update(img_tensor, real=False)
                fid_score = self.fid_metric.compute().item()
                
                scientific_metrics['fid_score'] = fid_score
                
                # Reset para próxima análise
                self.fid_metric.reset()
                
            except Exception as e:
                logger.warning(f"Erro no cálculo FID: {e}")
        
        # CLIP Score
        if self.clip_model is not None:
            try:
                # Gerar prompts baseados em características da pastagem
                prompts = [
                    "healthy brazilian pasture with native grass species",
                    "natural grassland landscape with diverse vegetation",
                    "realistic agricultural field with good vegetation cover"
                ]
                
                # Calcular CLIP score médio
                clip_scores = []
                for prompt in prompts:
                    score = self._calculate_clip_score(image, prompt)
                    if score is not None:
                        clip_scores.append(score)
                
                if clip_scores:
                    scientific_metrics['clip_score'] = np.mean(clip_scores)
                
            except Exception as e:
                logger.warning(f"Erro no cálculo CLIP: {e}")
        
        return scientific_metrics
    
    def _analyze_agricultural_quality(self, img_array: np.ndarray, metadata: Optional[Dict]) -> Dict[str, float]:
        """Análise específica para qualidade agronômica"""
        
        # 1. Índice de vegetação (aproximação do NDVI usando RGB)
        r_band = img_array[:, :, 0].astype(np.float32)
        g_band = img_array[:, :, 1].astype(np.float32)
        
        # NDVI aproximado: (G-R)/(G+R)
        ndvi_approx = np.divide(
            g_band - r_band,
            g_band + r_band + 1e-8,  # Evitar divisão por zero
            out=np.zeros_like(g_band),
            where=(g_band + r_band) != 0
        )
        
        vegetation_index = np.clip(np.mean(ndvi_approx[ndvi_approx > 0]), 0, 1)
        
        # 2. Visibilidade do solo (pixels com baixa vegetação)
        # Detectar pixels de solo (tons marrons/vermelhos)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Máscara para solo (ajustar baseado em conhecimento de solos brasileiros)
        soil_mask = (
            ((hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 30)) |  # Tons avermelhados
            ((hsv[:, :, 0] >= 120) & (hsv[:, :, 0] <= 180))  # Tons marrons
        ) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 30)
        
        soil_visibility = np.sum(soil_mask) / soil_mask.size
        
        # 3. Consistência sazonal
        seasonal_consistency = self._evaluate_seasonal_consistency(img_array, metadata)
        
        # 4. Autenticidade do bioma
        biome_authenticity = self._evaluate_biome_authenticity(img_array, metadata)
        
        return {
            'vegetation_index': vegetation_index,
            'soil_visibility': soil_visibility,
            'seasonal_consistency': seasonal_consistency,
            'biome_authenticity': biome_authenticity
        }
    
    def _detect_all_artifacts(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Executa todos os detectores de artefatos"""
        
        detected_artifacts = []
        artifact_scores = {}
        
        for detector_name, detector_func in self.artifact_detectors.items():
            try:
                has_artifact, severity = detector_func(img_array)
                artifact_scores[detector_name] = severity
                
                if has_artifact:
                    detected_artifacts.append(detector_name)
                    
            except Exception as e:
                logger.warning(f"Erro no detector {detector_name}: {e}")
        
        # Calcular severidade geral
        overall_severity = np.mean(list(artifact_scores.values())) if artifact_scores else 0.0
        
        return {
            'has_artifacts': len(detected_artifacts) > 0,
            'types': detected_artifacts,
            'severity': overall_severity,
            'detailed_scores': artifact_scores
        }
    
    # Implementações dos detectores específicos
    def _detect_blur(self, img_array: np.ndarray) -> Tuple[bool, float]:
        """Detecta blur excessivo"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Threshold baseado em experiência
        blur_threshold = 100  # Ajustar baseado em testes
        is_blurry = laplacian_var < blur_threshold
        severity = 1.0 - min(1.0, laplacian_var / blur_threshold)
        
        return is_blurry, severity
    
    def _detect_noise(self, img_array: np.ndarray) -> Tuple[bool, float]:
        """Detecta ruído excessivo"""
        noise_level = self._estimate_noise_level(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
        
        noise_threshold = 0.15
        has_noise = noise_level > noise_threshold
        severity = min(1.0, noise_level / noise_threshold) if has_noise else 0.0
        
        return has_noise, severity
    
    def _detect_compression_artifacts(self, img_array: np.ndarray) -> Tuple[bool, float]:
        """Detecta artefatos de compressão"""
        # Análise de blocos 8x8 típicos de JPEG
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # FFT para detectar padrões regulares
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Detectar picos regulares no espectro (simplificado)
        # Em implementação real, seria mais sofisticado
        compression_score = 0.1  # Placeholder
        
        has_compression = compression_score > 0.3
        return has_compression, compression_score
    
    def _detect_color_artifacts(self, img_array: np.ndarray) -> Tuple[bool, float]:
        """Detecta artefatos de cor (cores não naturais, oversaturation)"""
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Detectar supersaturação
        high_saturation = np.sum(hsv[:, :, 1] > 200) / hsv[:, :, 1].size
        
        # Detectar cores não naturais para pastagens
        # Pastagens devem ter principalmente verdes, marrons, amarelos
        unnatural_colors = self._detect_unnatural_colors(img_array)
        
        color_artifact_score = max(high_saturation, unnatural_colors)
        has_color_artifacts = color_artifact_score > 0.2
        
        return has_color_artifacts, color_artifact_score
    
    def _detect_geometric_distortions(self, img_array: np.ndarray) -> Tuple[bool, float]:
        """Detecta distorções geométricas"""
        # Detectar linhas muito curvadas onde deveriam ser retas
        # Em pastagens, horizonte deve ser relativamente reto
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Análise simplificada - em implementação real seria mais complexa
        distortion_score = 0.05  # Placeholder
        
        has_distortion = distortion_score > 0.3
        return has_distortion, distortion_score
    
    def _detect_unnatural_patterns(self, img_array: np.ndarray) -> Tuple[bool, float]:
        """Detecta padrões não naturais (repetitivos, artificiais)"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detectar padrões repetitivos usando autocorrelação
        # Implementação simplificada
        unnatural_score = 0.1  # Placeholder
        
        has_unnatural = unnatural_score > 0.4
        return has_unnatural, unnatural_score
    
    def _detect_diffusion_specific_artifacts(self, img_array: np.ndarray) -> Tuple[bool, float]:
        """Detecta artefatos específicos de modelos de diffusion"""
        # Artefatos comuns: texturas inconsistentes, objetos fundidos, etc.
        
        # Análise de textura usando LBP ou similar
        # Detecção de objetos com bordas mal definidas
        # Implementação simplificada por ora
        
        diffusion_artifact_score = 0.05  # Placeholder
        
        has_diffusion_artifacts = diffusion_artifact_score > 0.3
        return has_diffusion_artifacts, diffusion_artifact_score
    
    # Funções auxiliares
    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """Estima o nível de ruído na imagem"""
        # Método baseado em wavelet (aproximação)
        H, W = gray_image.shape
        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
        
        M = np.array(M)
        sigma = np.sum(np.sum(np.absolute(cv2.filter2D(gray_image.astype(np.float32), -1, M))))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
        
        return min(1.0, sigma / 255.0)  # Normalizar
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Converte PIL Image para tensor normalizado"""
        # Redimensionar para 299x299 (padrão Inception)
        image = image.resize((299, 299))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
        
        # Normalização ImageNet
        normalize = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img_tensor = (img_tensor - normalize) / std
        
        return img_tensor
    
    def _calculate_clip_score(self, image: Image.Image, prompt: str) -> Optional[float]:
        """Calcula CLIP score para imagem e prompt"""
        try:
            # Preprocessar imagem
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([prompt]).to(self.device)
            
            # Calcular features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                # Normalizar
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Calcular similaridade
                similarity = torch.cosine_similarity(image_features, text_features).item()
                
            return similarity
            
        except Exception as e:
            logger.warning(f"Erro no cálculo CLIP score: {e}")
            return None
    
    def _evaluate_seasonal_consistency(self, img_array: np.ndarray, metadata: Optional[Dict]) -> float:
        """Avalia consistência com a estação especificada"""
        if not metadata or 'season' not in metadata:
            return 0.5  # Neutro se não há informação
        
        season = metadata['season']
        
        # Análise de cor dominante
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        if season == 'seca':
            # Esperar mais tons amarelados/marrons
            yellow_brown_mask = (
                ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 45)) |
                ((hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 15))
            )
            consistency = np.sum(yellow_brown_mask) / yellow_brown_mask.size
            
        elif season == 'chuvas':
            # Esperar mais tons verdes
            green_mask = (hsv[:, :, 0] >= 45) & (hsv[:, :, 0] <= 85)
            consistency = np.sum(green_mask) / green_mask.size
            
        else:  # transicao
            # Mistura de cores
            consistency = 0.7  # Valor intermediário
        
        return min(1.0, consistency * 2)  # Amplificar um pouco
    
    def _evaluate_biome_authenticity(self, img_array: np.ndarray, metadata: Optional[Dict]) -> float:
        """Avalia autenticidade para o bioma especificado"""
        if not metadata or 'biome' not in metadata:
            return 0.5  # Neutro se não há informação
        
        biome = metadata['biome']
        
        # Análise simplificada por bioma
        # Em implementação real, usaria modelos mais sofisticados
        
        if biome == 'cerrado':
            # Cerrado: solos mais avermelhados, vegetação mais aberta
            authenticity = self._analyze_cerrado_features(img_array)
            
        elif biome == 'mata_atlantica':
            # Mata Atlântica: mais verdejante, topografia mais ondulada
            authenticity = self._analyze_mata_atlantica_features(img_array)
            
        elif biome == 'pampa':
            # Pampa: campos mais uniformes, menos árvores
            authenticity = self._analyze_pampa_features(img_array)
            
        else:
            authenticity = 0.5
        
        return authenticity
    
    def _analyze_diversity(self, image: Image.Image) -> Dict[str, float]:
        """Análise de diversidade da imagem"""
        
        # Diversidade de cores
        img_array = np.array(image)
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
        max_possible_colors = 256 ** 3  # RGB
        color_diversity = min(1.0, unique_colors / 10000)  # Normalizar
        
        # Diversidade de texturas (simplificada)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        texture_measure = np.std(gray) / 128.0  # Normalizar
        
        # Score de unicidade (comparar com cache se disponível)
        uniqueness_score = 1.0  # Placeholder - compararia com imagens anteriores
        
        diversity_score = np.mean([color_diversity, texture_measure])
        
        return {
            'diversity_score': diversity_score,
            'uniqueness_score': uniqueness_score,
            'color_diversity': color_diversity,
            'texture_diversity': texture_measure
        }
    
    def _calculate_overall_score(self, metrics: AdvancedQualityMetrics) -> float:
        """Calcula score geral combinando todas as métricas"""
        
        scores = []
        weights = []
        
        # Score técnico (peso alto)
        scores.append(metrics.technical_score)
        weights.append(0.25)
        
        # Métricas científicas
        if metrics.fid_score is not None:
            # FID: menor é melhor, converter para score 0-1
            fid_score = max(0, 1.0 - metrics.fid_score / 100.0)
            scores.append(fid_score)
            weights.append(0.2)
        
        if metrics.clip_score is not None:
            scores.append(metrics.clip_score)
            weights.append(0.15)
        
        # Métricas agrícolas
        agricultural_score = np.mean([
            metrics.vegetation_index,
            1.0 - metrics.soil_visibility,  # Solo menos visível é melhor
            metrics.seasonal_consistency,
            metrics.biome_authenticity
        ])
        scores.append(agricultural_score)
        weights.append(0.25)
        
        # Penalizar artefatos
        artifact_penalty = 1.0 - metrics.artifact_severity
        scores.append(artifact_penalty)
        weights.append(0.15)
        
        # Score de diversidade
        scores.append(metrics.diversity_score)
        weights.append(0.1)
        
        # Normalizar pesos
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calcular score ponderado
        overall_score = np.average(scores, weights=weights)
        
        return float(np.clip(overall_score, 0, 1))
    
    def _update_statistics(self, metrics: AdvancedQualityMetrics):
        """Atualiza estatísticas internas"""
        
        self.quality_statistics['total_analyzed'] += 1
        self.quality_statistics['quality_distribution'].append(metrics.overall_quality_score)
        
        # Estatísticas de artefatos
        for artifact_type in metrics.artifact_types or []:
            if artifact_type not in self.quality_statistics['artifact_detections']:
                self.quality_statistics['artifact_detections'][artifact_type] = 0
            self.quality_statistics['artifact_detections'][artifact_type] += 1
    
    # Análises específicas por bioma (implementações simplificadas)
    def _analyze_cerrado_features(self, img_array: np.ndarray) -> float:
        """Análise específica para Cerrado"""
        # Cerrado: solos vermelhos, vegetação mais esparsa
        # Implementação simplificada
        return 0.7  # Placeholder
    
    def _analyze_mata_atlantica_features(self, img_array: np.ndarray) -> float:
        """Análise específica para Mata Atlântica"""
        # Mata Atlântica: mais verde, topografia ondulada
        # Implementação simplificada
        return 0.8  # Placeholder
    
    def _analyze_pampa_features(self, img_array: np.ndarray) -> float:
        """Análise específica para Pampa"""
        # Pampa: campos mais uniformes
        # Implementação simplificada
        return 0.75  # Placeholder
    
    def _detect_unnatural_colors(self, img_array: np.ndarray) -> float:
        """Detecta cores não naturais para pastagens"""
        # Cores naturais: verdes, marrons, amarelos
        # Cores não naturais: azuis vibrantes, magentas, etc.
        
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Definir ranges de cores naturais para pastagens
        natural_hue_ranges = [
            (0, 30),    # Vermelhos/marrons
            (45, 85),   # Verdes
            (15, 45)    # Amarelos/laranjas
        ]
        
        natural_mask = np.zeros(hsv[:, :, 0].shape, dtype=bool)
        for h_min, h_max in natural_hue_ranges:
            natural_mask |= (hsv[:, :, 0] >= h_min) & (hsv[:, :, 0] <= h_max)
        
        unnatural_ratio = np.sum(~natural_mask) / natural_mask.size
        return unnatural_ratio
    
    def get_quality_statistics(self) -> Dict:
        """Retorna estatísticas de qualidade coletadas"""
        
        stats = self.quality_statistics.copy()
        
        if stats['quality_distribution']:
            stats['average_quality'] = np.mean(stats['quality_distribution'])
            stats['quality_std'] = np.std(stats['quality_distribution'])
            stats['quality_percentiles'] = {
                '25th': np.percentile(stats['quality_distribution'], 25),
                '50th': np.percentile(stats['quality_distribution'], 50),
                '75th': np.percentile(stats['quality_distribution'], 75),
                '95th': np.percentile(stats['quality_distribution'], 95)
            }
        
        return stats
    
    def generate_quality_report(self, output_path: str):
        """Gera relatório detalhado de qualidade"""
        
        stats = self.get_quality_statistics()
        report_data = {
            'generation_date': datetime.now().isoformat(),
            'configuration': asdict(self.config),
            'statistics': stats,
            'analysis_summary': {
                'total_images_analyzed': stats['total_analyzed'],
                'average_quality_score': stats.get('average_quality', 0),
                'most_common_artifacts': sorted(
                    stats['artifact_detections'].items(),
                    key=lambda x: x[1], reverse=True
                )[:5] if stats['artifact_detections'] else []
            }
        }
        
        # Salvar JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Relatório de qualidade salvo: {output_path}")

def quick_quality_check(image_path: str) -> Dict[str, float]:
    """Função helper para verificação rápida de qualidade"""
    
    controller = AdvancedQualityController()
    metrics = controller.analyze_image_quality(image_path)
    
    return {
        'overall_score': metrics.overall_quality_score,
        'technical_score': metrics.technical_score,
        'has_artifacts': metrics.has_artifacts,
        'vegetation_index': metrics.vegetation_index
    }
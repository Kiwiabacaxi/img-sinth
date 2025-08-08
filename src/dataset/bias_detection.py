"""
Sistema de detecção e análise de bias em datasets sintéticos de pastagens
Identifica desequilíbrios e problemas de representatividade
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class BiasAnalysisResult:
    """Resultado da análise de bias"""
    
    # Bias geral
    overall_bias_score: float = 0.0
    bias_severity: str = "low"  # low, medium, high, critical
    
    # Bias por dimensões
    biome_bias: Dict[str, float] = None
    seasonal_bias: Dict[str, float] = None
    quality_bias: Dict[str, float] = None
    class_bias: Dict[str, float] = None
    
    # Bias de representatividade
    underrepresented_groups: List[str] = None
    overrepresented_groups: List[str] = None
    
    # Bias visual/técnico
    color_bias: Dict[str, Any] = None
    composition_bias: Dict[str, Any] = None
    style_bias: Dict[str, Any] = None
    
    # Bias temporal/geográfico
    temporal_bias: Dict[str, float] = None
    geographic_bias: Dict[str, float] = None
    
    # Recomendações de correção
    recommendations: List[str] = None
    priority_actions: List[str] = None
    
    # Métricas estatísticas
    statistical_tests: Dict[str, Dict] = None
    confidence_intervals: Dict[str, Tuple[float, float]] = None

@dataclass
class BiasDetectionConfig:
    """Configuração para detecção de bias"""
    
    # Thresholds de significância
    bias_threshold_low: float = 0.15
    bias_threshold_medium: float = 0.30
    bias_threshold_high: float = 0.50
    
    # Configurações de análise
    min_samples_per_group: int = 100
    statistical_significance: float = 0.05
    
    # Análises específicas
    enable_visual_bias_detection: bool = True
    enable_demographic_bias_analysis: bool = True
    enable_temporal_bias_analysis: bool = True
    enable_quality_bias_analysis: bool = True
    
    # Configurações de clustering
    n_visual_clusters: int = 10
    
    # Output
    save_detailed_plots: bool = True
    generate_bias_report: bool = True

class BiasDetector:
    """
    Sistema de detecção de bias em datasets sintéticos
    """
    
    def __init__(self, config: Optional[BiasDetectionConfig] = None):
        self.config = config or BiasDetectionConfig()
        
        # Definições de grupos esperados
        self.expected_distributions = {
            'biome': {
                'cerrado': 0.45,      # 45% - maior bioma
                'mata_atlantica': 0.35, # 35% - segundo maior
                'pampa': 0.20         # 20% - menor bioma
            },
            'season': {
                'seca': 0.4,          # 40% - época crítica
                'chuvas': 0.4,        # 40% - crescimento
                'transicao': 0.2      # 20% - intermediário
            },
            'quality': {
                'boa': 0.25,          # 25% - bem manejada
                'moderada': 0.50,     # 50% - mais comum
                'degradada': 0.25     # 25% - problema real
            }
        }
        
        # Grupos críticos para monitoramento
        self.critical_groups = {
            'invasive_species': ['capim_gordura', 'capim_coloniao', 'capim_annoni'],
            'degradation_indicators': ['area_degradada', 'solo_exposto', 'cupinzeiro'],
            'native_species': ['graminea_nativa_cerrado', 'graminea_nativa_mata_atlantica']
        }
        
        # Cache para análises
        self.visual_features_cache = {}
        self.analysis_cache = {}
        
        logger.info("BiasDetector inicializado")
    
    def analyze_dataset_bias(
        self,
        dataset_path: str,
        metadata_path: Optional[str] = None,
        sample_size: Optional[int] = None
    ) -> BiasAnalysisResult:
        """
        Análise completa de bias do dataset
        
        Args:
            dataset_path: Caminho do dataset
            metadata_path: Caminho dos metadados
            sample_size: Tamanho da amostra (None = usar tudo)
            
        Returns:
            Resultado da análise de bias
        """
        
        logger.info("🔍 Iniciando análise de bias do dataset")
        
        # Carregar dados e metadados
        dataset_info = self._load_dataset_info(dataset_path, metadata_path, sample_size)
        
        if not dataset_info['images']:
            logger.error("Nenhuma imagem encontrada para análise")
            return BiasAnalysisResult()
        
        # Análises de diferentes tipos de bias
        demographic_bias = self._analyze_demographic_bias(dataset_info)
        visual_bias = self._analyze_visual_bias(dataset_info)
        quality_bias = self._analyze_quality_bias(dataset_info)
        temporal_bias = self._analyze_temporal_bias(dataset_info)
        class_bias = self._analyze_class_bias(dataset_info)
        
        # Testes estatísticos
        statistical_tests = self._perform_statistical_tests(dataset_info)
        
        # Gerar recomendações
        recommendations = self._generate_recommendations(
            demographic_bias, visual_bias, quality_bias, temporal_bias, class_bias
        )
        
        # Calcular score geral de bias
        overall_bias_score = self._calculate_overall_bias_score(
            demographic_bias, visual_bias, quality_bias, temporal_bias, class_bias
        )
        
        # Determinar severidade
        bias_severity = self._determine_bias_severity(overall_bias_score)
        
        # Criar resultado
        result = BiasAnalysisResult(
            overall_bias_score=overall_bias_score,
            bias_severity=bias_severity,
            biome_bias=demographic_bias.get('biome_bias'),
            seasonal_bias=demographic_bias.get('seasonal_bias'),
            quality_bias=quality_bias,
            class_bias=class_bias,
            color_bias=visual_bias.get('color_bias'),
            composition_bias=visual_bias.get('composition_bias'),
            temporal_bias=temporal_bias,
            recommendations=recommendations['general'],
            priority_actions=recommendations['priority'],
            statistical_tests=statistical_tests,
            underrepresented_groups=self._find_underrepresented_groups(dataset_info),
            overrepresented_groups=self._find_overrepresented_groups(dataset_info)
        )
        
        logger.info(f"✅ Análise de bias concluída - Severidade: {bias_severity}")
        
        return result
    
    def _load_dataset_info(
        self,
        dataset_path: str,
        metadata_path: Optional[str],
        sample_size: Optional[int]
    ) -> Dict[str, Any]:
        """Carrega informações do dataset e metadados"""
        
        dataset_dir = Path(dataset_path)
        
        # Encontrar imagens
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_dir.rglob(f"*{ext}"))
        
        if sample_size and len(image_files) > sample_size:
            # Amostragem aleatória estratificada se possível
            image_files = np.random.choice(image_files, sample_size, replace=False)
        
        # Carregar metadados se disponível
        metadata = {}
        if metadata_path and Path(metadata_path).exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Erro ao carregar metadados: {e}")
        
        # Organizar informações
        dataset_info = {
            'images': image_files,
            'metadata': metadata,
            'total_images': len(image_files),
            'dataset_path': dataset_dir
        }
        
        logger.info(f"📊 Dataset carregado: {len(image_files)} imagens")
        
        return dataset_info
    
    def _analyze_demographic_bias(self, dataset_info: Dict) -> Dict[str, Any]:
        """Análise de bias demográfico (biomas, estações, qualidade)"""
        
        logger.info("📈 Analisando bias demográfico...")
        
        # Extrair distribuições dos metadados
        biome_distribution = self._extract_distribution(dataset_info, 'biome')
        season_distribution = self._extract_distribution(dataset_info, 'season')
        quality_distribution = self._extract_distribution(dataset_info, 'quality')
        
        # Calcular bias para cada dimensão
        biome_bias = self._calculate_distribution_bias(
            biome_distribution, self.expected_distributions['biome']
        )
        
        seasonal_bias = self._calculate_distribution_bias(
            season_distribution, self.expected_distributions['season']
        )
        
        quality_bias = self._calculate_distribution_bias(
            quality_distribution, self.expected_distributions['quality']
        )
        
        return {
            'biome_bias': biome_bias,
            'seasonal_bias': seasonal_bias,
            'quality_bias_demographic': quality_bias,
            'biome_distribution': biome_distribution,
            'season_distribution': season_distribution,
            'quality_distribution': quality_distribution
        }
    
    def _analyze_visual_bias(self, dataset_info: Dict) -> Dict[str, Any]:
        """Análise de bias visual (cores, composição, estilo)"""
        
        if not self.config.enable_visual_bias_detection:
            return {}
        
        logger.info("🎨 Analisando bias visual...")
        
        # Amostrar imagens para análise visual (computacionalmente intensiva)
        sample_images = dataset_info['images'][:min(500, len(dataset_info['images']))]
        
        # Extrair features visuais
        visual_features = self._extract_visual_features(sample_images)
        
        # Análise de bias de cor
        color_bias = self._analyze_color_bias(visual_features['colors'])
        
        # Análise de bias de composição
        composition_bias = self._analyze_composition_bias(visual_features['compositions'])
        
        # Análise de bias de estilo
        style_bias = self._analyze_style_bias(visual_features['styles'])
        
        return {
            'color_bias': color_bias,
            'composition_bias': composition_bias,
            'style_bias': style_bias,
            'visual_diversity_score': self._calculate_visual_diversity(visual_features)
        }
    
    def _analyze_quality_bias(self, dataset_info: Dict) -> Dict[str, Any]:
        """Análise de bias de qualidade técnica"""
        
        if not self.config.enable_quality_bias_analysis:
            return {}
        
        logger.info("⭐ Analisando bias de qualidade...")
        
        # Amostra para análise de qualidade
        sample_images = dataset_info['images'][:min(200, len(dataset_info['images']))]
        
        quality_scores = []
        quality_by_group = defaultdict(list)
        
        for img_path in sample_images:
            try:
                # Análise básica de qualidade técnica
                quality_score = self._assess_image_quality(img_path)
                quality_scores.append(quality_score)
                
                # Agrupar por metadados se disponível
                img_metadata = self._get_image_metadata(img_path, dataset_info['metadata'])
                if img_metadata:
                    for key, value in img_metadata.items():
                        if key in ['biome', 'season', 'quality']:
                            quality_by_group[f"{key}_{value}"].append(quality_score)
                            
            except Exception as e:
                logger.warning(f"Erro ao analisar qualidade de {img_path}: {e}")
                continue
        
        # Calcular bias de qualidade entre grupos
        quality_bias = {}
        for group, scores in quality_by_group.items():
            if len(scores) >= self.config.min_samples_per_group:
                group_mean = np.mean(scores)
                overall_mean = np.mean(quality_scores)
                bias_score = abs(group_mean - overall_mean) / overall_mean
                quality_bias[group] = bias_score
        
        return {
            'overall_quality_mean': np.mean(quality_scores),
            'quality_std': np.std(quality_scores),
            'quality_bias_by_group': quality_bias,
            'quality_range': (np.min(quality_scores), np.max(quality_scores))
        }
    
    def _analyze_temporal_bias(self, dataset_info: Dict) -> Dict[str, Any]:
        """Análise de bias temporal"""
        
        if not self.config.enable_temporal_bias_analysis:
            return {}
        
        logger.info("📅 Analisando bias temporal...")
        
        # Extrair timestamps de geração se disponível
        temporal_data = self._extract_temporal_data(dataset_info)
        
        if not temporal_data:
            return {'temporal_bias': 'insufficient_data'}
        
        # Análise de distribuição temporal
        temporal_bias = self._analyze_temporal_distribution(temporal_data)
        
        return temporal_bias
    
    def _analyze_class_bias(self, dataset_info: Dict) -> Dict[str, Any]:
        """Análise de bias entre classes"""
        
        logger.info("🏷️ Analisando bias entre classes...")
        
        # Extrair distribuição de classes das anotações se disponível
        class_distribution = self._extract_class_distribution(dataset_info)
        
        if not class_distribution:
            return {'class_bias': 'no_annotations_found'}
        
        # Calcular bias entre classes críticas
        class_bias = {}
        
        for group_name, classes in self.critical_groups.items():
            group_counts = {cls: class_distribution.get(cls, 0) for cls in classes}
            total_group = sum(group_counts.values())
            
            if total_group > 0:
                # Calcular equilíbrio dentro do grupo
                expected_per_class = total_group / len(classes)
                bias_scores = []
                
                for cls, count in group_counts.items():
                    if expected_per_class > 0:
                        bias = abs(count - expected_per_class) / expected_per_class
                        bias_scores.append(bias)
                
                class_bias[group_name] = np.mean(bias_scores) if bias_scores else 0
        
        return class_bias
    
    def _extract_distribution(self, dataset_info: Dict, attribute: str) -> Dict[str, float]:
        """Extrai distribuição de um atributo dos metadados"""
        
        distribution = Counter()
        total = 0
        
        # Tentar extrair dos metadados principais
        if 'images' in dataset_info.get('metadata', {}):
            for img_data in dataset_info['metadata']['images']:
                if 'pasture_config' in img_data and attribute in img_data['pasture_config']:
                    value = img_data['pasture_config'][attribute]
                    distribution[value] += 1
                    total += 1
        
        # Se não encontrou nos metadados principais, tentar extrair dos nomes dos arquivos
        if total == 0:
            for img_path in dataset_info['images']:
                # Tentar inferir do nome do arquivo ou caminho
                path_parts = str(img_path).lower()
                
                if attribute == 'biome':
                    if 'cerrado' in path_parts:
                        distribution['cerrado'] += 1
                    elif 'mata' in path_parts or 'atlantica' in path_parts:
                        distribution['mata_atlantica'] += 1
                    elif 'pampa' in path_parts:
                        distribution['pampa'] += 1
                    else:
                        distribution['unknown'] += 1
                    total += 1
        
        # Normalizar para percentuais
        if total > 0:
            return {key: count/total for key, count in distribution.items()}
        else:
            return {}
    
    def _calculate_distribution_bias(
        self,
        observed: Dict[str, float],
        expected: Dict[str, float]
    ) -> Dict[str, float]:
        """Calcula bias entre distribuições observadas e esperadas"""
        
        bias_scores = {}
        
        for category in expected:
            observed_ratio = observed.get(category, 0)
            expected_ratio = expected[category]
            
            if expected_ratio > 0:
                bias = abs(observed_ratio - expected_ratio) / expected_ratio
                bias_scores[category] = bias
            else:
                bias_scores[category] = 0
        
        # Adicionar categorias observadas que não eram esperadas
        for category in observed:
            if category not in expected:
                bias_scores[category] = 1.0  # Bias máximo para categorias não esperadas
        
        return bias_scores
    
    def _extract_visual_features(self, image_paths: List) -> Dict[str, Any]:
        """Extrai features visuais das imagens"""
        
        features = {
            'colors': [],
            'compositions': [],
            'styles': []
        }
        
        for img_path in image_paths:
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Features de cor
                color_features = self._extract_color_features(image_rgb)
                features['colors'].append(color_features)
                
                # Features de composição
                composition_features = self._extract_composition_features(image_rgb)
                features['compositions'].append(composition_features)
                
                # Features de estilo (simplificado)
                style_features = self._extract_style_features(image_rgb)
                features['styles'].append(style_features)
                
            except Exception as e:
                logger.warning(f"Erro ao extrair features de {img_path}: {e}")
                continue
        
        return features
    
    def _extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extrai features de cor da imagem"""
        
        # Converter para HSV para análise de cor
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Histogramas de cor
        h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])  # 18 bins para Hue
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])   # 8 bins para Saturation
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])   # 8 bins para Value
        
        # Normalizar
        h_hist = h_hist.flatten() / h_hist.sum()
        s_hist = s_hist.flatten() / s_hist.sum()
        v_hist = v_hist.flatten() / v_hist.sum()
        
        # Estatísticas de cor
        mean_hue = np.mean(hsv[:, :, 0])
        mean_saturation = np.mean(hsv[:, :, 1])
        mean_value = np.mean(hsv[:, :, 2])
        
        return {
            'hue_histogram': h_hist.tolist(),
            'saturation_histogram': s_hist.tolist(),
            'value_histogram': v_hist.tolist(),
            'mean_hue': float(mean_hue),
            'mean_saturation': float(mean_saturation),
            'mean_value': float(mean_value),
            'color_diversity': float(len(np.unique(image.reshape(-1, 3), axis=0)))
        }
    
    def _extract_composition_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extrai features de composição da imagem"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detectar bordas para análise de composição
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Análise de textura usando LBP simplificado
        texture_score = np.std(gray)
        
        # Análise de simetria (simplificada)
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = np.flip(gray[:, w//2:], axis=1)
        
        # Redimensionar se necessário para comparação
        if left_half.shape != right_half.shape:
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
        
        symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        if np.isnan(symmetry_score):
            symmetry_score = 0.0
        
        return {
            'edge_density': float(edge_density),
            'texture_score': float(texture_score),
            'symmetry_score': float(symmetry_score),
            'brightness_variance': float(np.var(gray))
        }
    
    def _extract_style_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extrai features de estilo da imagem (simplificado)"""
        
        # Features básicas que podem indicar estilo
        
        # Análise de contraste
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        contrast = gray.std()
        
        # Análise de saturação média
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        
        # Análise de nitidez
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            'contrast': float(contrast),
            'mean_saturation': float(mean_saturation),
            'sharpness': float(laplacian_var),
            'brightness': float(np.mean(gray))
        }
    
    def _analyze_color_bias(self, color_features: List[Dict]) -> Dict[str, Any]:
        """Analisa bias de cor"""
        
        if not color_features:
            return {}
        
        # Agregar histogramas de cor
        all_hue_hists = np.array([cf['hue_histogram'] for cf in color_features])
        all_sat_hists = np.array([cf['saturation_histogram'] for cf in color_features])
        
        # Calcular diversidade de cor
        mean_hue_hist = np.mean(all_hue_hists, axis=0)
        hue_entropy = -np.sum(mean_hue_hist * np.log(mean_hue_hist + 1e-8))
        max_entropy = np.log(len(mean_hue_hist))
        hue_diversity = hue_entropy / max_entropy
        
        # Detectar cores dominantes
        dominant_hue_bin = np.argmax(mean_hue_hist)
        dominance_ratio = mean_hue_hist[dominant_hue_bin]
        
        # Bias de saturação
        saturations = [cf['mean_saturation'] for cf in color_features]
        saturation_std = np.std(saturations)
        
        return {
            'hue_diversity': float(hue_diversity),
            'dominant_hue_bin': int(dominant_hue_bin),
            'dominance_ratio': float(dominance_ratio),
            'saturation_std': float(saturation_std),
            'color_bias_score': float(1.0 - hue_diversity + dominance_ratio) / 2
        }
    
    def _analyze_composition_bias(self, composition_features: List[Dict]) -> Dict[str, Any]:
        """Analisa bias de composição"""
        
        if not composition_features:
            return {}
        
        # Analisar variabilidade nas features de composição
        edge_densities = [cf['edge_density'] for cf in composition_features]
        texture_scores = [cf['texture_score'] for cf in composition_features]
        symmetry_scores = [cf['symmetry_score'] for cf in composition_features]
        
        return {
            'edge_density_std': float(np.std(edge_densities)),
            'texture_score_std': float(np.std(texture_scores)),
            'symmetry_score_std': float(np.std(symmetry_scores)),
            'composition_diversity': float(np.mean([
                np.std(edge_densities),
                np.std(texture_scores),
                np.std(symmetry_scores)
            ])),
            'composition_bias_score': float(1.0 - min(1.0, np.std(edge_densities) * 5))
        }
    
    def _analyze_style_bias(self, style_features: List[Dict]) -> Dict[str, Any]:
        """Analisa bias de estilo"""
        
        if not style_features:
            return {}
        
        # Clustering para identificar estilos dominantes
        feature_matrix = np.array([
            [sf['contrast'], sf['mean_saturation'], sf['sharpness'], sf['brightness']]
            for sf in style_features
        ])
        
        # Normalizar features
        feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-8)
        
        # K-means clustering
        n_clusters = min(self.config.n_visual_clusters, len(style_features))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            
            # Analisar distribuição de clusters
            cluster_counts = Counter(cluster_labels)
            cluster_ratios = np.array(list(cluster_counts.values())) / len(style_features)
            
            # Calcular bias de estilo (quão desigual é a distribuição)
            expected_ratio = 1.0 / n_clusters
            style_bias = np.mean(np.abs(cluster_ratios - expected_ratio)) / expected_ratio
        else:
            style_bias = 0.0
            cluster_ratios = [1.0]
        
        return {
            'style_bias_score': float(style_bias),
            'n_style_clusters': n_clusters,
            'cluster_distribution': cluster_ratios.tolist() if n_clusters > 1 else [1.0],
            'style_diversity': float(1.0 - style_bias)
        }
    
    def _calculate_visual_diversity(self, visual_features: Dict) -> float:
        """Calcula score geral de diversidade visual"""
        
        if not visual_features['colors']:
            return 0.0
        
        # Combinar diferentes aspectos de diversidade
        color_diversity = np.mean([cf.get('color_diversity', 0) for cf in visual_features['colors']])
        
        # Normalizar e combinar
        max_color_diversity = 256**3  # RGB máximo
        normalized_color_diversity = min(1.0, color_diversity / 10000)  # Normalizar
        
        return normalized_color_diversity
    
    def _assess_image_quality(self, img_path: str) -> float:
        """Avaliação rápida de qualidade da imagem"""
        
        try:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 0.0
            
            # Análise de nitidez
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)
            
            # Análise de contraste
            contrast_score = min(1.0, image.std() / 64.0)
            
            # Score combinado
            quality_score = (sharpness_score + contrast_score) / 2
            
            return quality_score
            
        except Exception:
            return 0.0
    
    def _get_image_metadata(self, img_path: str, metadata: Dict) -> Optional[Dict]:
        """Obtém metadados de uma imagem específica"""
        
        img_name = Path(img_path).stem
        
        if 'images' in metadata:
            for img_data in metadata['images']:
                if img_data.get('filename', '').startswith(img_name):
                    return img_data.get('pasture_config', {})
        
        return None
    
    def _extract_temporal_data(self, dataset_info: Dict) -> List[datetime]:
        """Extrai dados temporais do dataset"""
        
        timestamps = []
        
        # Tentar extrair timestamps dos metadados
        if 'images' in dataset_info.get('metadata', {}):
            for img_data in dataset_info['metadata']['images']:
                if 'timestamp' in img_data:
                    try:
                        timestamp = datetime.fromisoformat(img_data['timestamp'])
                        timestamps.append(timestamp)
                    except ValueError:
                        continue
        
        return timestamps
    
    def _analyze_temporal_distribution(self, temporal_data: List[datetime]) -> Dict[str, Any]:
        """Analisa distribuição temporal"""
        
        if len(temporal_data) < 2:
            return {}
        
        # Analisar distribuição por hora do dia
        hours = [dt.hour for dt in temporal_data]
        hour_distribution = Counter(hours)
        
        # Analisar distribuição por dia da semana
        weekdays = [dt.weekday() for dt in temporal_data]
        weekday_distribution = Counter(weekdays)
        
        # Calcular bias temporal
        expected_hour_ratio = 1.0 / 24
        hour_ratios = np.array([hour_distribution.get(h, 0) for h in range(24)]) / len(temporal_data)
        hour_bias = np.mean(np.abs(hour_ratios - expected_hour_ratio)) / expected_hour_ratio
        
        return {
            'hour_bias': float(hour_bias),
            'hour_distribution': dict(hour_distribution),
            'weekday_distribution': dict(weekday_distribution),
            'temporal_span_days': (max(temporal_data) - min(temporal_data)).days
        }
    
    def _extract_class_distribution(self, dataset_info: Dict) -> Dict[str, int]:
        """Extrai distribuição de classes das anotações"""
        
        # Procurar por arquivos de anotação YOLO
        dataset_dir = dataset_info['dataset_path']
        
        class_counts = Counter()
        
        # Procurar em diretórios de anotação
        for ann_dir in ['labels', 'annotations']:
            ann_path = dataset_dir / ann_dir
            if ann_path.exists():
                for ann_file in ann_path.glob('*.txt'):
                    try:
                        with open(ann_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    class_id = int(line.strip().split()[0])
                                    class_counts[class_id] += 1
                    except (ValueError, IndexError, FileNotFoundError):
                        continue
        
        return dict(class_counts)
    
    def _perform_statistical_tests(self, dataset_info: Dict) -> Dict[str, Dict]:
        """Executa testes estatísticos para detectar bias"""
        
        statistical_tests = {}
        
        # Teste qui-quadrado para distribuições categóricas
        for attribute in ['biome', 'season', 'quality']:
            observed_dist = self._extract_distribution(dataset_info, attribute)
            expected_dist = self.expected_distributions.get(attribute, {})
            
            if observed_dist and expected_dist:
                # Preparar dados para teste qui-quadrado
                common_categories = set(observed_dist.keys()) & set(expected_dist.keys())
                
                if len(common_categories) >= 2:
                    observed_counts = [observed_dist[cat] * dataset_info['total_images'] 
                                     for cat in common_categories]
                    expected_counts = [expected_dist[cat] * dataset_info['total_images'] 
                                     for cat in common_categories]
                    
                    try:
                        chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)
                        statistical_tests[f'{attribute}_chi2'] = {
                            'statistic': float(chi2_stat),
                            'p_value': float(p_value),
                            'significant': p_value < self.config.statistical_significance
                        }
                    except ValueError:
                        pass
        
        return statistical_tests
    
    def _find_underrepresented_groups(self, dataset_info: Dict) -> List[str]:
        """Identifica grupos sub-representados"""
        
        underrepresented = []
        
        for attribute, expected_dist in self.expected_distributions.items():
            observed_dist = self._extract_distribution(dataset_info, attribute)
            
            for category, expected_ratio in expected_dist.items():
                observed_ratio = observed_dist.get(category, 0)
                
                if observed_ratio < expected_ratio * 0.7:  # 30% abaixo do esperado
                    underrepresented.append(f"{attribute}_{category}")
        
        return underrepresented
    
    def _find_overrepresented_groups(self, dataset_info: Dict) -> List[str]:
        """Identifica grupos sobre-representados"""
        
        overrepresented = []
        
        for attribute, expected_dist in self.expected_distributions.items():
            observed_dist = self._extract_distribution(dataset_info, attribute)
            
            for category, expected_ratio in expected_dist.items():
                observed_ratio = observed_dist.get(category, 0)
                
                if observed_ratio > expected_ratio * 1.5:  # 50% acima do esperado
                    overrepresented.append(f"{attribute}_{category}")
        
        return overrepresented
    
    def _generate_recommendations(self, *bias_analyses) -> Dict[str, List[str]]:
        """Gera recomendações baseadas nas análises de bias"""
        
        general_recommendations = []
        priority_actions = []
        
        # Analisar resultados e gerar recomendações específicas
        demographic_bias = bias_analyses[0] if len(bias_analyses) > 0 else {}
        visual_bias = bias_analyses[1] if len(bias_analyses) > 1 else {}
        
        # Recomendações demográficas
        if demographic_bias:
            for bias_type, bias_data in demographic_bias.items():
                if isinstance(bias_data, dict):
                    for category, bias_score in bias_data.items():
                        if bias_score > self.config.bias_threshold_high:
                            priority_actions.append(
                                f"Urgente: Corrigir bias extremo em {bias_type}_{category} (score: {bias_score:.3f})"
                            )
                        elif bias_score > self.config.bias_threshold_medium:
                            general_recommendations.append(
                                f"Balancear representação de {bias_type}_{category}"
                            )
        
        # Recomendações visuais
        if visual_bias:
            color_bias_score = visual_bias.get('color_bias', {}).get('color_bias_score', 0)
            if color_bias_score > self.config.bias_threshold_medium:
                general_recommendations.append(
                    "Aumentar diversidade de cores nas imagens geradas"
                )
        
        # Recomendações gerais sempre aplicáveis
        general_recommendations.extend([
            "Implementar amostragem estratificada para balancear grupos",
            "Monitorar bias continuamente durante a geração",
            "Validar com especialistas em diversidade de pastagens"
        ])
        
        return {
            'general': general_recommendations,
            'priority': priority_actions
        }
    
    def _calculate_overall_bias_score(self, *bias_analyses) -> float:
        """Calcula score geral de bias"""
        
        bias_scores = []
        
        # Coletar todos os scores de bias
        for analysis in bias_analyses:
            if isinstance(analysis, dict):
                for key, value in analysis.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)) and 'bias' in sub_key.lower():
                                bias_scores.append(float(sub_value))
                    elif isinstance(value, (int, float)) and 'bias' in key.lower():
                        bias_scores.append(float(value))
        
        if not bias_scores:
            return 0.0
        
        # Score geral como média ponderada
        return float(np.mean(bias_scores))
    
    def _determine_bias_severity(self, overall_bias_score: float) -> str:
        """Determina severidade do bias baseado no score"""
        
        if overall_bias_score >= self.config.bias_threshold_high:
            return "critical"
        elif overall_bias_score >= self.config.bias_threshold_medium:
            return "high"
        elif overall_bias_score >= self.config.bias_threshold_low:
            return "medium"
        else:
            return "low"
    
    def generate_bias_report(
        self,
        result: BiasAnalysisResult,
        output_path: str
    ):
        """Gera relatório detalhado de bias"""
        
        report_data = {
            'analysis_date': datetime.now().isoformat(),
            'bias_analysis_result': asdict(result),
            'configuration': asdict(self.config),
            'summary': {
                'overall_bias_score': result.overall_bias_score,
                'bias_severity': result.bias_severity,
                'critical_issues': len(result.priority_actions) if result.priority_actions else 0,
                'total_recommendations': len(result.recommendations) if result.recommendations else 0
            }
        }
        
        # Salvar relatório JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Relatório de bias salvo: {output_path}")

def quick_bias_check(dataset_path: str) -> Dict[str, Any]:
    """Função helper para verificação rápida de bias"""
    
    detector = BiasDetector()
    result = detector.analyze_dataset_bias(dataset_path, sample_size=100)
    
    return {
        'overall_bias_score': result.overall_bias_score,
        'bias_severity': result.bias_severity,
        'priority_actions_count': len(result.priority_actions) if result.priority_actions else 0,
        'underrepresented_groups': result.underrepresented_groups
    }
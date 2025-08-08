"""
Sistema de Data Augmentation especializado para pastagens brasileiras
Aplica transformações contextualmente apropriadas baseadas em condições reais
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Optional, Tuple, Union
import logging
import random
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.color import rgb2hsv, hsv2rgb
import torch

logger = logging.getLogger(__name__)

@dataclass
class AugmentationConfig:
    """Configuração para augmentations"""
    seasonal_shift_prob: float = 0.7
    weather_effects_prob: float = 0.5
    lighting_variation_prob: float = 0.6
    spatial_transform_prob: float = 0.8
    quality_degradation_prob: float = 0.3
    preserve_realism: bool = True

class DataAugmentation:
    """
    Sistema de augmentation especializado em pastagens brasileiras
    Aplica transformações realistas baseadas em variações naturais
    """
    
    def __init__(self, preserve_agricultural_realism: bool = True):
        self.preserve_realism = preserve_agricultural_realism
        
        # Configurações sazonais específicas
        self.seasonal_params = {
            'dry_season': {
                'hue_shift_range': (5, 25),      # Mais amarelado
                'saturation_range': (0.6, 0.9),  # Menos saturado
                'brightness_range': (1.1, 1.3),  # Mais claro
                'contrast_range': (1.1, 1.4)     # Maior contraste
            },
            'wet_season': {
                'hue_shift_range': (-10, 5),     # Mais verdejante
                'saturation_range': (1.1, 1.4),  # Mais saturado
                'brightness_range': (0.8, 1.0),  # Mais escuro
                'contrast_range': (0.9, 1.1)     # Contraste suave
            },
            'transition': {
                'hue_shift_range': (-5, 15),     # Variação moderada
                'saturation_range': (0.9, 1.2),  # Moderadamente saturado
                'brightness_range': (0.9, 1.1),  # Brilho natural
                'contrast_range': (1.0, 1.2)     # Contraste moderado
            }
        }
        
        # Efeitos climáticos por bioma
        self.biome_weather_effects = {
            'cerrado': {
                'dust_prob': 0.3,
                'haze_prob': 0.2,
                'intense_sun_prob': 0.4
            },
            'mata_atlantica': {
                'mist_prob': 0.4,
                'humidity_blur_prob': 0.3,
                'filtered_light_prob': 0.5
            },
            'pampa': {
                'wind_blur_prob': 0.3,
                'clear_sky_prob': 0.6,
                'grass_movement_prob': 0.2
            }
        }
        
        # Pipeline Albumentations para transformações espaciais
        self.spatial_transforms = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),  # Menos comum para vistas aéreas
            ], p=0.5),
            
            A.OneOf([
                A.Rotate(limit=(-5, 5), p=0.7),  # Pequenas rotações
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=(-0.1, 0.1), 
                    rotate_limit=3,
                    p=0.6
                ),
            ], p=0.4),
            
            A.OneOf([
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.3
                ),
                A.GridDistortion(p=0.2),
                A.OpticalDistortion(p=0.2),
            ], p=0.2),
        ], p=1.0)
        
        # Pipeline para efeitos atmosféricos
        self.atmospheric_transforms = A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
            ], p=0.3),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            ], p=0.2),
            
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
        ], p=1.0)
        
    def apply_seasonal_augmentation(
        self, 
        image: Union[Image.Image, np.ndarray],
        current_season: str,
        target_season: str,
        intensity: float = 1.0
    ) -> Image.Image:
        """
        Aplica transformação sazonal para simular mudança de estação
        
        Args:
            image: Imagem de entrada
            current_season: Estação atual da imagem
            target_season: Estação alvo desejada
            intensity: Intensidade da transformação (0-1)
            
        Returns:
            Imagem transformada
        """
        
        # Converter para PIL se necessário
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Se estações são iguais, retornar original
        if current_season == target_season:
            return image
            
        logger.debug(f"Aplicando transformação: {current_season} → {target_season}")
        
        # Obter parâmetros da estação alvo
        target_params = self.seasonal_params.get(target_season, self.seasonal_params['wet_season'])
        
        # Converter para array para manipulação
        img_array = np.array(image) / 255.0
        hsv = rgb2hsv(img_array)
        
        # Ajustar matiz (hue) para estação alvo
        hue_shift = random.uniform(*target_params['hue_shift_range']) * intensity / 360
        hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 1.0
        
        # Ajustar saturação
        sat_mult = random.uniform(*target_params['saturation_range']) * intensity
        hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_mult, 0, 1)
        
        # Converter de volta para RGB
        rgb = hsv2rgb(hsv)
        result_image = Image.fromarray((rgb * 255).astype(np.uint8))
        
        # Ajustar brilho e contraste
        brightness_mult = random.uniform(*target_params['brightness_range'])
        contrast_mult = random.uniform(*target_params['contrast_range'])
        
        # Aplicar ajustes de brilho/contraste
        enhancer = ImageEnhance.Brightness(result_image)
        result_image = enhancer.enhance(brightness_mult * intensity)
        
        enhancer = ImageEnhance.Contrast(result_image)
        result_image = enhancer.enhance(contrast_mult * intensity)
        
        return result_image
        
    def apply_weather_effects(
        self,
        image: Union[Image.Image, np.ndarray],
        biome: str,
        weather_type: Optional[str] = None,
        intensity: float = 0.5
    ) -> Image.Image:
        """
        Aplica efeitos climáticos específicos do bioma
        
        Args:
            image: Imagem de entrada
            biome: Bioma ('cerrado', 'mata_atlantica', 'pampa')
            weather_type: Tipo específico de clima (opcional)
            intensity: Intensidade do efeito
            
        Returns:
            Imagem com efeitos climáticos
        """
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        result = image.copy()
        biome_effects = self.biome_weather_effects.get(biome, {})
        
        # Aplicar efeito específico ou escolher aleatoriamente
        if weather_type:
            effect_func = getattr(self, f'_apply_{weather_type}', None)
            if effect_func:
                result = effect_func(result, intensity)
        else:
            # Aplicar efeitos baseados na probabilidade do bioma
            for effect, prob in biome_effects.items():
                if random.random() < prob:
                    effect_name = effect.replace('_prob', '')
                    effect_func = getattr(self, f'_apply_{effect_name}', None)
                    if effect_func:
                        result = effect_func(result, intensity)
                        
        return result
        
    def _apply_dust(self, image: Image.Image, intensity: float) -> Image.Image:
        """Aplica efeito de poeira (comum no cerrado seco)"""
        
        img_array = np.array(image)
        
        # Criar máscara de poeira
        dust_mask = np.random.random(img_array.shape[:2]) < intensity * 0.3
        
        # Aplicar tom amarelado e reduzir contraste
        hsv = rgb2hsv(img_array / 255.0)
        hsv[dust_mask, 0] = (hsv[dust_mask, 0] + 0.08) % 1.0  # Mais amarelado
        hsv[dust_mask, 1] *= 0.7  # Menos saturado
        
        rgb = hsv2rgb(hsv)
        result = Image.fromarray((rgb * 255).astype(np.uint8))
        
        # Adicionar leve blur para simular partículas
        if intensity > 0.5:
            result = result.filter(ImageFilter.GaussianBlur(radius=0.5))
            
        return result
        
    def _apply_mist(self, image: Image.Image, intensity: float) -> Image.Image:
        """Aplica efeito de neblina (comum na mata atlântica)"""
        
        # Criar overlay de neblina
        overlay = Image.new('RGB', image.size, (240, 240, 250))
        
        # Blend com a imagem original
        blend_factor = intensity * 0.2
        result = Image.blend(image, overlay, blend_factor)
        
        # Reduzir contraste para simular difusão
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1 - intensity * 0.3)
        
        return result
        
    def _apply_haze(self, image: Image.Image, intensity: float) -> Image.Image:
        """Aplica efeito de calor/névoa seca"""
        
        img_array = np.array(image)
        
        # Aplicar distorção sutil para simular ondas de calor
        rows, cols, _ = img_array.shape
        
        # Criar campo de deslocamento sutil
        displacement = np.sin(np.arange(cols) * 0.1) * intensity * 2
        
        result_array = img_array.copy()
        for i in range(rows):
            shift = int(displacement[i % len(displacement)])
            if shift > 0:
                result_array[i, shift:] = img_array[i, :-shift]
            elif shift < 0:
                result_array[i, :shift] = img_array[i, -shift:]
                
        # Adicionar leve blur vertical
        result = Image.fromarray(result_array)
        result = result.filter(ImageFilter.GaussianBlur(radius=0.3))
        
        return result
        
    def _apply_wind_blur(self, image: Image.Image, intensity: float) -> Image.Image:
        """Aplica blur direcional para simular vento (pampa)"""
        
        # Motion blur horizontal para simular vento
        kernel_size = int(intensity * 5) * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0
        kernel = kernel / kernel_size
        
        img_array = np.array(image)
        blurred = cv2.filter2D(img_array, -1, kernel)
        
        # Blend com original para efeito sutil
        blend_factor = intensity * 0.5
        result_array = img_array * (1 - blend_factor) + blurred * blend_factor
        
        return Image.fromarray(result_array.astype(np.uint8))
        
    def apply_quality_degradation(
        self,
        image: Union[Image.Image, np.ndarray],
        degradation_type: str = "compression",
        intensity: float = 0.3
    ) -> Image.Image:
        """
        Aplica degradação de qualidade para simular diferentes condições
        
        Args:
            image: Imagem de entrada
            degradation_type: Tipo de degradação ('compression', 'noise', 'blur')
            intensity: Intensidade da degradação
            
        Returns:
            Imagem degradada
        """
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        if degradation_type == "compression":
            return self._apply_compression_artifacts(image, intensity)
        elif degradation_type == "noise":
            return self._apply_sensor_noise(image, intensity)
        elif degradation_type == "blur":
            return self._apply_motion_blur(image, intensity)
        else:
            return image
            
    def _apply_compression_artifacts(self, image: Image.Image, intensity: float) -> Image.Image:
        """Simula artefatos de compressão JPEG"""
        
        import io
        
        # Qualidade baseada na intensidade
        quality = max(30, int(100 - intensity * 70))
        
        # Salvar e recarregar com compressão
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        return Image.open(buffer)
        
    def _apply_sensor_noise(self, image: Image.Image, intensity: float) -> Image.Image:
        """Aplica ruído de sensor"""
        
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Ruído gaussiano
        noise = np.random.normal(0, intensity * 0.05, img_array.shape)
        noisy = img_array + noise
        
        # Ruído de salt & pepper
        if intensity > 0.5:
            salt_pepper = np.random.random(img_array.shape[:2])
            noisy[salt_pepper < intensity * 0.01] = 0  # Pepper
            noisy[salt_pepper > 1 - intensity * 0.01] = 1  # Salt
            
        noisy = np.clip(noisy, 0, 1)
        
        return Image.fromarray((noisy * 255).astype(np.uint8))
        
    def _apply_motion_blur(self, image: Image.Image, intensity: float) -> Image.Image:
        """Aplica blur de movimento"""
        
        # Kernel de movimento diagonal
        kernel_size = int(intensity * 10) + 1
        kernel = np.eye(kernel_size) / kernel_size
        
        img_array = np.array(image)
        
        # Aplicar blur em cada canal
        blurred = np.zeros_like(img_array)
        for i in range(img_array.shape[2]):
            blurred[:,:,i] = cv2.filter2D(img_array[:,:,i], -1, kernel)
            
        return Image.fromarray(blurred)
        
    def create_augmentation_pipeline(
        self,
        config: AugmentationConfig,
        target_season: str = None,
        target_biome: str = None
    ) -> callable:
        """
        Cria pipeline de augmentation customizado
        
        Args:
            config: Configuração de augmentation
            target_season: Estação alvo para transformações sazonais
            target_biome: Bioma alvo para efeitos específicos
            
        Returns:
            Função de augmentation
        """
        
        def augmentation_pipeline(image, **kwargs):
            """Pipeline de augmentation aplicado"""
            
            result = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image)
            
            # Transformações espaciais
            if random.random() < config.spatial_transform_prob:
                img_array = np.array(result)
                transformed = self.spatial_transforms(image=img_array)
                result = Image.fromarray(transformed['image'])
                
            # Transformação sazonal
            if target_season and random.random() < config.seasonal_shift_prob:
                current_season = kwargs.get('current_season', 'wet_season')
                result = self.apply_seasonal_augmentation(
                    result, current_season, target_season
                )
                
            # Efeitos climáticos
            if target_biome and random.random() < config.weather_effects_prob:
                result = self.apply_weather_effects(result, target_biome)
                
            # Degradação de qualidade
            if random.random() < config.quality_degradation_prob:
                degradation_types = ['compression', 'noise', 'blur']
                degradation_type = random.choice(degradation_types)
                result = self.apply_quality_degradation(
                    result, degradation_type, intensity=0.3
                )
                
            return result
            
        return augmentation_pipeline
        
    def augment_dataset_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        config: AugmentationConfig,
        metadata_list: Optional[List[Dict]] = None
    ) -> List[Image.Image]:
        """
        Aplica augmentation em batch de imagens
        
        Args:
            images: Lista de imagens
            config: Configuração de augmentation
            metadata_list: Lista de metadados correspondentes
            
        Returns:
            Lista de imagens aumentadas
        """
        
        augmented = []
        
        for i, image in enumerate(images):
            metadata = metadata_list[i] if metadata_list else {}
            
            # Extrair informações do metadata
            current_season = metadata.get('season', 'wet_season')
            biome = metadata.get('biome', 'cerrado')
            
            # Criar pipeline específico
            pipeline = self.create_augmentation_pipeline(
                config, 
                target_season=current_season,
                target_biome=biome
            )
            
            # Aplicar augmentation
            aug_image = pipeline(image, current_season=current_season)
            augmented.append(aug_image)
            
        logger.info(f"✅ Augmentation aplicado em {len(images)} imagens")
        
        return augmented
        
    def get_augmentation_statistics(self) -> Dict:
        """Retorna estatísticas sobre augmentations disponíveis"""
        
        return {
            'seasonal_variations': len(self.seasonal_params),
            'biome_weather_effects': len(self.biome_weather_effects),
            'spatial_transforms': len(self.spatial_transforms.transforms),
            'atmospheric_effects': len(self.atmospheric_transforms.transforms),
            'quality_degradation_types': ['compression', 'noise', 'blur']
        }
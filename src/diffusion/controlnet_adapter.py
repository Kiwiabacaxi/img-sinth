"""
Adaptador ControlNet para controle preciso de geração de pastagens
Suporte a múltiplos tipos de condicionamento para aplicações agronômicas
"""

import torch
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path
from controlnet_aux import (
    CannyDetector, 
    MidasDetector, 
    MLSDdetector,
    OpenposeDetector,
    HEDdetector,
    PidiNetDetector
)
from controlnet_aux.processor import Processor

logger = logging.getLogger(__name__)

class ControlNetAdapter:
    """
    Adaptador para ControlNet com pré-processadores específicos
    para aplicações agronômicas em pastagens brasileiras
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processors = {}
        self.supported_types = [
            'canny', 'depth', 'mlsd', 'openpose', 
            'hed', 'pidinet', 'scribble'
        ]
        
        # Configurações otimizadas para pastagens
        self.pasture_configs = {
            'canny': {
                'low_threshold': 50,
                'high_threshold': 200,
                'detect_resolution': 512,
                'image_resolution': 1024
            },
            'depth': {
                'detect_resolution': 384,
                'image_resolution': 1024
            },
            'mlsd': {
                'thr_v': 0.1,
                'thr_d': 0.1,
                'detect_resolution': 512,
                'image_resolution': 1024
            },
            'openpose': {
                'detect_resolution': 512,
                'image_resolution': 1024
            }
        }
        
        self._initialize_processors()
        
    def _initialize_processors(self):
        """Inicializa pré-processadores ControlNet"""
        try:
            # Canny Edge Detection - ideal para definir limites de plantas
            self.processors['canny'] = CannyDetector()
            logger.info("✅ Canny detector inicializado")
            
            # Depth estimation - útil para topografia de pastagens  
            self.processors['depth'] = MidasDetector.from_pretrained('lllyasviel/Annotators')
            logger.info("✅ Depth detector inicializado")
            
            # Line detection - para estruturas lineares (cercas, trilhas)
            self.processors['mlsd'] = MLSDdetector.from_pretrained('lllyasviel/Annotators')
            logger.info("✅ MLSD detector inicializado")
            
            # OpenPose - pode ser usado para gado (se presente)
            self.processors['openpose'] = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
            logger.info("✅ OpenPose detector inicializado")
            
            # HED edge detection - alternativa ao Canny
            self.processors['hed'] = HEDdetector.from_pretrained('lllyasviel/Annotators')
            logger.info("✅ HED detector inicializado")
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao inicializar alguns processors: {e}")
            
    def process_condition_image(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        control_type: str,
        **kwargs
    ) -> Image.Image:
        """
        Processa imagem de condicionamento para ControlNet
        
        Args:
            image: Imagem de entrada (PIL, numpy array ou path)
            control_type: Tipo de condicionamento
            **kwargs: Parâmetros específicos do processor
            
        Returns:
            Imagem processada para ControlNet
        """
        
        if control_type not in self.supported_types:
            raise ValueError(f"Tipo '{control_type}' não suportado. Tipos disponíveis: {self.supported_types}")
            
        if control_type not in self.processors:
            raise RuntimeError(f"Processor '{control_type}' não inicializado")
            
        # Carregar imagem se for string (path)
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Formato de imagem não suportado")
            
        # Obter configuração específica para pastagens
        config = self.pasture_configs.get(control_type, {})
        config.update(kwargs)  # Sobrescrever com kwargs
        
        logger.info(f"Processando imagem com {control_type}")
        
        try:
            processor = self.processors[control_type]
            
            if control_type == 'canny':
                processed = processor(
                    image, 
                    low_threshold=config.get('low_threshold', 100),
                    high_threshold=config.get('high_threshold', 200),
                    detect_resolution=config.get('detect_resolution', 512),
                    image_resolution=config.get('image_resolution', 1024)
                )
                
            elif control_type == 'depth':
                processed = processor(
                    image,
                    detect_resolution=config.get('detect_resolution', 384),
                    image_resolution=config.get('image_resolution', 1024)
                )
                
            elif control_type == 'mlsd':
                processed = processor(
                    image,
                    thr_v=config.get('thr_v', 0.1),
                    thr_d=config.get('thr_d', 0.1),
                    detect_resolution=config.get('detect_resolution', 512),
                    image_resolution=config.get('image_resolution', 1024)
                )
                
            elif control_type == 'openpose':
                processed = processor(
                    image,
                    detect_resolution=config.get('detect_resolution', 512),
                    image_resolution=config.get('image_resolution', 1024)
                )
                
            elif control_type == 'hed':
                processed = processor(
                    image,
                    detect_resolution=config.get('detect_resolution', 512),
                    image_resolution=config.get('image_resolution', 1024),
                    scribble=config.get('scribble', False)
                )
                
            else:
                # Usar processor genérico
                processed = processor(image, **config)
                
            logger.info(f"✅ Imagem processada com {control_type}")
            return processed
            
        except Exception as e:
            logger.error(f"❌ Erro no processamento {control_type}: {e}")
            raise
            
    def create_pasture_layout_mask(
        self, 
        width: int = 1024, 
        height: int = 1024,
        layout_type: str = "uniform"
    ) -> Image.Image:
        """
        Cria máscaras de layout específicas para pastagens
        
        Args:
            width, height: Dimensões da máscara
            layout_type: Tipo de layout ('uniform', 'patchy', 'degraded', 'invaded')
            
        Returns:
            Máscara de layout como Image.Image
        """
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if layout_type == "uniform":
            # Layout uniforme - gradiente suave
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask = (255 * (1 - distance / distance.max())).astype(np.uint8)
            
        elif layout_type == "patchy":
            # Layout irregular - patches de diferentes densidades
            from skimage.filters import gaussian
            noise = np.random.random((height // 4, width // 4))
            noise = cv2.resize(noise, (width, height))
            mask = (255 * gaussian(noise, sigma=50)).astype(np.uint8)
            
        elif layout_type == "degraded":
            # Layout degradado - concentração central
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            
            # Múltiplos centros de degradação
            degraded_centers = [
                (center_y + height//4, center_x - width//4),
                (center_y - height//6, center_x + width//3),
                (center_y + height//6, center_x + width//6)
            ]
            
            for cy, cx in degraded_centers:
                distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                influence = np.exp(-distance / (width * 0.15))
                mask = np.maximum(mask, (255 * influence).astype(np.uint8))
                
        elif layout_type == "invaded":
            # Layout com invasões - clusters irregulares
            # Simular patches de plantas invasoras
            num_patches = np.random.randint(3, 8)
            
            for _ in range(num_patches):
                # Centro do patch aleatório
                cy = np.random.randint(height // 6, 5 * height // 6)
                cx = np.random.randint(width // 6, 5 * width // 6)
                
                # Tamanho do patch
                patch_size = np.random.randint(width // 15, width // 8)
                
                # Criar patch irregular
                y, x = np.ogrid[:height, :width]
                distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                # Forma irregular usando ruído
                angle = np.arctan2(y - cy, x - cx)
                irregularity = 1 + 0.3 * np.sin(4 * angle + np.random.random() * 2 * np.pi)
                
                patch_mask = distance < patch_size * irregularity
                mask[patch_mask] = np.maximum(mask[patch_mask], 
                                            (255 * np.exp(-distance[patch_mask] / patch_size)).astype(np.uint8))
        
        # Aplicar suavização para transições naturais
        mask = cv2.GaussianBlur(mask, (21, 21), 10)
        
        # Converter para PIL Image
        mask_img = Image.fromarray(mask).convert('RGB')
        
        logger.info(f"✅ Máscara de layout '{layout_type}' criada ({width}x{height})")
        return mask_img
        
    def create_seasonal_depth_map(
        self,
        width: int = 1024,
        height: int = 1024, 
        season: str = "seca",
        biome: str = "cerrado"
    ) -> Image.Image:
        """
        Cria mapas de profundidade simulando topografia por estação e bioma
        
        Args:
            width, height: Dimensões do mapa
            season: Estação ('seca', 'chuvas', 'transicao')
            biome: Bioma ('cerrado', 'mata_atlantica', 'pampa')
            
        Returns:
            Mapa de profundidade como Image.Image
        """
        
        # Configurações topográficas por bioma
        topo_configs = {
            'cerrado': {'relief_strength': 0.3, 'undulation': 2},
            'mata_atlantica': {'relief_strength': 0.8, 'undulation': 4}, 
            'pampa': {'relief_strength': 0.2, 'undulation': 1}
        }
        
        config = topo_configs.get(biome, topo_configs['cerrado'])
        
        # Criar topografia base
        x = np.linspace(0, config['undulation'] * 2 * np.pi, width)
        y = np.linspace(0, config['undulation'] * 2 * np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        # Ondulações base
        Z = (np.sin(X) * np.cos(Y) + 
             0.5 * np.sin(2*X + np.pi/3) * np.cos(1.5*Y) +
             0.3 * np.sin(0.5*X) * np.sin(3*Y))
        
        # Normalizar para [0, 1]
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        
        # Aplicar força do relevo
        Z = Z * config['relief_strength'] + 0.5
        
        # Ajustes sazonais
        if season == "chuvas":
            # Chuvas criam pequenas depressões (poças)
            noise = np.random.random((height, width)) * 0.1
            Z = Z - noise * 0.3
        elif season == "seca":
            # Seca enfatiza elevações (solo ressecado)
            Z = Z * 1.2
            
        # Normalizar novamente
        Z = np.clip((Z - Z.min()) / (Z.max() - Z.min()), 0, 1)
        
        # Converter para imagem
        depth_map = (255 * Z).astype(np.uint8)
        depth_img = Image.fromarray(depth_map).convert('RGB')
        
        logger.info(f"✅ Mapa de profundidade criado: {biome} - {season}")
        return depth_img
        
    def process_batch_conditions(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        control_type: str,
        batch_size: int = 4
    ) -> List[Image.Image]:
        """
        Processa batch de imagens de condicionamento
        
        Args:
            images: Lista de imagens
            control_type: Tipo de condicionamento
            batch_size: Tamanho do batch
            
        Returns:
            Lista de imagens processadas
        """
        
        processed = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            logger.info(f"Processando batch {i//batch_size + 1}/{len(images)//batch_size + 1}")
            
            for image in batch:
                try:
                    result = self.process_condition_image(image, control_type)
                    processed.append(result)
                except Exception as e:
                    logger.error(f"Erro no processamento de imagem: {e}")
                    continue
                    
        return processed
        
    def get_recommended_control_type(self, purpose: str) -> str:
        """
        Recomenda tipo de ControlNet baseado no propósito
        
        Args:
            purpose: Propósito ('plant_boundaries', 'terrain', 'layout', 'structure')
            
        Returns:
            Tipo recomendado de ControlNet
        """
        
        recommendations = {
            'plant_boundaries': 'canny',      # Limites precisos de plantas
            'terrain': 'depth',              # Topografia e relevo
            'layout': 'scribble',            # Layout e composição
            'structure': 'mlsd',             # Estruturas lineares (cercas, trilhas)
            'degradation': 'hed'             # Padrões de degradação
        }
        
        return recommendations.get(purpose, 'canny')
        
    def save_condition_image(self, image: Image.Image, filepath: str):
        """Salva imagem de condicionamento"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        image.save(filepath)
        logger.info(f"✅ Imagem salva: {filepath}")
        
    def get_processor_info(self) -> Dict:
        """Retorna informações sobre processadores disponíveis"""
        return {
            'supported_types': self.supported_types,
            'initialized': list(self.processors.keys()),
            'configs': self.pasture_configs
        }
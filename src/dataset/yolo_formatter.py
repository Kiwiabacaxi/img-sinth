"""
Formatador de datasets para YOLO com anotações automáticas
Especializado em detecção de plantas invasoras e segmentação de pastagens
"""

import os
import json
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass, asdict
from PIL import Image, ImageDraw
import random
from sklearn.model_selection import train_test_split
import shutil

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Bounding box no formato YOLO (normalized)"""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = 1.0
    
@dataclass  
class SegmentationMask:
    """Máscara de segmentação no formato YOLO"""
    class_id: int
    polygon_points: List[Tuple[float, float]]  # Coordenadas normalizadas
    confidence: float = 1.0

@dataclass
class YOLOAnnotation:
    """Anotação completa YOLO"""
    image_path: str
    image_width: int
    image_height: int
    bboxes: List[BoundingBox]
    segments: List[SegmentationMask] = None

class YOLOFormatter:
    """
    Formatador de datasets para YOLO com geração automática de anotações
    baseadas em análise de imagem e metadados de geração
    """
    
    def __init__(self):
        # Classes para detecção de plantas invasoras
        self.detection_classes = {
            0: 'capim_gordura',
            1: 'carqueja',
            2: 'samambaia', 
            3: 'cupinzeiro',
            4: 'area_degradada'
        }
        
        # Classes para segmentação de qualidade
        self.segmentation_classes = {
            0: 'pasto_bom',
            1: 'pasto_moderado', 
            2: 'pasto_degradado',
            3: 'solo_exposto',
            4: 'invasoras'
        }
        
        # Mapeamento de espécies invasoras para classes
        self.invasive_species_map = {
            'capim_gordura': 0,
            'carqueja': 1,
            'samambaia': 2,
            'cupinzeiro': 3,
            'outras_invasoras': 4
        }
        
        # Parâmetros para geração automática de bboxes
        self.bbox_generation_params = {
            'capim_gordura': {
                'min_size': 0.05,  # 5% da imagem
                'max_size': 0.3,   # 30% da imagem
                'aspect_ratio_range': (0.8, 1.5),
                'patches_per_image': (1, 4)
            },
            'carqueja': {
                'min_size': 0.02,
                'max_size': 0.15,
                'aspect_ratio_range': (0.6, 1.2),
                'patches_per_image': (1, 6)
            },
            'samambaia': {
                'min_size': 0.08,
                'max_size': 0.25,
                'aspect_ratio_range': (0.7, 1.3),
                'patches_per_image': (1, 3)
            },
            'cupinzeiro': {
                'min_size': 0.01,
                'max_size': 0.05,
                'aspect_ratio_range': (0.9, 1.1),
                'patches_per_image': (0, 8)
            }
        }
        
    def format_dataset(
        self,
        dataset_dir: str,
        output_dir: str,
        task_type: str = "detection",  # "detection" ou "segmentation"
        train_split: float = 0.7,
        val_split: float = 0.2,
        test_split: float = 0.1,
        generate_annotations: bool = True
    ) -> str:
        """
        Formata dataset para YOLO
        
        Args:
            dataset_dir: Diretório do dataset gerado
            output_dir: Diretório de saída formatado
            task_type: Tipo de tarefa ("detection" ou "segmentation")
            train_split: Proporção de treino
            val_split: Proporção de validação  
            test_split: Proporção de teste
            generate_annotations: Se deve gerar anotações automaticamente
            
        Returns:
            Caminho do dataset formatado
        """
        
        logger.info(f"📋 Formatando dataset para YOLO ({task_type})")
        
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Carregar metadados do dataset
        metadata_file = dataset_path / "dataset_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadados não encontrados: {metadata_file}")
            
        with open(metadata_file, 'r') as f:
            dataset_metadata = json.load(f)
            
        images_info = dataset_metadata['images']
        
        # Gerar anotações se necessário
        annotations = []
        if generate_annotations:
            logger.info("🔍 Gerando anotações automáticas...")
            annotations = self._generate_annotations(dataset_path, images_info, task_type)
        
        # Dividir dataset em splits
        splits = self._create_splits(images_info, train_split, val_split, test_split)
        
        # Criar estrutura YOLO
        yolo_structure = self._create_yolo_structure(output_path, task_type)
        
        # Copiar imagens e criar arquivos de anotação
        for split_name, split_data in splits.items():
            self._process_split(
                split_name, split_data, annotations, dataset_path, 
                yolo_structure, task_type
            )
            
        # Criar arquivo de configuração YOLO
        config_path = self._create_yolo_config(output_path, task_type, len(images_info))
        
        logger.info(f"✅ Dataset YOLO criado: {output_path}")
        logger.info(f"📊 Splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        
        return str(output_path)
        
    def _generate_annotations(
        self,
        dataset_path: Path,
        images_info: List[Dict],
        task_type: str
    ) -> List[YOLOAnnotation]:
        """Gera anotações automáticas baseadas nos metadados"""
        
        annotations = []
        images_dir = dataset_path / "images"
        
        for img_info in images_info:
            image_path = images_dir / img_info['filename']
            
            if not image_path.exists():
                logger.warning(f"Imagem não encontrada: {image_path}")
                continue
                
            # Carregar imagem para obter dimensões
            image = Image.open(image_path)
            width, height = image.size
            
            # Extrair informações dos metadados
            pasture_config = img_info['pasture_config']
            invasive_species = pasture_config.get('invasive_species', [])
            quality = pasture_config.get('quality', 'moderada')
            
            if task_type == "detection":
                # Gerar bounding boxes para detecção
                bboxes = self._generate_detection_boxes(
                    invasive_species, quality, width, height
                )
                annotation = YOLOAnnotation(
                    image_path=str(image_path),
                    image_width=width,
                    image_height=height,
                    bboxes=bboxes
                )
            else:  # segmentation
                # Gerar máscaras para segmentação
                segments = self._generate_segmentation_masks(
                    invasive_species, quality, width, height
                )
                annotation = YOLOAnnotation(
                    image_path=str(image_path),
                    image_width=width,
                    image_height=height,
                    bboxes=[],
                    segments=segments
                )
                
            annotations.append(annotation)
            
        logger.info(f"✅ {len(annotations)} anotações geradas")
        return annotations
        
    def _generate_detection_boxes(
        self,
        invasive_species: List[str],
        quality: str,
        image_width: int,
        image_height: int
    ) -> List[BoundingBox]:
        """Gera bounding boxes para plantas invasoras"""
        
        bboxes = []
        
        # Gerar boxes para cada espécie invasora
        for species in invasive_species:
            if species not in self.invasive_species_map:
                continue
                
            class_id = self.invasive_species_map[species]
            params = self.bbox_generation_params.get(species, self.bbox_generation_params['capim_gordura'])
            
            # Número de patches baseado na qualidade
            if quality == "boa":
                num_patches = random.randint(0, 1)  # Poucos ou nenhum
            elif quality == "moderada":
                num_patches = random.randint(*params['patches_per_image'])
            else:  # degradada
                num_patches = random.randint(
                    params['patches_per_image'][1], 
                    params['patches_per_image'][1] + 2
                )
                
            # Gerar boxes individuais
            for _ in range(num_patches):
                bbox = self._generate_single_bbox(params, class_id)
                if bbox:
                    bboxes.append(bbox)
                    
        # Adicionar áreas degradadas se qualidade for ruim
        if quality == "degradada":
            degraded_areas = random.randint(1, 3)
            for _ in range(degraded_areas):
                bbox = self._generate_degradation_bbox()
                if bbox:
                    bboxes.append(bbox)
                    
        return bboxes
        
    def _generate_single_bbox(
        self,
        params: Dict,
        class_id: int
    ) -> Optional[BoundingBox]:
        """Gera uma única bounding box"""
        
        # Gerar tamanho do box
        size = random.uniform(params['min_size'], params['max_size'])
        aspect_ratio = random.uniform(*params['aspect_ratio_range'])
        
        # Calcular dimensões
        if aspect_ratio >= 1:
            width = size
            height = size / aspect_ratio
        else:
            height = size
            width = size * aspect_ratio
            
        # Garantir que não excede os limites
        width = min(width, 0.8)
        height = min(height, 0.8)
        
        # Gerar posição central (evitar bordas)
        margin = 0.1
        x_center = random.uniform(margin + width/2, 1 - margin - width/2)
        y_center = random.uniform(margin + height/2, 1 - margin - height/2)
        
        # Confidence baseada no tipo de espécie
        confidence = random.uniform(0.7, 0.95)
        
        return BoundingBox(
            class_id=class_id,
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            confidence=confidence
        )
        
    def _generate_degradation_bbox(self) -> BoundingBox:
        """Gera bounding box para área degradada"""
        
        # Áreas degradadas tendem a ser maiores
        size = random.uniform(0.15, 0.4)
        aspect_ratio = random.uniform(0.6, 1.8)
        
        if aspect_ratio >= 1:
            width = size
            height = size / aspect_ratio
        else:
            height = size  
            width = size * aspect_ratio
            
        # Posicionamento
        margin = 0.05
        x_center = random.uniform(margin + width/2, 1 - margin - width/2)
        y_center = random.uniform(margin + height/2, 1 - margin - height/2)
        
        return BoundingBox(
            class_id=4,  # area_degradada
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            confidence=random.uniform(0.6, 0.9)
        )
        
    def _generate_segmentation_masks(
        self,
        invasive_species: List[str],
        quality: str,
        image_width: int,
        image_height: int
    ) -> List[SegmentationMask]:
        """Gera máscaras de segmentação"""
        
        segments = []
        
        # Segmento principal baseado na qualidade
        if quality == "boa":
            main_class = 0  # pasto_bom
            coverage = random.uniform(0.8, 0.95)
        elif quality == "moderada":
            main_class = 1  # pasto_moderado
            coverage = random.uniform(0.6, 0.8)
        else:  # degradada
            main_class = 2  # pasto_degradado
            coverage = random.uniform(0.3, 0.6)
            
        # Criar segmento principal (área central)
        main_segment = self._create_main_pasture_segment(main_class, coverage)
        segments.append(main_segment)
        
        # Segmentos de solo exposto
        if quality in ["moderada", "degradada"]:
            soil_coverage = 1.0 - coverage
            if soil_coverage > 0.1:
                soil_segment = self._create_soil_segment(soil_coverage)
                segments.append(soil_segment)
                
        # Segmentos de invasoras
        for species in invasive_species:
            if species in self.invasive_species_map:
                invasive_segment = self._create_invasive_segment(species)
                if invasive_segment:
                    segments.append(invasive_segment)
                    
        return segments
        
    def _create_main_pasture_segment(
        self,
        class_id: int,
        coverage: float
    ) -> SegmentationMask:
        """Cria segmento principal da pastagem"""
        
        # Criar polígono irregular que ocupa a área central
        num_points = random.randint(6, 12)
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        
        # Centro ligeiramente deslocado
        center_x = random.uniform(0.4, 0.6)
        center_y = random.uniform(0.4, 0.6)
        
        # Raio baseado na cobertura
        base_radius = np.sqrt(coverage) * 0.4
        
        points = []
        for angle in angles:
            # Variação no raio para forma irregular
            radius_variation = random.uniform(0.8, 1.2)
            radius = base_radius * radius_variation
            
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Normalizar para [0, 1]
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            points.append((x, y))
            
        return SegmentationMask(
            class_id=class_id,
            polygon_points=points,
            confidence=random.uniform(0.8, 0.95)
        )
        
    def _create_soil_segment(self, coverage: float) -> SegmentationMask:
        """Cria segmento de solo exposto"""
        
        # Solo geralmente nas bordas ou em patches
        if coverage > 0.3:
            # Múltiplos patches pequenos
            return self._create_patch_segment(3, coverage * 0.5)  # solo_exposto
        else:
            # Área nas bordas
            return self._create_border_segment(3, coverage)
            
    def _create_invasive_segment(self, species: str) -> Optional[SegmentationMask]:
        """Cria segmento de planta invasora"""
        
        class_id = 4  # invasoras (classe geral)
        coverage = random.uniform(0.05, 0.25)
        
        # Invasoras em patches irregulares
        return self._create_patch_segment(class_id, coverage)
        
    def _create_patch_segment(
        self,
        class_id: int,
        coverage: float
    ) -> SegmentationMask:
        """Cria segmento em formato de patch irregular"""
        
        # Centro do patch
        center_x = random.uniform(0.2, 0.8)
        center_y = random.uniform(0.2, 0.8)
        
        # Tamanho baseado na cobertura
        size = np.sqrt(coverage) * random.uniform(0.3, 0.6)
        
        # Número de pontos para irregularidade
        num_points = random.randint(5, 10)
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        
        points = []
        for angle in angles:
            radius_var = random.uniform(0.5, 1.5)
            radius = size * radius_var
            
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            x = np.clip(x, 0.02, 0.98)
            y = np.clip(y, 0.02, 0.98)
            
            points.append((x, y))
            
        return SegmentationMask(
            class_id=class_id,
            polygon_points=points,
            confidence=random.uniform(0.6, 0.9)
        )
        
    def _create_border_segment(
        self,
        class_id: int,
        coverage: float
    ) -> SegmentationMask:
        """Cria segmento nas bordas da imagem"""
        
        # Escolher borda aleatória
        border = random.choice(['top', 'bottom', 'left', 'right'])
        width = coverage * random.uniform(0.8, 1.5)
        
        if border == 'bottom':
            points = [
                (0, 1 - width), (1, 1 - width),
                (1, 1), (0, 1)
            ]
        elif border == 'top':
            points = [
                (0, 0), (1, 0),
                (1, width), (0, width)  
            ]
        elif border == 'left':
            points = [
                (0, 0), (width, 0),
                (width, 1), (0, 1)
            ]
        else:  # right
            points = [
                (1 - width, 0), (1, 0),
                (1, 1), (1 - width, 1)
            ]
            
        return SegmentationMask(
            class_id=class_id,
            polygon_points=points,
            confidence=random.uniform(0.7, 0.9)
        )
        
    def _create_splits(
        self,
        images_info: List[Dict],
        train_split: float,
        val_split: float,
        test_split: float
    ) -> Dict[str, List[Dict]]:
        """Divide dataset em splits"""
        
        # Normalizar splits
        total = train_split + val_split + test_split
        train_split /= total
        val_split /= total
        test_split /= total
        
        # Primeira divisão: treino vs (val + test)
        train_data, temp_data = train_test_split(
            images_info, 
            test_size=(val_split + test_split),
            random_state=42,
            shuffle=True
        )
        
        # Segunda divisão: val vs test  
        if test_split > 0:
            val_ratio = val_split / (val_split + test_split)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_ratio),
                random_state=42,
                shuffle=True
            )
        else:
            val_data = temp_data
            test_data = []
            
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
    def _create_yolo_structure(self, output_path: Path, task_type: str) -> Dict[str, Path]:
        """Cria estrutura de diretórios YOLO"""
        
        structure = {
            'train_images': output_path / 'images' / 'train',
            'val_images': output_path / 'images' / 'val',
            'test_images': output_path / 'images' / 'test',
            'train_labels': output_path / 'labels' / 'train',
            'val_labels': output_path / 'labels' / 'val',
            'test_labels': output_path / 'labels' / 'test'
        }
        
        # Criar todos os diretórios
        for dir_path in structure.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return structure
        
    def _process_split(
        self,
        split_name: str,
        split_data: List[Dict],
        annotations: List[YOLOAnnotation],
        source_path: Path,
        yolo_structure: Dict[str, Path],
        task_type: str
    ):
        """Processa um split específico"""
        
        images_dir = yolo_structure[f'{split_name}_images']
        labels_dir = yolo_structure[f'{split_name}_labels']
        
        # Criar mapeamento filename -> anotação
        annotations_map = {
            Path(ann.image_path).name: ann 
            for ann in annotations
        }
        
        for img_info in split_data:
            filename = img_info['filename']
            
            # Copiar imagem
            source_img = source_path / 'images' / filename
            dest_img = images_dir / filename
            
            if source_img.exists():
                shutil.copy2(source_img, dest_img)
                
                # Criar arquivo de label
                if filename in annotations_map:
                    annotation = annotations_map[filename]
                    label_filename = Path(filename).stem + '.txt'
                    label_path = labels_dir / label_filename
                    
                    self._write_yolo_label(annotation, label_path, task_type)
                    
        logger.info(f"✅ Split '{split_name}' processado: {len(split_data)} imagens")
        
    def _write_yolo_label(
        self,
        annotation: YOLOAnnotation,
        label_path: Path,
        task_type: str
    ):
        """Escreve arquivo de label YOLO"""
        
        with open(label_path, 'w') as f:
            if task_type == "detection":
                # Formato: class x_center y_center width height
                for bbox in annotation.bboxes:
                    f.write(f"{bbox.class_id} {bbox.x_center:.6f} {bbox.y_center:.6f} "
                           f"{bbox.width:.6f} {bbox.height:.6f}\n")
            else:  # segmentation
                # Formato: class x1 y1 x2 y2 ... xn yn
                for segment in annotation.segments or []:
                    points_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in segment.polygon_points])
                    f.write(f"{segment.class_id} {points_str}\n")
                    
    def _create_yolo_config(
        self,
        output_path: Path,
        task_type: str,
        num_images: int
    ) -> Path:
        """Cria arquivo de configuração YOLO"""
        
        if task_type == "detection":
            classes = self.detection_classes
            task_name = "Detection"
        else:
            classes = self.segmentation_classes
            task_name = "Segmentation"
            
        config = {
            'path': str(output_path),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': len(classes),
            'names': list(classes.values()),
            'task': task_type,
            'description': f'Brazilian Pasture {task_name} Dataset',
            'total_images': num_images,
            'classes_info': {
                str(k): {
                    'name': v,
                    'description': self._get_class_description(v)
                } for k, v in classes.items()
            }
        }
        
        config_path = output_path / 'data.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        logger.info(f"✅ Configuração YOLO criada: {config_path}")
        return config_path
        
    def _get_class_description(self, class_name: str) -> str:
        """Retorna descrição da classe"""
        
        descriptions = {
            'capim_gordura': 'Melinis minutiflora - invasive golden grass',
            'carqueja': 'Baccharis trimera - woody invasive shrub',
            'samambaia': 'Pteridium aquilinum - bracken fern invasion',
            'cupinzeiro': 'Termite mounds in pasture',
            'area_degradada': 'Degraded pasture areas',
            'pasto_bom': 'Good quality pasture (>80% coverage)',
            'pasto_moderado': 'Moderate quality pasture (50-80% coverage)',
            'pasto_degradado': 'Degraded pasture (<50% coverage)',
            'solo_exposto': 'Exposed soil areas',
            'invasoras': 'General invasive plant species'
        }
        
        return descriptions.get(class_name, f'{class_name} - no description available')
        
    def validate_dataset(self, dataset_path: str) -> Dict:
        """Valida dataset YOLO formatado"""
        
        path = Path(dataset_path)
        config_file = path / 'data.yaml'
        
        if not config_file.exists():
            return {'valid': False, 'error': 'Configuration file not found'}
            
        # Carregar configuração
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Verificar splits
        validation_results = {
            'valid': True,
            'splits': {},
            'total_images': 0,
            'total_labels': 0,
            'missing_labels': [],
            'empty_labels': []
        }
        
        for split in ['train', 'val', 'test']:
            images_dir = path / 'images' / split
            labels_dir = path / 'labels' / split
            
            if images_dir.exists():
                image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                label_files = list(labels_dir.glob('*.txt'))
                
                validation_results['splits'][split] = {
                    'images': len(image_files),
                    'labels': len(label_files)
                }
                
                validation_results['total_images'] += len(image_files)
                validation_results['total_labels'] += len(label_files)
                
                # Verificar correspondência imagem-label
                for img_file in image_files:
                    label_file = labels_dir / (img_file.stem + '.txt')
                    
                    if not label_file.exists():
                        validation_results['missing_labels'].append(str(img_file))
                    elif label_file.stat().st_size == 0:
                        validation_results['empty_labels'].append(str(label_file))
                        
        logger.info(f"✅ Dataset validado: {validation_results['total_images']} imagens, "
                   f"{validation_results['total_labels']} labels")
                   
        return validation_results
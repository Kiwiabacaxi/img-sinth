"""
Sistema de treinamento YOLO otimizado para pastagens brasileiras
Integra com datasets sintéticos e fornece configurações específicas para agricultura
"""

import os
import yaml
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import time
from datetime import datetime

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics")

logger = logging.getLogger(__name__)

@dataclass
class YOLOTrainingConfig:
    """Configuração para treinamento YOLO"""
    
    # Configurações básicas
    model_size: str = "yolov8s"  # n, s, m, l, x
    task_type: str = "detect"    # detect, segment, classify
    
    # Parâmetros de treinamento
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    
    # Learning rate e otimização
    learning_rate: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    
    # Loss function weights
    box_loss_gain: float = 7.5
    cls_loss_gain: float = 0.5
    dfl_loss_gain: float = 1.5
    
    # Data augmentation
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.1
    
    # Regularização
    dropout: float = 0.0
    
    # Training settings
    patience: int = 20
    save_period: int = 10
    val_period: int = 1
    
    # Hardware settings
    device: str = "auto"
    workers: int = 8
    
    # Output settings
    project: str = "yolo_training"
    name: Optional[str] = None
    exist_ok: bool = True
    
    # Evaluation
    val: bool = True
    plots: bool = True
    verbose: bool = True

class YOLOTrainer:
    """
    Trainer especializado para modelos YOLO em aplicações de pastagens brasileiras
    """
    
    def __init__(
        self,
        config: Optional[YOLOTrainingConfig] = None,
        output_dir: str = "/content/yolo_training"
    ):
        self.config = config or YOLOTrainingConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Histórico de treinamentos
        self.training_history = []
        
        # Configurações otimizadas por tamanho de modelo
        self.model_configs = {
            "yolov8n": {
                "recommended_batch": 32,
                "min_epochs": 80,
                "learning_rate": 0.01
            },
            "yolov8s": {
                "recommended_batch": 16,
                "min_epochs": 100,
                "learning_rate": 0.01
            },
            "yolov8m": {
                "recommended_batch": 8,
                "min_epochs": 120,
                "learning_rate": 0.008
            },
            "yolov8l": {
                "recommended_batch": 4,
                "min_epochs": 150,
                "learning_rate": 0.005
            },
            "yolov8x": {
                "recommended_batch": 2,
                "min_epochs": 200,
                "learning_rate": 0.003
            }
        }
        
        # Configurações específicas para pastagens
        self.pasture_optimizations = {
            "class_weights": {
                "capim_gordura": 1.2,    # Maior peso para invasora importante
                "carqueja": 1.1,
                "samambaia": 1.0,
                "cupinzeiro": 0.9,
                "area_degradada": 1.3     # Alto peso para degradação
            },
            "augmentation_agricultural": {
                "mosaic": 0.8,           # Reduzido para preservar contexto agrícola
                "mixup": 0.05,           # Muito baixo para manter realismo
                "hsv_h": 0.015,          # Pequenas variações de cor
                "hsv_s": 0.4,            # Saturação moderada
                "hsv_v": 0.4,            # Brilho moderado
                "degrees": 2.0,          # Rotações muito pequenas
                "translate": 0.1,        # Translações mínimas
                "scale": 0.1,            # Escala mínima
                "shear": 0.0,            # Sem shear (não natural em vistas aéreas)
                "perspective": 0.0,      # Sem perspectiva artificial
                "flipud": 0.0,           # Sem flip vertical (não natural)
                "fliplr": 0.5,           # Flip horizontal OK
            }
        }
        
        logger.info(f"YOLOTrainer inicializado - Output: {output_dir}")
        
    def optimize_config_for_hardware(self) -> YOLOTrainingConfig:
        """Otimiza configuração baseada no hardware disponível"""
        
        # Detectar GPU e memória
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            
            logger.info(f"GPU detectada: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Ajustar batch size baseado na memória GPU
            if gpu_memory >= 24:  # RTX 4090, A100
                recommended_batch = self.model_configs[self.config.model_size]["recommended_batch"]
            elif gpu_memory >= 16:  # V100, RTX 3080/4080
                recommended_batch = max(self.model_configs[self.config.model_size]["recommended_batch"] // 1.5, 2)
            elif gpu_memory >= 8:   # T4, RTX 3060
                recommended_batch = max(self.model_configs[self.config.model_size]["recommended_batch"] // 2, 1)
            else:  # GPUs menores
                recommended_batch = max(self.model_configs[self.config.model_size]["recommended_batch"] // 4, 1)
                
            # Ajustar workers baseado na GPU
            if "T4" in gpu_name:
                self.config.workers = 4
            elif "V100" in gpu_name:
                self.config.workers = 8
            else:
                self.config.workers = min(8, os.cpu_count() or 4)
                
            self.config.batch_size = int(recommended_batch)
            self.config.device = "0"  # Primeira GPU
            
        else:
            logger.warning("CUDA não disponível - usando CPU")
            self.config.batch_size = 2
            self.config.workers = min(4, os.cpu_count() or 2)
            self.config.device = "cpu"
            
        return self.config
        
    def validate_dataset(self, dataset_path: str) -> bool:
        """Valida se o dataset está no formato correto para YOLO"""
        
        dataset_file = Path(dataset_path)
        
        if not dataset_file.exists():
            logger.error(f"Dataset não encontrado: {dataset_path}")
            return False
            
        # Carregar configuração do dataset
        try:
            with open(dataset_file, 'r') as f:
                dataset_config = yaml.safe_load(f)
                
            # Verificar campos obrigatórios
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in dataset_config:
                    logger.error(f"Campo obrigatório ausente no dataset: {field}")
                    return False
                    
            # Verificar se diretórios existem
            dataset_dir = dataset_file.parent
            
            for split in ['train', 'val']:
                if split in dataset_config:
                    split_path = dataset_dir / dataset_config[split]
                    if not split_path.exists():
                        logger.error(f"Diretório {split} não encontrado: {split_path}")
                        return False
                        
            logger.info(f"Dataset válido: {dataset_config['nc']} classes")
            logger.info(f"Classes: {dataset_config['names']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao validar dataset: {e}")
            return False
            
    def train(
        self,
        dataset_path: str,
        custom_config: Optional[Dict] = None,
        resume: bool = False,
        pretrained: bool = True
    ) -> Optional[str]:
        """
        Executa treinamento YOLO
        
        Args:
            dataset_path: Caminho para arquivo data.yaml
            custom_config: Configurações customizadas
            resume: Continuar treinamento anterior
            pretrained: Usar pesos pré-treinados
            
        Returns:
            Caminho do modelo treinado ou None se falhou
        """
        
        logger.info("🚀 Iniciando treinamento YOLO para pastagens brasileiras")
        
        # Validar dataset
        if not self.validate_dataset(dataset_path):
            return None
            
        # Otimizar configuração para hardware
        self.optimize_config_for_hardware()
        
        # Aplicar configurações customizadas
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
        # Definir nome do experimento
        if not self.config.name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.name = f"{self.config.model_size}_{self.config.task_type}_pastures_{timestamp}"
            
        try:
            # Carregar modelo
            model_file = self._get_model_file(pretrained)
            logger.info(f"📥 Carregando modelo: {model_file}")
            
            model = YOLO(model_file)
            
            # Preparar argumentos de treinamento
            train_args = self._prepare_training_args(dataset_path)
            
            logger.info(f"🎯 Configuração do treinamento:")
            logger.info(f"   Modelo: {self.config.model_size}")
            logger.info(f"   Épocas: {self.config.epochs}")
            logger.info(f"   Batch Size: {self.config.batch_size}")
            logger.info(f"   Image Size: {self.config.image_size}")
            logger.info(f"   Device: {self.config.device}")
            logger.info(f"   Workers: {self.config.workers}")
            
            # Executar treinamento
            start_time = time.time()
            
            logger.info("⏳ Iniciando treinamento... (Isso pode demorar várias horas)")
            results = model.train(**train_args)
            
            training_time = time.time() - start_time
            
            # Salvar informações do treinamento
            training_info = {
                "timestamp": datetime.now().isoformat(),
                "config": asdict(self.config),
                "dataset_path": dataset_path,
                "training_time_hours": training_time / 3600,
                "results": results.__dict__ if hasattr(results, '__dict__') else str(results)
            }
            
            self.training_history.append(training_info)
            
            # Caminho do modelo treinado
            model_path = self.output_dir / self.config.name / "weights" / "best.pt"
            
            if model_path.exists():
                logger.info(f"✅ Treinamento concluído em {training_time/3600:.1f} horas!")
                logger.info(f"💾 Modelo salvo: {model_path}")
                
                # Salvar histórico
                self._save_training_history()
                
                return str(model_path)
            else:
                logger.error("❌ Modelo treinado não encontrado no local esperado")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erro durante o treinamento: {e}")
            return None
            
    def _get_model_file(self, pretrained: bool) -> str:
        """Determina arquivo do modelo a ser usado"""
        
        if self.config.task_type == "segment":
            model_file = f"{self.config.model_size}-seg.pt"
        elif self.config.task_type == "classify":
            model_file = f"{self.config.model_size}-cls.pt"
        else:  # detect
            model_file = f"{self.config.model_size}.pt"
            
        return model_file
        
    def _prepare_training_args(self, dataset_path: str) -> Dict:
        """Prepara argumentos para treinamento YOLO"""
        
        # Argumentos base
        train_args = {
            "data": dataset_path,
            "epochs": self.config.epochs,
            "batch": self.config.batch_size,
            "imgsz": self.config.image_size,
            "device": self.config.device,
            "workers": self.config.workers,
            "project": str(self.output_dir),
            "name": self.config.name,
            "exist_ok": self.config.exist_ok,
            "pretrained": True,
            "optimizer": "SGD",
            "verbose": self.config.verbose,
            "seed": 42,  # Para reprodutibilidade
            "deterministic": True
        }
        
        # Parâmetros de otimização
        train_args.update({
            "lr0": self.config.learning_rate,
            "momentum": self.config.momentum,
            "weight_decay": self.config.weight_decay,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_momentum": self.config.warmup_momentum,
            "box": self.config.box_loss_gain,
            "cls": self.config.cls_loss_gain,
            "dfl": self.config.dfl_loss_gain,
        })
        
        # Data augmentation otimizado para pastagens
        if self.config.augment:
            aug_config = self.pasture_optimizations["augmentation_agricultural"]
            train_args.update({
                "mosaic": aug_config["mosaic"],
                "mixup": aug_config["mixup"],
                "copy_paste": self.config.copy_paste,
                "hsv_h": aug_config["hsv_h"],
                "hsv_s": aug_config["hsv_s"],
                "hsv_v": aug_config["hsv_v"],
                "degrees": aug_config["degrees"],
                "translate": aug_config["translate"],
                "scale": aug_config["scale"],
                "shear": aug_config["shear"],
                "perspective": aug_config["perspective"],
                "flipud": aug_config["flipud"],
                "fliplr": aug_config["fliplr"]
            })
        else:
            train_args["augment"] = False
            
        # Configurações de validação
        train_args.update({
            "val": self.config.val,
            "patience": self.config.patience,
            "save": True,
            "save_period": self.config.save_period,
            "val_period": self.config.val_period,
            "plots": self.config.plots,
            "cache": False  # Evitar problemas de memória
        })
        
        return train_args
        
    def _save_training_history(self):
        """Salva histórico de treinamentos"""
        
        history_file = self.output_dir / "training_history.json"
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
                
            logger.info(f"📊 Histórico salvo: {history_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar histórico: {e}")
            
    def resume_training(self, checkpoint_path: str) -> Optional[str]:
        """Retoma treinamento de um checkpoint"""
        
        if not Path(checkpoint_path).exists():
            logger.error(f"Checkpoint não encontrado: {checkpoint_path}")
            return None
            
        try:
            logger.info(f"🔄 Retomando treinamento de: {checkpoint_path}")
            
            model = YOLO(checkpoint_path)
            results = model.train(resume=True)
            
            logger.info("✅ Treinamento retomado com sucesso!")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"❌ Erro ao retomar treinamento: {e}")
            return None
            
    def export_model(
        self,
        model_path: str,
        format_type: str = "onnx",
        optimize: bool = True
    ) -> Optional[str]:
        """
        Exporta modelo treinado para diferentes formatos
        
        Args:
            model_path: Caminho do modelo .pt
            format_type: Formato de exportação (onnx, torchscript, tflite, etc.)
            optimize: Otimizar modelo para inferência
            
        Returns:
            Caminho do modelo exportado
        """
        
        if not Path(model_path).exists():
            logger.error(f"Modelo não encontrado: {model_path}")
            return None
            
        try:
            logger.info(f"📦 Exportando modelo para {format_type}...")
            
            model = YOLO(model_path)
            
            export_args = {
                "format": format_type,
                "optimize": optimize,
                "half": True if torch.cuda.is_available() else False,
                "dynamic": False,
                "simplify": True
            }
            
            exported_path = model.export(**export_args)
            
            logger.info(f"✅ Modelo exportado: {exported_path}")
            return str(exported_path)
            
        except Exception as e:
            logger.error(f"❌ Erro na exportação: {e}")
            return None
            
    def get_training_summary(self) -> Dict:
        """Retorna resumo dos treinamentos realizados"""
        
        if not self.training_history:
            return {"message": "Nenhum treinamento realizado ainda"}
            
        summary = {
            "total_trainings": len(self.training_history),
            "models_trained": [t["config"]["model_size"] for t in self.training_history],
            "total_training_time_hours": sum(t.get("training_time_hours", 0) for t in self.training_history),
            "latest_training": self.training_history[-1] if self.training_history else None
        }
        
        return summary
        
    def create_training_config_from_template(
        self,
        template_name: str = "agricultural_detection"
    ) -> YOLOTrainingConfig:
        """Cria configuração baseada em templates pré-definidos"""
        
        templates = {
            "agricultural_detection": YOLOTrainingConfig(
                model_size="yolov8s",
                task_type="detect",
                epochs=150,
                batch_size=16,
                learning_rate=0.01,
                patience=25,
                augment=True,
                mosaic=0.8,
                mixup=0.05
            ),
            "high_precision": YOLOTrainingConfig(
                model_size="yolov8l",
                task_type="detect", 
                epochs=200,
                batch_size=8,
                learning_rate=0.005,
                patience=30,
                augment=True
            ),
            "fast_inference": YOLOTrainingConfig(
                model_size="yolov8n",
                task_type="detect",
                epochs=100,
                batch_size=32,
                learning_rate=0.015,
                patience=20
            ),
            "segmentation": YOLOTrainingConfig(
                model_size="yolov8m",
                task_type="segment",
                epochs=180,
                batch_size=12,
                learning_rate=0.008,
                patience=25
            )
        }
        
        if template_name in templates:
            logger.info(f"📋 Usando template: {template_name}")
            return templates[template_name]
        else:
            logger.warning(f"Template não encontrado: {template_name}. Usando padrão.")
            return YOLOTrainingConfig()
            
    def cleanup_training_artifacts(self, keep_best: bool = True):
        """Limpa artefatos de treinamento para economizar espaço"""
        
        logger.info("🧹 Limpando artefatos de treinamento...")
        
        for training_dir in self.output_dir.iterdir():
            if training_dir.is_dir():
                weights_dir = training_dir / "weights"
                
                if weights_dir.exists():
                    for weight_file in weights_dir.iterdir():
                        if weight_file.name == "best.pt" and keep_best:
                            continue  # Manter o melhor modelo
                        elif weight_file.suffix == ".pt":
                            weight_file.unlink()
                            logger.info(f"🗑️ Removido: {weight_file}")
                            
                # Remover outros artefatos grandes
                for artifact in ["runs", "tensorboard", "wandb"]:
                    artifact_path = training_dir / artifact
                    if artifact_path.exists():
                        import shutil
                        shutil.rmtree(artifact_path)
                        logger.info(f"🗑️ Removido diretório: {artifact_path}")
                        
        logger.info("✅ Limpeza concluída")
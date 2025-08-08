"""
Brazilian Pasture Synthetic Image Generator

Sistema completo para geração de imagens sintéticas de pastagens brasileiras
usando Stable Diffusion para treinamento de modelos YOLOv8/v9.
"""

__version__ = "1.0.0"
__author__ = "Brazilian Pasture AI Team"
__email__ = "contact@pasture-ai.com"

from .diffusion import PipelineManager, PromptEngine
from .dataset import DatasetGenerator, YOLOFormatter
from .training import YOLOTrainer, ModelEvaluator

__all__ = [
    'PipelineManager',
    'PromptEngine', 
    'DatasetGenerator',
    'YOLOFormatter',
    'YOLOTrainer',
    'ModelEvaluator'
]
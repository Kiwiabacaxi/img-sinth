# Project Structure & Organization

## Evolution: Colab Prototyping → Standalone Production

O projeto segue uma arquitetura evolutiva, começando com prototipagem em Google Colab e migrando gradualmente para um sistema standalone robusto.

### Phase 1: Current Structure (Colab-Centric)

```
SYNTH_IMAGE/
├── notebooks/             # 🧪 COLAB: Prototipagem interativa
│   ├── 00_Setup_Environment.ipynb      # Configuração básica
│   ├── 01_Explore_Prompts.ipynb        # Experimentação com prompts
│   ├── 02_Generate_Dataset.ipynb       # Geração de datasets
│   ├── 03_GrassClover_Brazilian_Style.ipynb  # ⭐ Core: Metodologia GrassClover
│   ├── 04_GrassClover_Simple_Generation.ipynb
│   └── 05_Evaluation_Benchmark.ipynb   # Validação e métricas
├── src/                   # 🚀 STANDALONE: Código modular (em desenvolvimento)
│   ├── diffusion/         # Pipeline Stable Diffusion
│   ├── dataset/           # Geração e processamento
│   └── training/          # Treinamento e avaliação
├── configs/               # ⚙️ Configurações compartilhadas
├── assets/                # 📚 Materiais de referência
├── outputs/               # 📊 Resultados gerados
├── docs/                  # 📖 Documentação (incluindo agent_reference.md)
├── scripts/               # 🔧 Utilitários
└── tests/                 # ✅ Testes automatizados
```

### Phase 2: Target Structure (Standalone-First)

```
SYNTH_IMAGE/
├── src/                   # 🎯 CORE: Sistema standalone principal
│   ├── grassclover/       # Implementação metodologia GrassClover
│   │   ├── generator.py   # Gerador principal
│   │   ├── species/       # Modelos de espécies brasileiras
│   │   └── metrics.py     # Métricas de validação
│   ├── diffusion/         # Pipeline Stable Diffusion otimizado
│   ├── api/               # REST API para integração
│   └── core/              # Componentes base reutilizáveis
├── deployment/            # 🐳 Deployment e containerização
│   ├── docker/            # Docker configurations
│   ├── k8s/               # Kubernetes manifests
│   └── terraform/         # Infrastructure as Code
├── notebooks/             # 📝 Mantidos para experimentação
├── configs/               # ⚙️ Configurações de ambiente
├── tools/                 # 🔨 CLI tools para administração
└── examples/              # 📚 Exemplos de uso da API
```

## Current Colab Architecture (Phase 1)

### 🧪 **Notebooks Workflow** - Prototipagem Sequencial

#### **Core GrassClover Implementation**
1. `03_GrassClover_Brazilian_Style.ipynb` - ⭐ **PRINCIPAL**
   - Implementação completa da metodologia GrassClover
   - Geração ultra-compatível (apenas NumPy + OpenCV + PIL)
   - Classes hierárquicas para pastagens brasileiras
   - Pipeline completo: soil → plantas → segmentação → avaliação
   - Modelo DeepLabV3+ para benchmarking

2. `04_GrassClover_Simple_Generation.ipynb` - Versão simplificada
   - Geração rápida para testes
   - Menor overhead computacional

#### **Supporting Notebooks**
- `00_Setup_Environment.ipynb` - Configuração de ambiente
- `01_Explore_Prompts.ipynb` - Experimentação com prompts
- `02_Generate_Dataset.ipynb` - Geração de datasets diversos
- `05_Evaluation_Benchmark.ipynb` - Métricas e comparações

### 🚀 **Standalone Modules** (src/) - Em Desenvolvimento

#### **Current Implementation**
- `src/diffusion/`: Pipeline Stable Diffusion modular
- `src/dataset/`: Processamento e geração de dados
- `src/training/`: Treinamento de modelos (YOLOv8, DeepLab)

#### **Target Expansion**
```python
src/
├── grassclover/           # 🌾 Core GrassClover brasileiro
│   ├── generator.py       # BrazilianGrassCloverGenerator
│   ├── species/           
│   │   ├── brachiaria.py  # Brachiaria spp. models
│   │   ├── panicum.py     # Panicum spp. models  
│   │   ├── cynodon.py     # Cynodon spp. models
│   │   └── legumes.py     # Leguminosas forrageiras
│   ├── soil/              # Modelos de solo brasileiro
│   │   ├── cerrado.py     # Solos de cerrado
│   │   ├── tropical.py    # Solos tropicais
│   │   └── coastal.py     # Solos costeiros
│   └── metrics/           # Métricas específicas
│       ├── biomass.py     # Análise de biomassa
│       ├── lai.py         # Leaf Area Index
│       └── grassclover.py # Métricas do paper original
├── api/                   # 🌐 REST API
│   ├── routes/            # Endpoints
│   ├── models/            # Pydantic models
│   └── middleware/        # Auth, validation, etc.
└── core/                  # 🔧 Base components
    ├── config.py          # Configuration management
    ├── logging.py         # Structured logging
    └── utils.py           # Shared utilities
```

## Migration Strategy: Colab → Standalone

### 🔄 **Migration Phases**

#### **Phase 1.5: Hybrid** (Next 2-3 months)
- [ ] Extrair código do notebook 03_GrassClover para módulos Python
- [ ] Criar `src/grassclover/` baseado na implementação do notebook
- [ ] Manter notebooks para experimentação rápida
- [ ] Implementar CLI tools para geração batch

#### **Phase 2: API Development** (4-6 months)
- [ ] FastAPI REST endpoints
- [ ] Docker containerization
- [ ] Database integration (PostgreSQL) para metadados
- [ ] Job queue (Celery/RQ) para processamento assíncrono

#### **Phase 3: Production** (6+ months)
- [ ] Kubernetes deployment
- [ ] Monitoring e observabilidade (Prometheus/Grafana)
- [ ] Auto-scaling baseado em load
- [ ] Multi-tenant support

### 🎯 **Key Extraction Points**

#### **From Notebook to Module**
```python
# notebooks/03_GrassClover_Brazilian_Style.ipynb
class BrazilianGrassCloverGenerator:
    # Ultra-compatible implementation
    
↓ EXTRACT TO ↓

# src/grassclover/generator.py  
class BrazilianGrassCloverGenerator:
    # Production-ready with error handling, logging, config
```

#### **Configuration Migration**
```yaml
# Colab: Hardcoded values
image_size = (512, 512)
lai_range = (1.0, 3.5)

↓ MIGRATE TO ↓

# configs/grassclover.yaml
generation:
  image_size: [512, 512]
  lai_range: [1.0, 3.5]
  species_composition:
    brachiaria: 0.4
    panicum: 0.3
    cynodon: 0.2
    legumes: 0.1
```

## Development Conventions

### 🐍 **Python Code Standards**
- **Type hints**: Obrigatório em produção
- **Docstrings**: Google style
- **Error handling**: Custom exceptions + structured logging
- **Testing**: pytest com coverage ≥90%

### 📦 **Package Structure**
```python
# Colab style (notebook)
import numpy as np
import cv2
from PIL import Image

# Production style (standalone)
from grassclover.core import BaseGenerator
from grassclover.species import BrachiariaModel
from grassclover.metrics import calculate_miou
```

### 🔧 **Migration Tools**

#### **Extract Script**
```python
# tools/extract_from_notebook.py
def extract_class_from_notebook(notebook_path, class_name):
    # Extract Python class from Jupyter notebook
    # Convert to standalone .py module
    pass
```

#### **Testing Migration**
```python
# tests/test_migration.py
def test_notebook_vs_module_equivalence():
    # Ensure extracted code produces same results
    notebook_result = run_notebook_cell()
    module_result = run_standalone_module()
    assert notebook_result == module_result
```

## Asset Organization & Data Management

### 📚 **Assets Structure** - Referência e Materiais

```
assets/
├── reference_images/          # 📸 Imagens reais para validação
│   ├── brachiaria/           # Fotos de Brachiaria spp.
│   ├── panicum/              # Fotos de Panicum spp.
│   └── grassclover_original/ # Dataset GrassClover para comparação
├── controlnet_conditions/     # 🎨 Condições para ControlNet
│   ├── edge_maps/            # Mapas de bordas
│   └── depth_maps/           # Mapas de profundidade
├── prompt_examples/           # 💭 Exemplos de prompts curados
└── evaluation_samples/        # 🧪 Amostras para benchmarking
```

### 📊 **Outputs Structure** - Resultados Gerados

```
outputs/
├── grassclover_datasets/      # 🌾 Datasets no formato GrassClover
│   ├── scene_XXXX.png        # Imagens RGB sintéticas
│   ├── scene_XXXX_mask.png   # Máscaras de segmentação
│   └── metadata/             # Metadados JSON por cena
├── models/                   # 🧠 Modelos treinados
│   ├── deeplabv3plus/        # Modelos de segmentação
│   └── yolo/                 # Modelos YOLO (futuro)
├── evaluations/              # 📈 Relatórios de performance
│   ├── miou_reports/         # Relatórios mIoU
│   └── benchmarks/           # Comparações com literatura
└── cache/                    # 💾 Cache temporário
    ├── stable_diffusion/     # Cache de modelos SD
    └── generated_components/ # Componentes gerados (folhas, solo)
```

## Development Standards & Conventions

### 🐍 **Naming Conventions**

#### **Files & Directories**
- **Notebooks**: `##_Description_Name.ipynb` (sequência numerada)
- **Python modules**: `snake_case.py`
- **Configs**: `snake_case.yaml`
- **Docker images**: `grassclover-generator:v1.0`

#### **Code Elements**
```python
# Classes: PascalCase
class BrazilianGrassCloverGenerator:

# Functions/variables: snake_case
def generate_synthetic_scene():
    target_lai = 2.0

# Constants: UPPER_SNAKE_CASE
GRASS_CLOVER_CLASSES = {...}

# Private methods: _leading_underscore
def _apply_natural_effects():
```

#### **Brazilian-Specific Naming**
```python
# Biomes em português
biomes = ['cerrado', 'mata_atlantica', 'pampa']

# Espécies científicas em latim
species = ['Brachiaria brizantha', 'Panicum maximum']

# Termos sazonais em português
seasons = ['seca', 'chuvas', 'transicao']
```

### 📦 **Import Organization**

#### **Colab Style** (Notebooks)
```python
# Ultra-compatible imports
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt

# Avoid problematic imports
# ❌ from sklearn.metrics import ...
# ❌ import scikit-image
# ✅ Use custom implementations
```

#### **Standalone Style** (Production)
```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Third-party
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import yaml
from loguru import logger

# Local modules
from grassclover.core import BaseGenerator
from grassclover.species import BrachiariaModel
from grassclover.metrics import calculate_miou_grassclover
```

### ⚙️ **Configuration Management**

#### **Environment-Specific Configs**
```yaml
# configs/colab.yaml - Para notebooks
environment: "colab"
gpu_memory_fraction: 0.8
batch_size: 2
ultra_compatible_mode: true

# configs/standalone.yaml - Para produção
environment: "standalone" 
gpu_memory_fraction: 0.95
batch_size: 8
ultra_compatible_mode: false
enable_api: true
```

#### **Loading Patterns**
```python
# Colab: Simple loading
with open('configs/grassclover_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Standalone: Environment-aware loading
from grassclover.core.config import load_config
config = load_config(environment='production')
```

### 🧪 **Testing Strategy**

#### **Current (Phase 1)**
```python
# Manual testing in notebooks
def test_grassclover_generation():
    generator = BrazilianGrassCloverGenerator()
    scene = generator.generate_synthetic_scene()
    assert scene['image'] is not None
    print("✅ Generation test passed")
```

#### **Target (Phase 2)**
```python
# tests/test_grassclover.py - Automated testing
import pytest
from grassclover import BrazilianGrassCloverGenerator

class TestGrassCloverGeneration:
    def test_scene_generation(self):
        """Test basic scene generation"""
        generator = BrazilianGrassCloverGenerator()
        scene = generator.generate_synthetic_scene(lai=2.0)
        
        assert scene['image'].size == (512, 512)
        assert 'segmentation_mask' in scene
        assert len(scene['plant_positions']) > 0
    
    def test_species_distribution(self):
        """Test species distribution in generated scenes"""
        generator = BrazilianGrassCloverGenerator()
        
        composition = {
            'brachiaria': 0.4,
            'panicum': 0.3,
            'cynodon': 0.2,
            'legumes': 0.1
        }
        
        scene = generator.generate_synthetic_scene(composition=composition)
        
        # Validate composition ratios
        actual_composition = calculate_actual_composition(scene)
        for species, expected_ratio in composition.items():
            actual_ratio = actual_composition.get(species, 0)
            assert abs(actual_ratio - expected_ratio) < 0.1
```

### 📋 **Documentation Standards**

#### **Docstring Format** (Google Style)
```python
def generate_synthetic_scene(self, target_lai: float = 2.0, 
                           composition: Optional[Dict[str, float]] = None) -> Dict:
    """Generate a synthetic pasture scene following GrassClover methodology.
    
    Args:
        target_lai: Target Leaf Area Index (1.0-3.5)
        composition: Species composition ratios. If None, uses default Brazilian mix.
        
    Returns:
        Dict containing:
            - 'image': PIL Image of the synthetic scene
            - 'segmentation_mask': numpy array with pixel-level labels
            - 'plant_positions': List of plant placement metadata
            - 'metadata': Generation parameters and timestamps
            
    Raises:
        ValueError: If target_lai is outside valid range [1.0, 3.5]
        RuntimeError: If scene generation fails due to resource constraints
        
    Example:
        >>> generator = BrazilianGrassCloverGenerator()
        >>> scene = generator.generate_synthetic_scene(target_lai=2.5)
        >>> scene['image'].save('synthetic_pasture.png')
    """
```

### 🔄 **Migration Checklist**

#### **Notebook → Module Extraction**
- [ ] Extract class definitions to separate .py files
- [ ] Replace hardcoded values with config files
- [ ] Add proper error handling and logging
- [ ] Implement unit tests
- [ ] Add type hints
- [ ] Document public APIs

#### **Dependencies Migration**
```python
# Phase 1: Ultra-compatible (Colab)
dependencies = ['numpy', 'opencv-python-headless', 'pillow', 'matplotlib']

# Phase 2: Full-featured (Standalone)  
dependencies = [
    'torch', 'diffusers', 'transformers',  # AI/ML
    'fastapi', 'uvicorn', 'pydantic',      # API
    'loguru', 'pyyaml', 'click',           # Utils
    'pytest', 'black', 'mypy'             # Development
]
```

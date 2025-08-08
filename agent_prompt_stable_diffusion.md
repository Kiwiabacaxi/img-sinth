# AGENT PROMPT STEERING.md
## 🌱 Gerador de Imagens Sintéticas de Pastagens Brasileiras via Stable Diffusion

---

## 🎯 MISSÃO DO AI AGENT

Você é um **especialista em agricultura de precisão e geração de imagens sintéticas** com foco específico em **pastagens brasileiras**. Sua missão é implementar um sistema completo baseado em **Stable Diffusion** para gerar datasets de alta qualidade destinados ao treinamento de modelos **YOLOv8/v9** para:

- **Detecção de plantas invasoras** em pastagens brasileiras
- **Segmentação de qualidade** de pastagens (boa/moderada/degradada)
- **Classificação de espécies** de gramíneas nativas

**IMPORTANTE**: Todo o desenvolvimento deve ser otimizado para execução no **Google Colab**, com configuração simples via notebooks.

---

## 📚 FUNDAMENTAÇÃO CIENTÍFICA

Sua implementação deve se basear nas evidências dos seguintes estudos:

### **Estudo Principal (Moreno et al., 2023)**
- **mAP 0.91** usando apenas imagens sintéticas do Stable Diffusion
- **mAP 0.99** quando combinado com dados reais
- **Performance superior às GANs** em detecção de plantas invasoras
- **YOLOv8l** como modelo de escolha (10.2ms FDS)

### **Review Abrangente (Chen et al., 2025)**
- **Stable Diffusion amplamente usado** na agricultura
- **Aplicações comprovadas** em detecção de doenças e pragas  
- **Resultados significativamente superiores** a métodos tradicionais

### **Estudo ControlNet (Deng et al., 2025)**
- **10 classes de invasoras** com ControlNet + Stable Diffusion
- **1.4% melhora no mAP@50:95** do YOLOv8
- **FID score 0.98** (qualidade visual excelente)

---

## 🇧🇷 ESPECIFICAÇÕES PARA PASTAGENS BRASILEIRAS

### **Biomas Prioritários:**
```yaml
biomas:
  cerrado:
    características: "Red latosol soil, 0-25° slopes, distinct dry/wet seasons"
    gramíneas: ["Brachiaria brizantha", "Panicum maximum"]
    degradação: "Compaction, bare soil patches, termite mounds"
  
  mata_atlantica:
    características: "Argisol/cambisol, 5-45° slopes, high humidity"
    gramíneas: ["Brachiaria decumbens", "Panicum maximum"]
    degradação: "Invasive ferns, woody species encroachment"
  
  pampa:
    características: "Planosol, gentle hills, constant winds"
    gramíneas: ["Native grasses mix", "Brachiaria species"]
    degradação: "Overgrazing, soil erosion"
```

### **Plantas Invasoras Brasileiras:**
```yaml
invasoras_prioritarias:
  capim_gordura:
    nome_cientifico: "Melinis minutiflora"
    padrão: "Dense patches, golden color when dry"
    densidade: "5-30% coverage"
  
  carqueja:
    nome_cientifico: "Baccharis trimera"
    padrão: "Scattered woody shrubs, green-gray color"
    densidade: "2-20% coverage"
  
  samambaia:
    nome_cientifico: "Pteridium aquilinum"
    padrão: "Clustered growth, bright green fronds"
    densidade: "10-25% coverage"
```

### **Condições Sazonais:**
```yaml
estações:
  seca: # Maio-Setembro
    gramíneas: "Golden, yellow-brown, sparse coverage"
    solo: "30-70% exposed, compacted trails"
    iluminação: "Intense direct sunlight, harsh shadows"
  
  chuvas: # Outubro-Abril  
    gramíneas: "Vibrant green, dense coverage 85-100%"
    solo: "Mostly covered, muddy patches"
    iluminação: "Diffuse light, cloudy conditions"
```

---

## 🛠️ STACK TECNOLÓGICA

### **Ambiente de Execução:**
- **Plataforma**: Google Colab (GPU T4/V100)
- **Python**: 3.8+
- **Repositório**: GitHub com notebooks executáveis

### **Bibliotecas Principais:**
```python
# Core Stable Diffusion
diffusers==0.24.0           # Hugging Face Diffusers
transformers==4.36.0        # Text encoders
accelerate==0.25.0          # GPU optimization
xformers                    # Memory efficiency

# ControlNet Extensions  
controlnet-aux==0.4.0       # ControlNet preprocessors
opencv-python==4.8.1.78    # Image processing

# Computer Vision
ultralytics==8.0.206        # YOLOv8/v9
roboflow==1.1.9            # Dataset management
albumentations==1.3.1      # Data augmentation

# Utilities
wandb==0.16.0              # Experiment tracking
tqdm==4.66.1               # Progress bars
matplotlib==3.7.2          # Visualization
Pillow==10.1.0             # Image handling
```

---

## 📁 ESTRUTURA DO REPOSITÓRIO

```
brazilian-pasture-synthesis/
├── README.md                          # Documentação principal
├── AGENT_PROMPT_STEERING.md          # Este arquivo
├── requirements.txt                   # Dependências Python
├── setup_colab.py                    # Script de configuração automática
│
├── notebooks/                         # Notebooks executáveis no Colab
│   ├── 00_Setup_Environment.ipynb       # Configuração inicial
│   ├── 01_Explore_Prompts.ipynb         # Teste de prompts
│   ├── 02_Generate_Dataset.ipynb        # Geração do dataset
│   ├── 03_Quality_Control.ipynb         # Controle de qualidade
│   ├── 04_YOLO_Training.ipynb          # Treinamento YOLO
│   └── 05_Evaluation_Benchmark.ipynb    # Avaliação final
│
├── src/                               # Código fonte modular
│   ├── __init__.py
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── pipeline_manager.py          # Gerenciador SD pipeline
│   │   ├── prompt_engine.py             # Sistema de prompts
│   │   ├── controlnet_adapter.py        # Integração ControlNet
│   │   └── image_postprocess.py         # Pós-processamento
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── generator.py                 # Gerador de datasets
│   │   ├── augmentation.py              # Data augmentation
│   │   ├── yolo_formatter.py            # Formatação YOLO
│   │   └── quality_metrics.py           # Métricas de qualidade
│   └── training/
│       ├── __init__.py
│       ├── yolo_trainer.py              # Wrapper YOLOv8/v9
│       ├── evaluation.py                # Métricas de avaliação
│       └── benchmark.py                 # Benchmark contra reais
│
├── configs/                           # Arquivos de configuração
│   ├── prompts/
│   │   ├── base_prompts.yaml            # Prompts fundamentais
│   │   ├── seasonal_variations.yaml     # Variações sazonais
│   │   ├── species_specific.yaml        # Prompts por espécie
│   │   └── degradation_patterns.yaml    # Padrões de degradação
│   ├── models/
│   │   ├── stable_diffusion.yaml        # Config SD models
│   │   ├── controlnet.yaml              # Config ControlNet
│   │   └── yolo.yaml                    # Config YOLO training
│   └── generation/
│       ├── dataset_specs.yaml           # Especificações do dataset
│       ├── augmentation.yaml            # Config data augmentation
│       └── quality_thresholds.yaml      # Thresholds de qualidade
│
├── assets/                            # Recursos de referência
│   ├── reference_images/                # Imagens de referência
│   ├── prompt_examples/                 # Exemplos de prompts
│   ├── controlnet_conditions/           # Condições ControlNet
│   └── evaluation_samples/              # Amostras para avaliação
│
└── outputs/                           # Outputs gerados
    ├── generated_images/                # Imagens sintéticas
    ├── datasets/                       # Datasets formatados
    ├── models/                         # Modelos treinados  
    ├── evaluations/                    # Relatórios de avaliação
    └── benchmarks/                     # Resultados de benchmark
```

---

## 🎨 SISTEMA DE PROMPT ENGINEERING

### **Template Base:**
```python
BASE_TEMPLATE = """
{bioma} Brazilian pasture in {season} season, 
{grass_type} grass with {coverage}% coverage,
{invasive_description},
{soil_condition},
{lighting_condition},
realistic drone photography, agricultural field,
high resolution, photorealistic, detailed
"""

NEGATIVE_PROMPT = """
cartoon, painting, illustration, unrealistic, 
blurry, low quality, distorted, artificial looking,
urban elements, buildings, roads, people
"""
```

### **Prompts Específicos por Cenário:**

#### **Pastagem Saudável:**
```yaml
brachiaria_saudavel_chuvas:
  prompt: "Lush Brazilian Brachiaria brizantha pasture in rainy season, vibrant green grass 90% coverage, dense uniform growth, red latosol soil barely visible, morning diffuse light, photorealistic aerial drone view, agricultural field"
  
brachiaria_saudavel_seca:
  prompt: "Brazilian Brachiaria brizantha pasture in dry season, golden-yellow grass 60% coverage, uniform growth pattern, red soil patches visible, intense midday sunlight, realistic drone photography"
```

#### **Pastagem com Invasoras:**
```yaml
capim_gordura_invasion:
  prompt: "Brazilian Brachiaria pasture invaded by capim-gordura (Melinis minutiflora), golden patches of invasive grass 20% coverage, mixed with native green Brachiaria, patchy distribution, dry season conditions, red soil visible, aerial drone photography"

carqueja_scattered:
  prompt: "Brazilian cerrado pasture with scattered carqueja shrubs (Baccharis trimera), woody invasive plants 15% coverage, green-gray shrubs among native grass, sparse distribution, realistic agricultural photography"
```

#### **Pastagem Degradada:**
```yaml
pastagem_degradada_severa:
  prompt: "Severely degraded Brazilian pasture, 50% bare red soil exposure, compacted cattle trails, sparse dying Brachiaria grass, termite mounds scattered, erosion patterns visible, harsh dry season lighting, drone aerial view"

compactacao_gado:
  prompt: "Brazilian pasture showing cattle compaction damage, bare soil patches along water points, sparse grass coverage 30%, muddy areas, realistic agricultural field conditions, overcast lighting"
```

---

## 🎛️ CONFIGURAÇÃO CONTROLNET

### **Tipos de Condicionamento:**
```python
CONTROLNET_CONFIGS = {
    'canny_edge': {
        'model': 'lllyasviel/sd-controlnet-canny',
        'use_case': 'Plant boundary definition',
        'preprocessing': 'canny_edge_detection'
    },
    
    'depth_map': {
        'model': 'lllyasviel/sd-controlnet-depth', 
        'use_case': 'Terrain topology control',
        'preprocessing': 'depth_estimation'
    },
    
    'segmentation': {
        'model': 'lllyasviel/sd-controlnet-seg',
        'use_case': 'Precise species placement',
        'preprocessing': 'semantic_segmentation'
    },
    
    'sketch': {
        'model': 'lllyasviel/sd-controlnet-scribble',
        'use_case': 'Layout composition control',
        'preprocessing': 'scribble_detection'
    }
}
```

---

## 📊 ESPECIFICAÇÕES DO DATASET

### **Distribuição de Classes:**
```yaml
dataset_composition:
  total_images: 50000  # Inicial para proof of concept
  resolution: 1024x1024
  format: 'jpg'
  
  class_distribution:
    pasto_bom: 30%        # 15,000 imagens - cobertura >80%
    pasto_moderado: 40%   # 20,000 imagens - cobertura 50-80%  
    pasto_degradado: 30%  # 15,000 imagens - cobertura <50%
    
  invasive_distribution:
    capim_gordura: 35%    # Mais comum no cerrado
    carqueja: 25%         # Comum em áreas degradadas
    samambaia: 20%        # Mata atlântica principalmente
    outras_invasoras: 20% # Diversas espécies menores
    
  seasonal_distribution:
    estacao_seca: 45%     # Condição crítica
    estacao_chuvas: 40%   # Condição ideal
    transicao: 15%        # Início/fim estações
    
  biome_distribution:
    cerrado: 50%          # Principal bioma pecuário
    mata_atlantica: 30%   # Importante região
    pampa: 20%            # Região sul
```

### **Variações de Augmentation:**
```python
AUGMENTATION_PIPELINE = {
    'seasonal_shift': {
        'hue_shift': (-15, 30),      # Verde → amarelo (seca)
        'saturation': (-0.2, 0.3),   # Variação de saturação
        'brightness': (-0.15, 0.2)   # Condições de luz
    },
    
    'weather_conditions': {
        'blur': (0, 1.5),           # Condições atmosféricas
        'noise': (0, 0.02),         # Variações de captura
        'contrast': (0.8, 1.3)      # Diferentes iluminações
    },
    
    'spatial_variations': {
        'rotation': (-5, 5),         # Pequenas rotações
        'horizontal_flip': 0.5,      # Espelhamento
        'elastic_transform': 0.3     # Deformações sutis
    }
}
```

---

## 🧪 PIPELINE DE CONTROLE DE QUALIDADE

### **Métricas Automáticas:**
```python
QUALITY_METRICS = {
    'technical_quality': {
        'min_resolution': (1024, 1024),
        'max_blur_score': 100,       # Laplacian variance
        'min_contrast': 0.3,
        'max_noise_level': 0.05
    },
    
    'semantic_quality': {
        'grass_coverage_range': (0.2, 0.95),
        'color_distribution': 'green_yellow_brown_dominant',
        'edge_coherence': 0.7,
        'texture_realism': 0.8
    },
    
    'agricultural_realism': {
        'soil_visibility': (0.05, 0.6),    # Baseado na estação
        'plant_distribution': 'natural_pattern',
        'lighting_consistency': True,
        'seasonal_coherence': True
    }
}
```

### **Filtros de Qualidade:**
```python
def quality_filter_pipeline(image, metadata):
    """
    Pipeline completo de filtros de qualidade
    """
    checks = [
        check_technical_quality(image),
        check_agricultural_realism(image, metadata),
        check_species_coherence(image, metadata['species']),
        check_seasonal_consistency(image, metadata['season']),
        check_biome_characteristics(image, metadata['biome'])
    ]
    
    return all(checks) and calculate_overall_score(image) > 0.75
```

---

## 🎯 CONFIGURAÇÃO YOLO

### **Arquiteturas Suportadas:**
```yaml
yolo_configs:
  detection:
    model: 'yolov8l.pt'  # Baseado no estudo Moreno et al.
    classes:
      0: 'capim_gordura'
      1: 'carqueja'  
      2: 'samambaia'
      3: 'cupinzeiro'
      4: 'area_degradada'
    
  segmentation:
    model: 'yolov8l-seg.pt'
    classes:
      0: 'pasto_bom'
      1: 'pasto_moderado'
      2: 'pasto_degradado'
      3: 'solo_exposto'
      4: 'invasoras'
```

### **Parâmetros de Treinamento:**
```python
TRAINING_CONFIG = {
    'epochs': 200,
    'batch_size': 16,          # Otimizado para Colab
    'learning_rate': 0.01,
    'weight_decay': 0.0005,
    'momentum': 0.937,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'box_loss_gain': 0.05,
    'cls_loss_gain': 0.5,
    'dfl_loss_gain': 1.5,
    'augmentation': True,
    'mosaic': 1.0,
    'mixup': 0.1
}
```

---

## 📈 MÉTRICAS DE BENCHMARK

### **Comparação com Estudos Base:**
```python
BENCHMARK_TARGETS = {
    'moreno_et_al_2023': {
        'metric': 'mAP@0.5',
        'synthetic_only': 0.91,
        'mixed_dataset': 0.99,
        'target': 0.85  # 85% do performance do estudo
    },
    
    'deng_et_al_2025': {
        'metric': 'mAP@0.5:0.95', 
        'improvement': 0.014,     # 1.4% melhora
        'fid_score': 0.98,
        'target_fid': 1.0
    }
}
```

### **Métricas de Avaliação:**
```python
EVALUATION_METRICS = [
    # Detecção
    'mAP@0.5',           # Precisão principal
    'mAP@0.5:0.95',      # Precisão rigorosa  
    'Precision',         # Precisão por classe
    'Recall',            # Recall por classe
    'F1-Score',          # Harmônica prec/recall
    
    # Performance
    'Inference_Time',    # Tempo de inferência
    'FPS',              # Frames por segundo
    'Model_Size',       # Tamanho do modelo
    
    # Qualidade de Imagem
    'FID_Score',        # Fréchet Inception Distance
    'IS_Score',         # Inception Score
    'LPIPS',            # Perceptual similarity
    
    # Domain Gap
    'Real_vs_Synthetic_mAP',  # Diferença performance
    'Cross_Season_Stability', # Estabilidade sazonal
    'Cross_Biome_Transfer'    # Transferência entre biomas
]
```

---

## 🚀 COMANDOS DE EXECUÇÃO

### **Setup Inicial (Colab):**
```bash
# Clone do repositório
!git clone https://github.com/seu-usuario/brazilian-pasture-synthesis.git
%cd brazilian-pasture-synthesis

# Configuração automática
!python setup_colab.py

# Instalação de dependências
!pip install -r requirements.txt

# Verificação de GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")
```

### **Execução do Pipeline Principal:**
```python
# Importar pipeline
from src.pipeline import BrazilianPasturePipeline

# Configurar geração
pipeline = BrazilianPasturePipeline(
    model_name='stabilityai/stable-diffusion-xl-base-1.0',
    use_controlnet=True,
    output_dir='/content/outputs'
)

# Gerar dataset
dataset = pipeline.generate_dataset(
    num_images=5000,      # Para teste inicial
    biomes=['cerrado', 'mata_atlantica'],
    seasons=['seca', 'chuvas'],
    quality_threshold=0.8
)

# Treinar YOLO
model = pipeline.train_yolo(
    dataset_path=dataset,
    model_type='detection',  # ou 'segmentation'
    epochs=100
)

# Avaliar performance  
results = pipeline.evaluate(
    model=model,
    test_dataset='real_validation_set',
    metrics=['mAP', 'FID', 'domain_gap']
)
```

---

## 🎪 PROMPTS DE EXEMPLO PARA TESTES

### **Teste Rápido de Qualidade:**
```python
QUICK_TEST_PROMPTS = [
    # Teste básico - pastagem saudável
    "Brazilian Brachiaria brizantha pasture, vibrant green grass 85% coverage, uniform growth, red latosol soil, morning light, realistic drone photography, high resolution",
    
    # Teste invasoras - capim-gordura
    "Brazilian pasture with capim-gordura invasion, golden Melinis minutiflora patches 25% coverage, mixed with green Brachiaria, patchy distribution, aerial view",
    
    # Teste degradação
    "Degraded Brazilian pasture, 40% bare red soil, sparse dying grass, cattle trails, erosion, harsh sunlight, agricultural drone photography",
    
    # Teste sazonal
    "Brazilian cerrado pasture in dry season, golden-brown grass 50% coverage, red soil visible, intense sunlight, scattered termite mounds, realistic aerial view"
]
```

---

## ⚡ OTIMIZAÇÕES PARA COLAB

### **Configurações de Memória:**
```python
# Otimização de GPU para Colab
COLAB_OPTIMIZATIONS = {
    'torch_compile': True,           # PyTorch 2.0 compilation
    'attention_slicing': True,       # Reduz uso de VRAM
    'cpu_offload': True,            # Offload para CPU quando necessário
    'sequential_cpu_offload': True,  # Offload sequencial
    'enable_xformers': True,        # Otimização de atenção
    'gradient_checkpointing': True, # Trade compute por memória
    'mixed_precision': 'fp16',      # Half precision
    'batch_size_auto': True         # Batch size automático
}
```

### **Configuração de Cache:**
```python
# Cache inteligente para modelos
CACHE_CONFIG = {
    'model_cache_dir': '/content/model_cache',
    'generated_cache_dir': '/content/generated_cache', 
    'preload_models': True,
    'cache_generated': True,
    'max_cache_size': '10GB'
}
```

---

## 📋 DELIVERABLES ESPERADOS

### **Dataset Sintético:**
- ✅ **50.000 imagens** iniciais (1024x1024)
- ✅ **Anotações YOLO** automáticas (bbox + máscaras)
- ✅ **Splits organizados** (70% train, 20% val, 10% test)
- ✅ **Metadados completos** (bioma, estação, espécies, qualidade)

### **Modelos Treinados:**
- ✅ **YOLOv8l detecção** (invasoras + cupinzeiros)  
- ✅ **YOLOv8l segmentação** (qualidade de pastagens)
- ✅ **Performance ≥ 85%** dos benchmarks dos papers
- ✅ **Otimização para edge** (mobile deployment)

### **Benchmarks e Avaliação:**
- ✅ **Comparação com estudos base** (Moreno, Deng, Chen)
- ✅ **Análise de domain gap** (sintético vs real)
- ✅ **Estudo de transferência** entre biomas/estações
- ✅ **Relatórios de qualidade** (FID, IS, LPIPS)

### **Documentação:**
- ✅ **Paper técnico** (metodologia + resultados)
- ✅ **Tutoriais Colab** reproduzíveis
- ✅ **API documentation** completa
- ✅ **Case studies** de aplicação

---

## 🎯 OBJETIVO FINAL

Estabelecer um **pipeline de referência** para geração de imagens sintéticas de pastagens brasileiras que:

1. **Supere os benchmarks** dos estudos internacionais (mAP > 0.85)
2. **Seja facilmente reproduzível** via Google Colab
3. **Cubra cenários brasileiros específicos** (biomas, espécies, condições)
4. **Permita treinamento eficaz** de modelos YOLO para agricultura de precisão
5. **Demonstre viabilidade econômica** vs coleta manual de dados

**O sucesso será medido pela capacidade de treinar modelos YOLO que detectem plantas invasoras em pastagens brasileiras reais com performance comparável a modelos treinados com dados reais, usando apenas imagens sintéticas.**

---

*Este documento serve como guia completo para implementação. Foque na qualidade dos prompts, otimização para Colab, e validação rigorosa contra os benchmarks dos papers citados.*
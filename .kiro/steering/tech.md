# Technology Stack & Build System

## Evolution: Colab → Standalone Architecture

O stack tecnológico está estruturado para suportar a transição de protótipos no Google Colab para sistemas standalone robustos.

## Phase 1: Colab Prototyping Stack

### 🧪 **Ultra-Compatible Core** (Notebook Implementation)

#### **Minimal Dependencies** - Máxima Compatibilidade
```bash
# Ultra-compatible mode - apenas bibliotecas essenciais
pip install "opencv-python-headless" "pillow" "matplotlib" --quiet

# Bibliotecas core sempre disponíveis no Colab
numpy>=1.21.0          # Computação numérica base
opencv-python-headless # Processamento de imagem (sem GUI)
pillow>=9.0.0          # Manipulação de imagens
matplotlib>=3.5.0      # Visualização e plots
```

#### **Avoided Dependencies** - Problemas de Compatibilidade
```bash
# ❌ Evitados por causar conflitos no Colab
scikit-image          # Substituído por implementações OpenCV
scikit-learn          # Métricas customizadas com NumPy
stable-diffusion-sdks # Implementação própria mais leve
tensorflow            # Conflito com PyTorch
```

#### **GrassClover Methodology Implementation**
```python
# Core: BrazilianGrassCloverGenerator
# - Geração procedural usando apenas NumPy + PIL + OpenCV
# - Ultra-compatible: funciona mesmo com dependências mínimas
# - Metodologia: Baseada em Skovsen et al. (CVPR 2019)

class BrazilianGrassCloverGenerator:
    def __init__(self, image_size=(512, 512), ground_sampling_distance=6):
        # Ultra-compatible initialization
    
    def generate_soil_texture(self, soil_type="tropical"):
        # Geração de solo usando OpenCV + NumPy
        
    def generate_grass_blade(self, grass_type="brachiaria"):
        # Geração de folhas usando PIL + matemática
        
    def generate_synthetic_scene(self, target_lai=2.0):
        # Pipeline completo GrassClover
```

### 🎯 **AI/ML Stack** (Quando GPU Disponível)

#### **Deep Learning Core**
```bash
# PyTorch ecosystem para Colab
torch>=2.0.0+cu118      # CUDA 11.8 para Colab
torchvision>=0.15.0     # Computer vision
diffusers==0.24.0       # Stable Diffusion pipeline
transformers==4.36.0    # Text encoders
```

#### **Segmentation Models** 
```python
# DeepLabV3+ implementation para benchmark
class SimpleDeepLabV3Plus(nn.Module):
    # Otimizado para GPU Colab (Tesla T4/V100)
    # ResNet-50 backbone + ASPP + Decoder
    # Target: mIoU ≥0.55 (baseline GrassClover)
```

## Phase 2: Standalone Production Stack

### 🚀 **Production Dependencies**

#### **Core AI/ML Framework**
```bash
# Full-featured ML stack
torch>=2.1.0            # Latest PyTorch
torchvision>=0.16.0     # Computer vision
diffusers>=0.25.0       # Stable Diffusion
transformers>=4.37.0    # NLP models
xformers>=0.0.22        # Memory efficient attention
```

#### **Computer Vision & Processing**
```bash
# Advanced image processing
opencv-python>=4.8.1    # Full OpenCV (não headless)
albumentations>=1.3.1   # Advanced augmentation
pillow-simd>=9.0.0      # SIMD-optimized PIL
imageio>=2.31.0         # Multi-format I/O
scikit-image>=0.21.0    # Scientific image processing
```

#### **API & Web Services**
```bash
# REST API stack
fastapi>=0.104.0        # Modern Python API framework
uvicorn[standard]>=0.24.0  # ASGI server
pydantic>=2.5.0         # Data validation
sqlalchemy>=2.0.0       # Database ORM
alembic>=1.12.0         # Database migrations
redis>=5.0.0            # Caching and sessions
```

#### **Deployment & Infrastructure**
```bash
# Container & orchestration
docker>=24.0.0          # Containerization
kubernetes>=1.28.0      # Container orchestration
helm>=3.13.0            # K8s package management

# Monitoring & observability
prometheus-client>=0.19.0  # Metrics collection
structlog>=23.2.0       # Structured logging
sentry-sdk>=1.38.0      # Error tracking
```

## Hardware Requirements & Optimization

### 📊 **Colab Constraints** (Phase 1)

#### **Free Colab**
```yaml
gpu: Tesla T4 (16GB VRAM)
ram: 12GB system RAM  
storage: 25GB temporary
runtime_limit: 12 hours
session_timeout: 90 minutes

optimizations:
  batch_size: 1-2
  image_size: 512x512 max
  ultra_compatible_mode: true
  model_caching: essential
```

#### **Colab Pro/Pro+**
```yaml
gpu: Tesla V100/A100 (32GB+ VRAM)
ram: 25-50GB system RAM
storage: 100GB+ persistent
runtime_limit: 24 hours
session_timeout: longer

optimizations:
  batch_size: 4-8
  image_size: 1024x1024
  full_features: enabled
  concurrent_generation: supported
```

### 🖥️ **Standalone Requirements** (Phase 2)

#### **Minimum Production**
```yaml
cpu: 8 cores (Intel i7 / AMD Ryzen 7)
ram: 32GB DDR4
gpu: NVIDIA RTX 4070 (12GB VRAM)
storage: 1TB NVMe SSD
network: 1Gbps

performance:
  throughput: 50+ images/hour
  concurrent_users: 10
  api_response_time: <2s
```

#### **Recommended Production**  
```yaml
cpu: 16+ cores (Intel Xeon / AMD EPYC)
ram: 64GB+ DDR4/DDR5
gpu: NVIDIA RTX 4090 / A6000 (24GB+ VRAM)
storage: 2TB+ NVMe SSD RAID
network: 10Gbps

performance:
  throughput: 200+ images/hour
  concurrent_users: 50+
  api_response_time: <1s
  distributed_processing: supported
```

## Development Environment Setup

### 🧪 **Colab Setup** (Current)

#### **Quick Start** 
```python
# Notebook: 03_GrassClover_Brazilian_Style.ipynb
!pip install "opencv-python-headless" "pillow" "matplotlib" --quiet

# GPU check
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")

# Ultra-compatible generator
generator = BrazilianGrassCloverGenerator(image_size=(512, 512))
```

#### **Full Features** (quando GPU disponível)
```python
# Enhanced setup with PyTorch
!pip install torch torchvision --quiet
!pip install diffusers transformers --quiet

# DeepLabV3+ for benchmarking
model = SimpleDeepLabV3Plus(num_classes=7)
model.to("cuda" if torch.cuda.is_available() else "cpu")
```

### 🚀 **Standalone Setup** (Target)

#### **Local Development**
```bash
# Clone repository
git clone https://github.com/Kiwiabacaxi/SYNTH_IMAGE.git
cd SYNTH_IMAGE

# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements/standalone.txt

# Run API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### **Docker Deployment**
```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements/production.txt .
RUN pip install -r production.txt

COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t grassclover-generator:v1.0 .
docker run -p 8000:8000 --gpus all grassclover-generator:v1.0
```

#### **Kubernetes Deployment**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grassclover-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: grassclover-api
  template:
    metadata:
      labels:
        app: grassclover-api
    spec:
      containers:
      - name: api
        image: grassclover-generator:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "8Gi"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            nvidia.com/gpu: 1
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: GPU_MEMORY_FRACTION
          value: "0.95"
```

## Command Line Interfaces

### 🧪 **Colab Commands** (Phase 1)

#### **Basic Dataset Generation**
```python
# Notebook cell execution
generator = BrazilianGrassCloverGenerator(image_size=(512, 512))

# Generate single scene
scene = generator.generate_synthetic_scene(
    target_lai=2.5,
    composition={'brachiaria': 0.4, 'panicum': 0.3, 'cynodon': 0.3}
)

# Export dataset
dataset_path = export_grassclover_dataset(synthetic_dataset)
```

#### **Model Training & Evaluation**
```python
# DeepLabV3+ training
model = SimpleDeepLabV3Plus(num_classes=7)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation metrics
miou, iou_per_class = calculate_miou(pred_mask, true_mask)
pixel_acc = calculate_pixel_accuracy(pred_mask, true_mask)
```

### 🚀 **Standalone Commands** (Phase 2)

#### **CLI Tools**
```bash
# Generate datasets
grassclover generate \
  --biome cerrado \
  --species brachiaria,panicum \
  --size 1000 \
  --output outputs/datasets/cerrado_v1

# Train segmentation model
grassclover train \
  --dataset outputs/datasets/cerrado_v1 \
  --model deeplabv3plus \
  --epochs 50 \
  --batch-size 8

# Run benchmarks
grassclover benchmark \
  --model models/deeplabv3plus_cerrado.pth \
  --test-dataset assets/reference_images/
```

#### **API Server**
```bash
# Development server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production server
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000

# Health check
curl http://localhost:8000/health
```

#### **API Endpoints**
```bash
# Generate single scene
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "lai": 2.5,
    "composition": {
      "brachiaria": 0.4,
      "panicum": 0.3,
      "cynodon": 0.3
    },
    "image_size": [512, 512]
  }'

# Batch generation
curl -X POST "http://localhost:8000/batch-generate" \
  -H "Content-Type: application/json" \
  -d '{
    "num_images": 100,
    "lai_range": [1.0, 3.5],
    "output_format": "grassclover"
  }'
```

## Performance Optimization Strategies

### 🧪 **Colab Optimizations**

#### **Memory Management**
```python
# GPU memory management
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# Model caching
model_cache_dir = "/content/model_cache"
os.makedirs(model_cache_dir, exist_ok=True)

# Ultra-compatible mode
ULTRA_COMPATIBLE_MODE = True  # Use minimal dependencies
```

#### **Colab-Specific Settings**
```python
# Environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Session persistence
from google.colab import drive
drive.mount('/content/drive')
```

### 🚀 **Standalone Optimizations**

#### **Production Configuration**
```yaml
# configs/production.yaml
performance:
  gpu_memory_fraction: 0.95
  mixed_precision: true
  batch_size_auto: true
  xformers_attention: true
  torch_compile: true

caching:
  redis_url: "redis://localhost:6379"
  model_cache_size: "10GB"
  generated_cache_ttl: 3600

api:
  workers: 4
  max_requests_per_worker: 1000
  timeout: 30
```

#### **Docker Optimizations**
```dockerfile
# Multi-stage build for smaller images
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel as builder

# Install production dependencies only
COPY requirements/production.txt .
RUN pip install --no-cache-dir -r production.txt

# Final stage
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
COPY --from=builder /opt/conda /opt/conda

# Optimize for production
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV CUDA_VISIBLE_DEVICES=0
```

## Monitoring & Observability

### 📊 **Metrics Collection**
```python
# Production metrics
from prometheus_client import Counter, Histogram, Gauge

generation_counter = Counter('grassclover_scenes_generated_total')
generation_time = Histogram('grassclover_generation_seconds')
gpu_memory_usage = Gauge('grassclover_gpu_memory_bytes')
```

### 🔍 **Logging & Debugging**
```python
# Structured logging
import structlog

logger = structlog.get_logger()
logger.info(
    "scene_generated",
    scene_id="scene_0001",
    lai=2.5,
    num_plants=45,
    generation_time=12.5
)
```

### 🚨 **Error Tracking**
```python
# Sentry integration
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1
)
```

## Configuration Management

### 🔧 **Environment-Specific Configs**

#### **Development (Colab)**
```yaml
# configs/colab.yaml
environment: colab
debug: true
ultra_compatible_mode: true

generation:
  image_size: [512, 512]
  batch_size: 2
  num_workers: 0

gpu:
  memory_fraction: 0.8
  mixed_precision: false
```

#### **Production (Standalone)**
```yaml  
# configs/production.yaml
environment: production
debug: false
ultra_compatible_mode: false

generation:
  image_size: [1024, 1024] 
  batch_size: 8
  num_workers: 4

gpu:
  memory_fraction: 0.95
  mixed_precision: true

api:
  host: "0.0.0.0"
  port: 8000
  reload: false

database:
  url: "postgresql://user:pass@localhost/grassclover"
  pool_size: 20
  max_overflow: 0

redis:
  url: "redis://localhost:6379"
  db: 0
```

### 🛠️ **Configuration Loading**
```python
# src/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional, List

class GrassCloverSettings(BaseSettings):
    environment: str = "development"
    debug: bool = False
    
    # Generation settings
    default_image_size: List[int] = [512, 512]
    default_lai_range: List[float] = [1.0, 3.5]
    
    # GPU settings
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Load configuration
def load_config(environment: Optional[str] = None) -> GrassCloverSettings:
    if environment:
        os.environ["ENVIRONMENT"] = environment
    return GrassCloverSettings()
```

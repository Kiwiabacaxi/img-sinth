# 📥 Guia de Instalação Completo

Este guia abrange todas as formas de instalar e configurar o Brazilian Pasture Synthetic Image Generator, desde o setup mais simples no Google Colab até instalações avançadas em servidores de pesquisa.

---

## 🚀 **Instalação Rápida (Google Colab) - Recomendada**

### **Opção 1: Notebook Pré-configurado (Mais Fácil)**

1. **Abrir o notebook principal**:
   ```bash
   # Clique no badge para abrir diretamente no Colab:
   ```
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kiwiabacaxi/img-sinth/blob/main/notebooks/00_Setup_Environment.ipynb)

2. **Executar células de setup**:
   - Execute a primeira célula para clonar o repositório
   - Execute a segunda célula para instalar dependências
   - Aguarde ~5-10 minutos para conclusão

3. **Verificar instalação**:
   ```python
   # Esta célula deve mostrar ✅ para todos os componentes
   from src.utils.system_check import verify_installation
   verify_installation()
   ```

### **Opção 2: Setup Manual no Colab**

```python
# 1. Clonar repositório
!git clone https://github.com/Kiwiabacaxi/img-sinth.git
%cd img-sinth

# 2. Setup automático
!python setup_colab.py

# 3. Verificar GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

# 4. Teste rápido
from src.diffusion.pipeline_manager import PipelineManager
pipeline = PipelineManager()
print("✅ Instalação concluída com sucesso!")
```

---

## 💻 **Instalação Local (Linux/Windows/macOS)**

### **Pré-requisitos**
- Python 3.8 ou superior
- Git
- CUDA 11.8+ (para GPU NVIDIA)
- 16GB+ RAM recomendado
- 50GB+ espaço livre

### **Passo 1: Clonar Repositório**
```bash
git clone https://github.com/Kiwiabacaxi/img-sinth.git
cd img-sinth
```

### **Passo 2: Criar Ambiente Virtual**

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### **Passo 3: Instalar Dependências**

**Instalação Completa (Recomendada):**
```bash
pip install -r requirements.txt
```

**Instalação Mínima (Apenas Inferência):**
```bash
pip install -r requirements-minimal.txt
```

**Instalação de Desenvolvimento:**
```bash
pip install -r requirements-dev.txt
```

### **Passo 4: Configurar CUDA (GPU)**

**Para NVIDIA GPUs:**
```bash
# Verificar versão CUDA
nvidia-smi

# Instalar PyTorch com CUDA (ajuste a versão conforme necessário)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Para AMD GPUs (ROCm):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

### **Passo 5: Verificar Instalação**
```python
python -c "
from src.utils.system_check import verify_installation
verify_installation()
"
```

---

## 🐳 **Instalação com Docker**

### **Opção 1: Imagem Pré-construída**
```bash
# Baixar e executar
docker pull ghcr.io/kiwiabacaxi/img-sinth:latest
docker run --gpus all -p 8888:8888 ghcr.io/kiwiabacaxi/img-sinth:latest
```

### **Opção 2: Build Local**
```bash
# Clonar repositório
git clone https://github.com/Kiwiabacaxi/img-sinth.git
cd img-sinth

# Build da imagem
docker build -t pasture-synthesis .

# Executar com GPU
docker run --gpus all -v $(pwd):/workspace -p 8888:8888 pasture-synthesis
```

### **Docker Compose para Desenvolvimento**
```yaml
# docker-compose.yml
version: '3.8'
services:
  pasture-synthesis:
    build: .
    volumes:
      - ./:/workspace
      - ./data:/data
      - ./outputs:/outputs
    ports:
      - "8888:8888"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

```bash
docker-compose up -d
```

---

## ☁️ **Instalação em Cloud (AWS/GCP/Azure)**

### **AWS EC2 com Deep Learning AMI**

1. **Criar instância EC2**:
   - AMI: Deep Learning AMI (Ubuntu)
   - Tipo: p3.2xlarge ou superior
   - Storage: 100GB+ EBS

2. **Conectar e instalar**:
   ```bash
   ssh -i sua-chave.pem ubuntu@seu-ip-ec2
   
   # Ativar ambiente conda
   conda activate pytorch_p39
   
   # Clonar e instalar
   git clone https://github.com/Kiwiabacaxi/img-sinth.git
   cd img-sinth
   pip install -r requirements.txt
   ```

### **Google Cloud Platform (Compute Engine)**

1. **Criar VM com GPU**:
   ```bash
   gcloud compute instances create pasture-synthesis-vm \
     --zone=us-central1-a \
     --machine-type=n1-standard-4 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=pytorch-latest-gpu \
     --image-project=deeplearning-platform-release \
     --boot-disk-size=100GB \
     --maintenance-policy=TERMINATE
   ```

2. **SSH e setup**:
   ```bash
   gcloud compute ssh pasture-synthesis-vm
   
   # Setup do projeto
   git clone https://github.com/Kiwiabacaxi/img-sinth.git
   cd img-sinth
   pip install -r requirements.txt
   ```

### **Azure Machine Learning**

1. **Criar Compute Instance**:
   - Tipo: Standard_NC6s_v3
   - OS: Ubuntu 18.04

2. **Setup via terminal**:
   ```bash
   git clone https://github.com/Kiwiabacaxi/img-sinth.git
   cd img-sinth
   pip install -r requirements.txt
   ```

---

## 🔧 **Configuração Avançada**

### **Configurar Modelos Base**
```python
# Download automático dos modelos (primeira execução)
from src.diffusion.pipeline_manager import PipelineManager

pipeline = PipelineManager(
    model_name='stabilityai/stable-diffusion-xl-base-1.0',
    cache_dir='/seu/caminho/para/cache'  # Opcional: definir localização
)
```

### **Configurar Datasets de Referência**
```bash
# Criar diretório para datasets de referência
mkdir -p ./assets/reference_images

# Download de datasets públicos (opcional)
python scripts/download_reference_datasets.py
```

### **Configurar Variáveis de Ambiente**
```bash
# .env file
HF_TOKEN=seu_huggingface_token
WANDB_API_KEY=seu_wandb_key  # Para logging
CUDA_VISIBLE_DEVICES=0,1     # GPUs a usar
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Gestão memória
```

### **Configuração de Memória (GPUs Limitadas)**
```python
# config/memory_config.yaml
memory_optimization:
  enable_attention_slicing: true
  enable_cpu_offload: true
  enable_sequential_cpu_offload: false
  enable_model_cpu_offload: true
  enable_xformers: true
  gradient_checkpointing: true
```

---

## ✅ **Verificação de Instalação**

### **Teste Básico de Funcionamento**
```python
# test_installation.py
import sys
import torch
from src.diffusion.pipeline_manager import PipelineManager
from src.dataset.generator import DatasetGenerator
from src.training.yolo_trainer import YOLOTrainer

print("🔍 Testando instalação...")

# 1. Verificar Python e pacotes
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# 2. Testar componentes principais
try:
    pipeline = PipelineManager()
    print("✅ Pipeline Manager")
except Exception as e:
    print(f"❌ Pipeline Manager: {e}")

try:
    generator = DatasetGenerator()
    print("✅ Dataset Generator")
except Exception as e:
    print(f"❌ Dataset Generator: {e}")

try:
    trainer = YOLOTrainer()
    print("✅ YOLO Trainer")
except Exception as e:
    print(f"❌ YOLO Trainer: {e}")

print("🎉 Teste de instalação concluído!")
```

### **Teste de Geração Rápida**
```python
# Teste rápido de geração (5 minutos)
from src.pipeline.quick_test import run_quick_test

results = run_quick_test(
    num_test_images=3,
    resolution=(512, 512),
    steps=10
)

print(f"✅ Teste concluído: {results}")
```

---

## 🐛 **Solução de Problemas Comuns**

### **Erro: "CUDA out of memory"**
```python
# Soluções ordenadas por efetividade:

# 1. Reduzir batch size
config.batch_size = 1

# 2. Ativar CPU offload
config.enable_cpu_offload = True

# 3. Reduzir resolução
config.resolution = (512, 512)

# 4. Usar attention slicing
config.enable_attention_slicing = True
```

### **Erro: "ModuleNotFoundError"**
```bash
# Reinstalar com dependências completas
pip install --upgrade --force-reinstall -r requirements.txt

# Verificar instalação pip
python -m pip install --upgrade pip

# Limpar cache
pip cache purge
```

### **Erro: "Hugging Face Model Download Failed"**
```bash
# Configurar token Hugging Face
huggingface-cli login

# Download manual (se necessário)
python -c "
from diffusers import DiffusionPipeline
pipeline = DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0')
"
```

### **Performance Lenta**
```python
# Otimizações de performance
import torch
torch.backends.cudnn.benchmark = True  # Para inputs de tamanho fixo
torch.backends.cuda.matmul.allow_tf32 = True  # Para A100/H100

# Usar compilação de modelo (PyTorch 2.0+)
pipeline.unet = torch.compile(pipeline.unet)
```

---

## 📊 **Benchmark de Instalação**

Execute este benchmark para verificar a performance da sua instalação:

```python
from src.utils.benchmark import run_installation_benchmark

results = run_installation_benchmark()
print(f"""
📊 BENCHMARK DE INSTALAÇÃO
=========================
GPU: {results['gpu_name']}
Memória GPU: {results['gpu_memory_gb']:.1f}GB
Tempo de geração (512x512): {results['generation_time_512']:.2f}s
Tempo de geração (1024x1024): {results['generation_time_1024']:.2f}s
Throughput: {results['images_per_hour']:.0f} imagens/hora
Score geral: {results['overall_score']}/10
""")
```

---

## 🔄 **Atualizações**

### **Atualizar para Nova Versão**
```bash
# Backup de configurações personalizadas
cp -r configs/custom configs_backup

# Atualizar código
git pull origin main

# Atualizar dependências
pip install --upgrade -r requirements.txt

# Restaurar configurações personalizadas
cp -r configs_backup/* configs/custom/
```

### **Verificar Versão**
```python
from src import __version__
print(f"Versão atual: {__version__}")
```

---

## 📞 **Suporte**

Se você ainda tiver problemas após seguir este guia:

1. **Documentação**: Consulte [Troubleshooting](troubleshooting.md)
2. **Issues**: Abra uma issue no [GitHub](https://github.com/Kiwiabacaxi/img-sinth/issues)
3. **Discord**: Entre no [servidor da comunidade](https://discord.gg/pastagens-ia)
4. **Email**: pastagens.ia@projeto.br

---

<div align="center">

**✅ Instalação concluída com sucesso? [Próximo: Quick Start Guide →](quick-start.md)**

</div>
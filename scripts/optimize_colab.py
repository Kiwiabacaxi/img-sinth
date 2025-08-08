#!/usr/bin/env python3
"""
Script de otimização para Google Colab
Detecta configuração do Colab e aplica otimizações apropriadas
"""

import os
import sys
import json
import psutil
import subprocess
from pathlib import Path
import logging

# Setup básico de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def detect_colab_environment():
    """
    Detecta se está rodando no Google Colab e qual tipo
    
    Returns:
        Dict com informações do ambiente
    """
    
    environment = {
        'is_colab': False,
        'colab_type': None,
        'gpu_available': False,
        'gpu_type': None,
        'gpu_memory_gb': 0,
        'ram_gb': 0,
        'disk_free_gb': 0
    }
    
    # Detectar Google Colab
    try:
        import google.colab
        environment['is_colab'] = True
        logger.info("✅ Google Colab detectado")
    except ImportError:
        logger.info("ℹ️ Não está rodando no Google Colab")
        return environment
    
    # Detectar tipo de GPU
    try:
        import torch
        if torch.cuda.is_available():
            environment['gpu_available'] = True
            environment['gpu_type'] = torch.cuda.get_device_name(0)
            environment['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Classificar tipo de Colab baseado na GPU
            gpu_name = environment['gpu_type'].lower()
            if 'a100' in gpu_name or 'v100' in gpu_name:
                environment['colab_type'] = 'pro_plus'
            elif 'p100' in gpu_name or 'v100' in gpu_name:
                environment['colab_type'] = 'pro'
            elif 't4' in gpu_name:
                environment['colab_type'] = 'free_or_pro'
            else:
                environment['colab_type'] = 'unknown'
                
            logger.info(f"🖥️ GPU: {environment['gpu_type']} ({environment['gpu_memory_gb']:.1f}GB)")
        else:
            logger.warning("⚠️ GPU não disponível")
    except ImportError:
        logger.warning("⚠️ PyTorch não instalado")
    
    # Informações de sistema
    environment['ram_gb'] = psutil.virtual_memory().total / (1024**3)
    environment['disk_free_gb'] = psutil.disk_usage('.').free / (1024**3)
    
    logger.info(f"💾 RAM: {environment['ram_gb']:.1f}GB")
    logger.info(f"💽 Disco livre: {environment['disk_free_gb']:.1f}GB")
    
    return environment

def optimize_memory_settings():
    """Aplica otimizações de memória para Colab"""
    
    logger.info("🔧 Aplicando otimizações de memória...")
    
    # Configurações de ambiente para PyTorch
    optimizations = {
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,garbage_collection_threshold:0.6',
        'CUDA_LAUNCH_BLOCKING': '0',  # Async para melhor performance
        'TOKENIZERS_PARALLELISM': 'false',  # Evitar warnings
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
        logger.info(f"   {key}={value}")
    
    # Configurações Python
    try:
        import gc
        import torch
        
        # Garbage collection mais agressivo
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Configurações CUDA
            torch.backends.cudnn.benchmark = True  # Otimizar para inputs de tamanho fixo
            torch.backends.cuda.matmul.allow_tf32 = True  # TF32 para A100/H100
            
            logger.info("✅ Otimizações CUDA aplicadas")
        
    except ImportError:
        logger.warning("⚠️ PyTorch não disponível para otimizações")

def install_system_dependencies():
    """Instala dependências do sistema necessárias"""
    
    logger.info("📦 Verificando dependências do sistema...")
    
    # Dependências APT necessárias
    apt_packages = [
        'ffmpeg',           # Para processamento de vídeo se necessário
        'libsm6',           # Para OpenCV
        'libxext6',         # Para OpenCV
        'libxrender-dev',   # Para matplotlib
        'libglib2.0-0',     # Para OpenCV
        'libgl1-mesa-glx',  # Para OpenCV
    ]
    
    try:
        # Verificar se precisa instalar
        missing_packages = []
        for package in apt_packages:
            result = subprocess.run(['dpkg', '-l', package], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                missing_packages.append(package)
        
        if missing_packages:
            logger.info(f"Instalando pacotes: {missing_packages}")
            subprocess.run(['apt-get', 'update', '-qq'], check=True)
            subprocess.run(['apt-get', 'install', '-y', '-qq'] + missing_packages, 
                         check=True)
            logger.info("✅ Dependências do sistema instaladas")
        else:
            logger.info("✅ Dependências do sistema já instaladas")
            
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Erro instalando dependências: {e}")
    except FileNotFoundError:
        logger.warning("⚠️ Sistema APT não disponível")

def optimize_pip_settings():
    """Otimiza configurações do pip para Colab"""
    
    logger.info("🐍 Otimizando configurações pip...")
    
    # Configurações pip otimizadas
    pip_config = """
[global]
cache-dir = /tmp/pip-cache
timeout = 60
retries = 3
trusted-host = pypi.org
                pypi.python.org
                files.pythonhosted.org

[install]
compile = false
no-warn-script-location = true
"""
    
    # Criar diretório de configuração
    pip_config_dir = Path.home() / '.config' / 'pip'
    pip_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar configuração
    config_file = pip_config_dir / 'pip.conf'
    with open(config_file, 'w') as f:
        f.write(pip_config)
    
    # Criar cache directory
    cache_dir = Path('/tmp/pip-cache')
    cache_dir.mkdir(exist_ok=True)
    
    logger.info("✅ Configurações pip otimizadas")

def setup_huggingface_cache():
    """Configura cache otimizado para Hugging Face"""
    
    logger.info("🤗 Configurando cache Hugging Face...")
    
    # Usar /tmp para cache (mais espaço no Colab)
    cache_dir = Path('/tmp/hf_cache')
    cache_dir.mkdir(exist_ok=True)
    
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / 'transformers')
    os.environ['HF_DATASETS_CACHE'] = str(cache_dir / 'datasets')
    
    logger.info(f"✅ Cache HF configurado: {cache_dir}")

def apply_colab_specific_optimizations(environment):
    """Aplica otimizações específicas baseadas no tipo de Colab"""
    
    colab_type = environment.get('colab_type', 'unknown')
    gpu_memory = environment.get('gpu_memory_gb', 0)
    
    logger.info(f"🎯 Aplicando otimizações para: {colab_type}")
    
    # Configurações por tipo
    if colab_type == 'pro_plus':
        # Colab Pro+ com A100
        config = {
            'max_batch_size': 32,
            'use_attention_slicing': False,
            'use_cpu_offload': False,
            'enable_xformers': True,
            'mixed_precision': True
        }
    elif colab_type == 'pro':
        # Colab Pro com V100/P100
        config = {
            'max_batch_size': 16,
            'use_attention_slicing': False,
            'use_cpu_offload': False,
            'enable_xformers': True,
            'mixed_precision': True
        }
    elif gpu_memory >= 15:  # T4 com boa memória
        config = {
            'max_batch_size': 8,
            'use_attention_slicing': True,
            'use_cpu_offload': False,
            'enable_xformers': True,
            'mixed_precision': True
        }
    elif gpu_memory >= 8:   # T4 básico
        config = {
            'max_batch_size': 4,
            'use_attention_slicing': True,
            'use_cpu_offload': True,
            'enable_xformers': True,
            'mixed_precision': True
        }
    else:  # GPU pequena ou CPU
        config = {
            'max_batch_size': 2,
            'use_attention_slicing': True,
            'use_cpu_offload': True,
            'enable_xformers': False,
            'mixed_precision': False
        }
    
    # Salvar configuração otimizada
    config_path = Path('/tmp/colab_optimization_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("✅ Configuração otimizada salva")
    
    # Log das configurações aplicadas
    for key, value in config.items():
        logger.info(f"   {key}: {value}")
    
    return config

def setup_monitoring():
    """Configura monitoramento de recursos"""
    
    logger.info("📊 Configurando monitoramento...")
    
    monitoring_script = """
import psutil
import GPUtil
import time

def monitor_resources():
    '''Monitor de recursos em tempo real'''
    try:
        # CPU e RAM
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        print(f"CPU: {cpu_percent}% | RAM: {memory.percent}% ({memory.available/1e9:.1f}GB livre)")
        
        # GPU se disponível
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                print(f"GPU: {gpu.load*100:.1f}% | VRAM: {gpu.memoryUtil*100:.1f}% ({gpu.memoryFree}MB livre)")
        
    except Exception as e:
        print(f"Erro no monitoramento: {e}")

# Função para usar nos notebooks
def check_resources():
    monitor_resources()
"""
    
    # Salvar script de monitoramento
    monitor_file = Path('/tmp/resource_monitor.py')
    with open(monitor_file, 'w') as f:
        f.write(monitoring_script)
    
    logger.info(f"✅ Monitor salvo: {monitor_file}")

def cleanup_colab():
    """Limpeza de espaço no Colab"""
    
    logger.info("🧹 Limpando espaço no Colab...")
    
    # Diretórios para limpar
    cleanup_paths = [
        '/tmp/*',
        '/content/sample_data',
        '/root/.cache/pip',
        '/var/log/*'
    ]
    
    freed_space = 0
    
    for path_pattern in cleanup_paths:
        try:
            if '*' in path_pattern:
                # Usar shell para wildcards
                result = subprocess.run(f'du -sb {path_pattern} 2>/dev/null || true', 
                                      shell=True, capture_output=True, text=True)
                if result.stdout:
                    size = sum(int(line.split()[0]) for line in result.stdout.strip().split('\n') if line)
                    freed_space += size
                
                subprocess.run(f'rm -rf {path_pattern}', shell=True)
            else:
                path = Path(path_pattern)
                if path.exists():
                    if path.is_file():
                        freed_space += path.stat().st_size
                        path.unlink()
                    else:
                        # Calcular tamanho do diretório
                        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        freed_space += total_size
                        subprocess.run(['rm', '-rf', str(path)])
        except Exception as e:
            logger.warning(f"Erro limpando {path_pattern}: {e}")
    
    freed_gb = freed_space / (1024**3)
    logger.info(f"✅ Espaço liberado: {freed_gb:.1f}GB")

def main():
    """Função principal de otimização"""
    
    print("🌱 Otimizador do Brazilian Pasture Synthesis para Google Colab")
    print("=" * 60)
    
    # 1. Detectar ambiente
    environment = detect_colab_environment()
    
    if not environment['is_colab']:
        logger.info("ℹ️ Não está no Colab. Aplicando otimizações gerais...")
    
    # 2. Limpeza inicial
    cleanup_colab()
    
    # 3. Instalar dependências do sistema
    install_system_dependencies()
    
    # 4. Otimizar pip
    optimize_pip_settings()
    
    # 5. Configurar caches
    setup_huggingface_cache()
    
    # 6. Otimizações de memória
    optimize_memory_settings()
    
    # 7. Aplicar otimizações específicas
    config = apply_colab_specific_optimizations(environment)
    
    # 8. Configurar monitoramento
    setup_monitoring()
    
    print("\n✅ OTIMIZAÇÃO CONCLUÍDA!")
    print("=" * 30)
    
    # Resumo das otimizações aplicadas
    print(f"🖥️ Ambiente: {environment.get('colab_type', 'local')}")
    if environment['gpu_available']:
        print(f"🎮 GPU: {environment['gpu_type']}")
    print(f"📦 Batch size recomendado: {config['max_batch_size']}")
    print(f"💾 Attention slicing: {config['use_attention_slicing']}")
    print(f"⚡ XFormers: {config['enable_xformers']}")
    
    # Instruções para uso
    print("\n📋 PRÓXIMOS PASSOS:")
    print("1. Execute: !python setup_colab.py")
    print("2. Use as configurações otimizadas nos notebooks")
    print("3. Monitore recursos com: exec(open('/tmp/resource_monitor.py').read())")
    
    return environment, config

if __name__ == "__main__":
    main()
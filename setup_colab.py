"""
Setup automático para Google Colab
Configura ambiente completo para geração de imagens sintéticas de pastagens brasileiras
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def check_gpu():
    """Verifica disponibilidade e tipo de GPU"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU disponível: {gpu_name}")
        print(f"✅ Memória GPU: {gpu_memory:.1f} GB")
        return True
    else:
        print("❌ GPU não disponível - verificar runtime do Colab")
        return False

def install_dependencies():
    """Instala dependências com otimizações para Colab"""
    print("🔧 Instalando dependências...")
    
    # Instalar requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                   check=True, capture_output=True)
    
    # Otimizações específicas do Colab
    try:
        # Instalar xformers com versão específica para Colab
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "xformers==0.0.22.post7", "--index-url", 
                       "https://download.pytorch.org/whl/cu118"], 
                       check=True, capture_output=True)
        print("✅ xformers instalado com sucesso")
    except:
        print("⚠️  xformers não instalado - performance pode ser reduzida")

def setup_directories():
    """Cria e verifica estrutura de diretórios"""
    print("📁 Configurando diretórios...")
    
    # Diretórios essenciais para Colab
    colab_dirs = [
        '/content/model_cache',
        '/content/generated_cache', 
        '/content/outputs',
        '/content/datasets',
        '/content/temp'
    ]
    
    for dir_path in colab_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ Diretórios configurados")

def download_models():
    """Pre-download de modelos essenciais para cache"""
    print("📥 Fazendo download de modelos base...")
    
    from diffusers import StableDiffusionPipeline, ControlNetModel
    from transformers import CLIPTextModel
    
    try:
        # Cache do modelo principal
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir="/content/model_cache",
            torch_dtype=torch.float16
        )
        print("✅ Stable Diffusion XL baixado")
        
        # Cache ControlNet models  
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            cache_dir="/content/model_cache",
            torch_dtype=torch.float16
        )
        print("✅ ControlNet Canny baixado")
        
    except Exception as e:
        print(f"⚠️  Erro no download de modelos: {e}")

def setup_environment():
    """Configura variáveis de ambiente otimizadas"""
    print("⚙️  Configurando ambiente...")
    
    # Otimizações de memória
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Cache directories
    os.environ['HF_HOME'] = '/content/model_cache'
    os.environ['TRANSFORMERS_CACHE'] = '/content/model_cache'
    os.environ['DIFFUSERS_CACHE'] = '/content/model_cache'
    
    print("✅ Ambiente configurado")

def verify_installation():
    """Verifica se instalação está correta"""
    print("🧪 Verificando instalação...")
    
    try:
        import diffusers
        import transformers
        import ultralytics
        import controlnet_aux
        print("✅ Importações principais OK")
        
        # Teste rápido de GPU
        if torch.cuda.is_available():
            test_tensor = torch.randn(100, 100).cuda()
            print("✅ Teste de GPU OK")
            
        return True
    except Exception as e:
        print(f"❌ Erro na verificação: {e}")
        return False

def main():
    """Função principal de setup"""
    print("🌱 Configurando Sistema de Geração de Pastagens Brasileiras")
    print("=" * 60)
    
    # Verificações
    if not check_gpu():
        print("⚠️  Continuando sem GPU (apenas CPU)")
    
    # Setup
    try:
        install_dependencies()
        setup_directories() 
        setup_environment()
        
        # Download de modelos (opcional - pode ser demorado)
        download_choice = input("Fazer download de modelos agora? (s/N): ")
        if download_choice.lower() in ['s', 'sim', 'y', 'yes']:
            download_models()
        
        # Verificação final
        if verify_installation():
            print("\n✅ Setup completo! Sistema pronto para uso.")
            print("📚 Próximo passo: executar notebook 00_Setup_Environment.ipynb")
        else:
            print("\n❌ Setup incompleto - verificar erros acima")
            
    except Exception as e:
        print(f"\n❌ Erro durante setup: {e}")
        print("Tente executar novamente ou verificar dependências")

if __name__ == "__main__":
    main()
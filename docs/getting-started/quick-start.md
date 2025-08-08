# 🏃‍♂️ Guia de Início Rápido - 30 Minutos para Seu Primeiro Modelo

Este guia te levará do zero a um modelo funcional de detecção de plantas invasoras em apenas 30 minutos usando Google Colab.

---

## 🎯 **O que você vai conseguir**

Ao final deste guia, você terá:
- ✅ **500 imagens sintéticas** de pastagens brasileiras
- ✅ **Modelo YOLOv8s treinado** para detecção de invasoras
- ✅ **Avaliação de performance** com métricas científicas
- ✅ **Visualizações interativas** dos resultados

**Tempo estimado**: 30-45 minutos (dependendo da GPU do Colab)

---

## 🚀 **Passo 1: Setup Inicial (5 minutos)**

### **1.1 Abrir o Notebook**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kiwiabacaxi/img-sinth/blob/main/notebooks/Quick_Start.ipynb)

### **1.2 Verificar GPU**
```python
# Primeira célula - sempre execute esta
import torch
print("🔍 Verificando ambiente...")
print(f"✅ CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🖥️ GPU: {torch.cuda.get_device_name()}")
    print(f"💾 Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    print("⚠️ GPU não detectada. Vá em Runtime > Change runtime type > GPU")
```

### **1.3 Instalar Dependências**
```python
# Segunda célula - aguarde ~3-5 minutos
!git clone https://github.com/Kiwiabacaxi/img-sinth.git
%cd img-sinth
!python setup_colab.py

print("🎉 Setup concluído!")
```

---

## 🎨 **Passo 2: Gerar Imagens Sintéticas (10 minutos)**

### **2.1 Configuração Rápida**
```python
# Configure seu primeiro dataset
from src.pipeline.main_pipeline import BrazilianPasturePipeline

# Inicializar pipeline
pipeline = BrazilianPasturePipeline(
    output_dir='/content/meu_primeiro_dataset',
    device='auto'  # Detecta GPU automaticamente
)

print("✅ Pipeline inicializado!")
```

### **2.2 Gerar Dataset de Teste**
```python
# Gerar 500 imagens para teste (10-15 minutos)
dataset_path = pipeline.generate_quick_dataset(
    num_images=500,           # Quantidade para teste rápido
    biome='cerrado',          # Focar no Cerrado primeiro
    resolution=(768, 768),    # Resolução boa para treinamento
    quality_threshold=0.7,    # Manter apenas imagens boas
    save_metadata=True        # Salvar metadados para análise
)

print(f"🎉 Dataset gerado: {dataset_path}")
```

**💡 Dica**: Enquanto gera, você pode monitorar o progresso. Se der erro de memória, reinicie o runtime e reduza `num_images` para 300.

### **2.3 Visualizar Resultados**
```python
# Visualizar algumas imagens geradas
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random

def show_generated_samples(dataset_path, n_samples=6):
    images_dir = Path(dataset_path) / 'images'
    image_files = list(images_dir.glob('*.jpg'))
    
    # Selecionar amostras aleatórias
    samples = random.sample(image_files, min(n_samples, len(image_files)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, img_path in enumerate(samples):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f'Amostra {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Mostrar resultados
show_generated_samples(dataset_path)
print("🖼️ Suas primeiras imagens sintéticas de pastagens!")
```

---

## 🤖 **Passo 3: Treinar Modelo YOLO (12 minutos)**

### **3.1 Configurar Treinamento**
```python
# Configuração otimizada para início rápido
training_config = {
    'model_size': 'yolov8s',      # Modelo pequeno e rápido
    'epochs': 50,                 # Poucas épocas para teste
    'batch_size': 8,              # Ajustado para GPUs menores
    'patience': 15,               # Parar cedo se não melhorar
    'image_size': 640,            # Tamanho padrão YOLO
}

print("⚙️ Configuração definida para treinamento rápido")
```

### **3.2 Executar Treinamento**
```python
# Treinar modelo (10-15 minutos)
model_path = pipeline.train_detection_model(
    dataset_path=dataset_path,
    **training_config,
    project_name='meu_primeiro_modelo'
)

if model_path:
    print(f"🎉 Modelo treinado salvo em: {model_path}")
    print("📊 Aguarde... gerando relatório de performance...")
else:
    print("❌ Erro no treinamento. Verifique os logs acima.")
```

**💡 Dica**: Se o treinamento estiver muito lento, reduza `epochs` para 30 ou `batch_size` para 4.

---

## 📊 **Passo 4: Avaliar Performance (5 minutos)**

### **4.1 Métricas de Performance**
```python
# Carregar resultados do treinamento
from src.training.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(
    model_path=model_path,
    dataset_path=dataset_path
)

# Mostrar métricas principais
print("📈 RESULTADOS DO SEU MODELO:")
print("=" * 40)
print(f"mAP@0.5:     {results.map50:.3f}")
print(f"Precisão:    {results.precision:.3f}")
print(f"Recall:      {results.recall:.3f}")
print(f"F1-Score:    {results.f1_score:.3f}")
print("=" * 40)

# Interpretar resultados
if results.map50 >= 0.7:
    print("🎉 Excelente! Seu modelo está funcionando bem!")
elif results.map50 >= 0.5:
    print("👍 Bom resultado para um primeiro teste!")
elif results.map50 >= 0.3:
    print("📈 Resultado moderado. Tente mais épocas ou mais dados.")
else:
    print("🔄 Resultado baixo. Vamos ajustar a configuração.")
```

### **4.2 Visualizar Detecções**
```python
# Testar modelo em imagens novas
from src.utils.visualization import visualize_detections

# Gerar algumas imagens para teste
test_images = pipeline.generate_test_images(
    num_images=5,
    biome='cerrado',
    save_path='/content/test_images'
)

# Fazer predições e visualizar
visualize_detections(
    model_path=model_path,
    image_paths=test_images,
    confidence_threshold=0.5,
    save_path='/content/detections.png'
)

print("🔍 Visualizações salvas! Veja como seu modelo está detectando invasoras.")
```

---

## 🎊 **Parabéns! Você completou seu primeiro modelo!**

### **📋 Resumo do que você conseguiu:**

✅ **Gerou 500 imagens sintéticas** de pastagens do Cerrado  
✅ **Treinou um modelo YOLOv8s** em ~12 minutos  
✅ **Alcançou mAP@0.5 > 0.X** (insira seu resultado)  
✅ **Visualizou detecções** em imagens de teste  

### **🎯 Próximos Passos Recomendados:**

1. **📈 Melhorar Performance**:
   ```python
   # Experimente estas configurações para melhorar:
   - Aumentar epochs para 100-150
   - Usar modelo maior (yolov8m)
   - Gerar mais imagens (2000-5000)
   - Adicionar mais biomas
   ```

2. **🌍 Expandir Dataset**:
   ```python
   # Adicionar mais diversidade
   pipeline.generate_multi_biome_dataset(
       biomes=['cerrado', 'mata_atlantica', 'pampa'],
       num_images_per_biome=1000,
       seasons=['seca', 'chuvas', 'transicao']
   )
   ```

3. **🔬 Análise Científica**:
   ```python
   # Comparar com benchmarks científicos
   from src.training.benchmark import ScientificBenchmarkSuite
   
   benchmark_suite = ScientificBenchmarkSuite()
   comparison = benchmark_suite.compare_with_benchmarks({
       'map50': results.map50,
       'precision': results.precision,
       'recall': results.recall
   })
   ```

---

## 🔧 **Troubleshooting Rápido**

### **Erro: "CUDA out of memory"**
```python
# Soluções rápidas (teste uma por vez):
1. Reiniciar runtime: Runtime > Restart runtime
2. Reduzir batch_size para 4 ou 2
3. Usar modelo menor: 'yolov8n'
4. Reduzir resolução para (512, 512)
```

### **Geração muito lenta**
```python
# Acelerar geração:
1. Reduzir num_inference_steps para 15
2. Desativar ControlNet temporariamente
3. Usar resolução menor (512, 512)
4. Reduzir quality_threshold para 0.6
```

### **Performance baixa do modelo**
```python
# Melhorar performance:
1. Aumentar num_images para 1000+
2. Aumentar epochs para 100+
3. Verificar se classes estão balanceadas
4. Usar augmentação de dados mais agressiva
```

---

## 📚 **Próximos Tutoriais**

Agora que você tem a base funcionando, explore:

1. **[📊 Controle de Qualidade](../guides/quality-control.md)** - Melhorar qualidade das imagens
2. **[🌍 Multi-Bioma](../examples/multi-biome.md)** - Trabalhar com múltiplos biomas
3. **[🔬 Setup Científico](../examples/research-setup.md)** - Configuração para pesquisa
4. **[⚡ Otimização](../guides/optimization.md)** - Acelerar treinamento e inferência

---

## 🎥 **Video Tutorial**

Prefere seguir um vídeo? Assista nosso tutorial passo-a-passo:

[![Tutorial Quick Start](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

---

## 💬 **Compartilhe Seus Resultados!**

Conseguiu completar o tutorial? Compartilhe seus resultados:

- 🐦 **Twitter**: [@PastagensBr_IA](https://twitter.com/PastagensBr_IA) com #MeuPrimeiroModelo
- 💬 **Discord**: [Canal #sucessos](https://discord.gg/pastagens-ia)
- 📧 **Email**: seus-resultados@projeto.br

**Template para compartilhar**:
```
🎉 Completei meu primeiro modelo de pastagens!
📊 mAP@0.5: X.XXX
⏱️ Tempo total: XX minutos
🖥️ GPU usada: [sua GPU]
💭 Impressões: [seu feedback]
#PastagensIA #MeuPrimeiroModelo
```

---

<div align="center">

**🎉 Parabéns por completar o Quick Start!**

**Próximo**: [📊 Controle de Qualidade →](../guides/quality-control.md)

---

*Teve dificuldades? [Abra uma issue](https://github.com/Kiwiabacaxi/img-sinth/issues) ou [entre no Discord](https://discord.gg/pastagens-ia)*

</div>
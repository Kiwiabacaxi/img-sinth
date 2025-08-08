# 🌱 Brazilian Pasture Synthetic Image Generator

Sistema completo de geração de imagens sintéticas de pastagens brasileiras usando **Stable Diffusion**, otimizado para treinamento de modelos **YOLOv8/v9** em detecção de plantas invasoras e análise de qualidade de pastagens.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seu-usuario/brazilian-pasture-synthesis)

## 🎯 Objetivos

- **Detecção de plantas invasoras** em pastagens brasileiras
- **Segmentação de qualidade** de pastagens (boa/moderada/degradada) 
- **Classificação de espécies** de gramíneas nativas
- **Performance superior a mAP 0.85** baseado em estudos científicos

## 📚 Fundamentação Científica

Baseado nos estudos:
- **Moreno et al. (2023)**: mAP 0.91 com imagens sintéticas, mAP 0.99 com dados mistos
- **Chen et al. (2025)**: Review abrangente de Stable Diffusion na agricultura
- **Deng et al. (2025)**: ControlNet + YOLOv8 com FID score 0.98

## 🚀 Início Rápido (Google Colab)

### 1. Setup Automático
```bash
# Clone do repositório
!git clone https://github.com/seu-usuario/brazilian-pasture-synthesis.git
%cd brazilian-pasture-synthesis

# Configuração automática
!python setup_colab.py

# Verificação de GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")
```

### 2. Execução do Pipeline
```python
from src.pipeline import BrazilianPasturePipeline

# Configurar pipeline
pipeline = BrazilianPasturePipeline(
    model_name='stabilityai/stable-diffusion-xl-base-1.0',
    use_controlnet=True,
    output_dir='/content/outputs'
)

# Gerar dataset
dataset = pipeline.generate_dataset(
    num_images=5000,
    biomes=['cerrado', 'mata_atlantica'],
    seasons=['seca', 'chuvas'],
    quality_threshold=0.8
)

# Treinar YOLO
model = pipeline.train_yolo(
    dataset_path=dataset,
    model_type='detection',
    epochs=100
)
```

## 📁 Estrutura do Projeto

```
brazilian-pasture-synthesis/
├── notebooks/              # Notebooks executáveis no Colab
│   ├── 00_Setup_Environment.ipynb
│   ├── 01_Explore_Prompts.ipynb
│   ├── 02_Generate_Dataset.ipynb
│   ├── 03_Quality_Control.ipynb
│   ├── 04_YOLO_Training.ipynb
│   └── 05_Evaluation_Benchmark.ipynb
├── src/                    # Código fonte modular
│   ├── diffusion/         # Pipeline Stable Diffusion
│   ├── dataset/           # Geração e formatação de datasets
│   └── training/          # Treinamento YOLO e avaliação
├── configs/               # Configurações YAML
├── assets/                # Recursos de referência
├── outputs/               # Outputs gerados
└── docs/                  # Documentação completa
```

## 🇧🇷 Especificações Brasileiras

### Biomas Suportados
- **Cerrado**: Brachiaria brizantha, solo latossolo, 0-25° inclinação
- **Mata Atlântica**: Brachiaria decumbens, argissolo/cambissolo, 5-45°
- **Pampa**: Gramíneas nativas, planossolos, relevo suave

### Plantas Invasoras Detectadas
- **Capim-gordura** (Melinis minutiflora): 5-30% cobertura
- **Carqueja** (Baccharis trimera): 2-20% cobertura  
- **Samambaia** (Pteridium aquilinum): 10-25% cobertura

### Condições Sazonais
- **Estação Seca** (Mai-Set): Gramíneas douradas, solo exposto 30-70%
- **Estação Chuvosa** (Out-Abr): Gramíneas verdes, cobertura 85-100%

## 🛠️ Stack Tecnológica

### Principais Dependências
```
diffusers==0.24.0          # Stable Diffusion
ultralytics==8.0.206       # YOLOv8/v9
controlnet-aux==0.4.0       # ControlNet
transformers==4.36.0        # Text encoders
```

### Requisitos de Sistema
- **Google Colab** (GPU T4/V100 recomendado)
- **Python 3.8+**
- **PyTorch 2.0+** com CUDA
- **Memória GPU**: Mínimo 8GB, ideal 16GB+

## 📊 Performance Esperada

| Métrica | Target | Baseline (Estudos) |
|---------|--------|-------------------|
| mAP@0.5 | ≥0.85 | 0.91 (Moreno et al.) |
| mAP@0.5:0.95 | ≥0.75 | 0.78 (Deng et al.) |
| FID Score | ≤1.0 | 0.98 (Deng et al.) |
| Inference Time | ≤15ms | 10.2ms (YOLOv8l) |

## 📚 Documentação

- **[Setup Guide](docs/setup.md)**: Configuração detalhada
- **[API Reference](docs/api.md)**: Documentação da API
- **[Prompt Engineering](docs/prompts.md)**: Guia de prompts
- **[Training Guide](docs/training.md)**: Treinamento YOLO
- **[Troubleshooting](docs/troubleshooting.md)**: Solução de problemas

## 🏆 Resultados Esperados

### Dataset Sintético
- ✅ 50.000+ imagens (1024x1024)
- ✅ Anotações YOLO automáticas
- ✅ Splits organizados (70/20/10)
- ✅ Metadados completos

### Modelos Treinados
- ✅ YOLOv8l detecção (invasoras)
- ✅ YOLOv8l segmentação (qualidade)
- ✅ Performance ≥85% dos benchmarks
- ✅ Otimização para edge deployment

## 🤝 Contribuição

1. Fork o repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📝 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

## 📞 Suporte

- **Issues**: [GitHub Issues](https://github.com/seu-usuario/brazilian-pasture-synthesis/issues)
- **Discussões**: [GitHub Discussions](https://github.com/seu-usuario/brazilian-pasture-synthesis/discussions)
- **Email**: [seu-email@exemplo.com](mailto:seu-email@exemplo.com)

## 🙏 Agradecimentos

Baseado nas pesquisas de:
- Moreno et al. (2023) - Synthetic image generation for weed detection
- Chen et al. (2025) - Stable Diffusion in agriculture review
- Deng et al. (2025) - ControlNet for agricultural applications

---

**🌱 Desenvolvido para agricultura de precisão brasileira - Transformando pastagens com IA**
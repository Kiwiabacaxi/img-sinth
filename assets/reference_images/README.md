# 📸 Imagens de Referência - Pastagens Brasileiras

Este diretório contém imagens de referência para auxiliar no desenvolvimento e validação do sistema de geração sintética.

## 📁 Estrutura dos Diretórios

```
reference_images/
├── biomes/                 # Imagens por bioma
│   ├── cerrado/
│   ├── mata_atlantica/
│   └── pampa/
├── species/               # Espécies específicas
│   ├── invasive/
│   ├── native/
│   └── indicators/
├── conditions/            # Diferentes condições
│   ├── seasons/
│   ├── degradation/
│   └── management/
└── validation/           # Conjunto para validação
    ├── ground_truth/
    └── expert_annotated/
```

## 🌿 Biomas Brasileiros

### Cerrado
- **Características**: Solo vermelho laterítico, vegetação de savana, estações seca/chuvosa distintas
- **Gramíneas Dominantes**: Brachiaria brizantha, Panicum maximum, Andropogon gayanus
- **Invasoras Típicas**: Melinis minutiflora (capim-gordura), Urochloa decumbens
- **Indicadores de Degradação**: Cupinzeiros, solo exposto, compactação

### Mata Atlântica
- **Características**: Clima tropical úmido, maior diversidade, relevo ondulado
- **Gramíneas Dominantes**: Paspalum notatum, Axonopus compressus
- **Invasoras Típicas**: Megathyrsus maximus, Pteridium aquilinum (samambaia)
- **Indicadores**: Bordas florestais, transição mata-pasto

### Pampa
- **Características**: Campos sulinos, clima subtropical, topografia plana
- **Gramíneas Dominantes**: Andropogon lateralis, Aristida laevis
- **Invasoras Típicas**: Eragrostis plana (capim-annoni), Baccharis trimera (carqueja)
- **Indicadores**: Mosaicos de campo nativo e exóticas

## 🔍 Espécies de Referência

### Invasoras Críticas

#### Melinis minutiflora (Capim-gordura)
- **Habitat**: Cerrado principalmente
- **Identificação Visual**: Folhas peludas, colmos avermelhados, touceiras densas
- **Época de Detecção**: Mais visível na época seca
- **Impacto**: Altera regime de fogo, reduz biodiversidade

#### Megathyrsus maximus (Capim-colonião)
- **Habitat**: Multi-bioma, especialmente Mata Atlântica
- **Identificação Visual**: Porte alto (até 3m), panículas grandes
- **Época de Detecção**: Ano todo, mais evidente no verão
- **Impacto**: Competição com nativas, sombreamento

#### Eragrostis plana (Capim-annoni)
- **Habitat**: Pampa
- **Identificação Visual**: Touceiras claras, panículas abertas
- **Época de Detecção**: Primavera/verão
- **Impacto**: Dominância em campos nativos

### Gramíneas Nativas de Referência

#### Andropogon gayanus (Cerrado)
- **Identificação**: Touceiras altas, folhas estreitas, panículas digitadas
- **Valor**: Forrageira nativa de qualidade

#### Paspalum notatum (Grama-forquilha)
- **Identificação**: Crescimento rasteiro, inflorescências em V
- **Valor**: Cobertura densa, resistente ao pisoteio

#### Andropogon lateralis (Capim-caninha)
- **Identificação**: Touceiras médias, panículas plumosas
- **Valor**: Espécie-chave dos campos sulinos

## 📊 Padrões de Degradação

### Níveis de Degradação

#### Pastagem Saudável (Boa)
- **Cobertura vegetal**: >80%
- **Diversidade**: Alta diversidade de gramíneas
- **Solo**: Sem sinais de erosão
- **Indicadores**: Ausência de invasoras dominantes

#### Pastagem Moderadamente Degradada
- **Cobertura vegetal**: 50-80%
- **Diversidade**: Reduzida, algumas invasoras presentes
- **Solo**: Sinais iniciais de compactação
- **Indicadores**: Patches de solo exposto, cupinzeiros esparsos

#### Pastagem Severamente Degradada
- **Cobertura vegetal**: <50%
- **Diversidade**: Muito baixa, dominância de invasoras
- **Solo**: Erosão evidente, compactação severa
- **Indicadores**: Solo exposto extenso, voçorocas, cupinzeiros abundantes

### Indicadores Visuais de Degradação

#### Cupinzeiros
- **Tipo**: Principalmente Cornitermes e Syntermes
- **Significado**: Indicador de compactação e alteração do solo
- **Detecção**: Estruturas cônicas marrons/cinzas

#### Solo Exposto
- **Causas**: Superpastejo, erosão, compactação
- **Aparência**: Patches sem vegetação, coloração típica do solo local
- **Progressão**: Pode formar voçorocas em encostas

#### Plantas Indicadoras de Degradação
- **Assa-peixe** (Vernonia spp.): Indica solos ácidos e pobres
- **Maria-mole** (Senecio brasiliensis): Solos compactados
- **Capim-barba-de-bode** (Aristida spp.): Solos muito degradados

## 🗓️ Variações Sazonais

### Estação Seca (Maio-Setembro)
- **Características**: Vegetação amarelada/seca, maior visibilidade do solo
- **Detecção**: Invasoras mais evidentes, estrutura da vegetação clara
- **Desafios**: Menor contraste de cores, plantas dormentes

### Estação Chuvosa (Outubro-Abril)
- **Características**: Vegetação verde e densa, crescimento ativo
- **Detecção**: Maior biomassa, cores mais vivas
- **Desafios**: Sobreposição de espécies, maior complexidade visual

### Transição
- **Características**: Mistura de vegetação seca e verde
- **Detecção**: Padrões heterogêneos, brotação desigual
- **Desafios**: Maior variabilidade intra-imagem

## 📏 Especificações Técnicas das Imagens

### Resolução e Qualidade
- **Resolução mínima**: 1024x1024 pixels
- **Formato**: JPEG (qualidade >90%) ou PNG
- **Profundidade de cor**: 24-bit RGB
- **Compressão**: Mínima para preservar detalhes

### Condições de Captura
- **Altura de voo**: 30-100m (drones) ou orbital (satélites)
- **GSD (Ground Sample Distance)**: 5-30cm/pixel
- **Horário**: 10:00-14:00 (melhor iluminação)
- **Condições climáticas**: Céu claro ou parcialmente nublado

### Metadados Necessários
```yaml
metadata:
  location:
    biome: "cerrado|mata_atlantica|pampa"
    coordinates: [lat, lon]
    altitude_m: 500
  
  temporal:
    date: "YYYY-MM-DD"
    season: "seca|chuvas|transicao"
    time: "HH:MM"
  
  technical:
    resolution: [width, height]
    gsd_cm: 15.0
    sensor: "RGB|Multispectral"
    platform: "drone|satellite|aircraft"
  
  ecological:
    degradation_level: "boa|moderada|degradada"
    dominant_species: ["species1", "species2"]
    invasive_presence: boolean
    management_type: "extensivo|intensivo|orgânico"
```

## 🎯 Uso para Validação

### Conjunto de Validação
- **Tamanho**: Mínimo 500 imagens por bioma
- **Anotação**: Bounding boxes para todas as espécies de interesse
- **Verificação**: Validação por especialistas em pastagens
- **Balanceamento**: Distribuição equilibrada por classes e condições

### Métricas de Qualidade
- **Nitidez**: Sem blur excessivo
- **Exposição**: Sem super/sub-exposição
- **Contraste**: Contraste adequado para detecção de features
- **Artefatos**: Sem compressão visível ou ruído excessivo

### Protocolo de Anotação
1. **Identificação de espécies**: Baseada em características botânicas
2. **Delimitação de áreas**: Bounding boxes precisos
3. **Classificação de degradação**: Baseada em critérios objetivos
4. **Validação cruzada**: Múltiplos anotadores para amostras críticas

## 📚 Fontes e Créditos

### Instituições Colaboradoras
- **Embrapa Cerrados**: Imagens de referência do Cerrado
- **Instituto Florestal (SP)**: Mata Atlântica
- **Embrapa Pecuária Sul**: Campos do Pampa
- **INPE**: Imagens de satélite

### Especialistas Consultados
- **Botânicos**: Identificação de espécies
- **Ecólogos**: Padrões de degradação
- **Técnicos em pastagens**: Manejo e indicadores práticos

### Licenciamento
- **Uso acadêmico**: Livre para pesquisa científica
- **Uso comercial**: Consultar licenças específicas
- **Atribuição**: Citar fontes apropriadamente

## 🔧 Como Usar Este Conjunto

### Para Desenvolvimento
1. **Referência visual**: Comparar imagens sintéticas com reais
2. **Calibração de prompts**: Ajustar prompts baseados em features reais
3. **Validação de qualidade**: Usar como ground truth para métricas

### Para Treinamento
1. **Data augmentation**: Inspiração para variações realísticas
2. **Fine-tuning**: Ajuste fino de modelos pré-treinados
3. **Validation set**: Conjunto independente para avaliação

### Para Avaliação
1. **Benchmark**: Comparar performance em condições reais
2. **Generalização**: Testar robustez do modelo
3. **Análise de casos**: Identificar limitações e melhorias

## ⚠️ Considerações Importantes

### Limitações
- **Representatividade**: Nem todas as condições possíveis estão representadas
- **Viés geográfico**: Concentração em algumas regiões
- **Variação temporal**: Limitado a alguns períodos do ano

### Recomendações de Uso
- **Combinar com dados sintéticos**: Para máxima robustez
- **Validação local**: Incluir dados da região específica de aplicação
- **Atualização periódica**: Incorporar novas imagens regularmente

### Contato para Dúvidas
- **Email técnico**: pastagens.ia@projeto.br
- **Issues no GitHub**: Para reportar problemas ou sugestões
- **Documentação**: Consultar guias técnicos detalhados
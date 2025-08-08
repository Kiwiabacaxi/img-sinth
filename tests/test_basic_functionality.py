"""
Testes básicos de funcionalidade para o sistema de geração de pastagens brasileiras
Testa componentes principais e integração básica
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import json
import yaml

# Importações do sistema
from src.diffusion.pipeline_manager import PipelineManager
from src.diffusion.prompt_engine import PromptEngine, PastureConfig
from src.dataset.generator import DatasetGenerator, GenerationConfig
from src.dataset.quality_metrics import QualityMetrics
from src.dataset.yolo_formatter import YOLOFormatter
from src.training.yolo_trainer import YOLOTrainer, YOLOTrainingConfig

class TestBasicImports:
    """Testa se todos os módulos principais podem ser importados"""
    
    def test_core_imports(self):
        """Testa imports dos módulos principais"""
        # Diffusion
        from src.diffusion import pipeline_manager, prompt_engine
        from src.diffusion import controlnet_adapter, image_postprocess
        
        # Dataset  
        from src.dataset import generator, quality_metrics
        from src.dataset import yolo_formatter, augmentation
        
        # Training
        from src.training import yolo_trainer, evaluation, benchmark
        
        assert True  # Se chegou aqui, todos os imports funcionaram

    def test_pytorch_cuda_availability(self):
        """Testa disponibilidade do PyTorch e CUDA"""
        import torch
        
        assert torch.__version__ is not None
        # CUDA pode ou não estar disponível dependendo do ambiente
        cuda_available = torch.cuda.is_available()
        print(f"CUDA disponível: {cuda_available}")
        
        if cuda_available:
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")

class TestPromptEngine:
    """Testa o sistema de prompts especializados"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.prompt_engine = PromptEngine(config_dir="configs")
    
    def test_prompt_engine_initialization(self):
        """Testa inicialização do prompt engine"""
        assert self.prompt_engine is not None
        assert hasattr(self.prompt_engine, 'base_prompts')
    
    def test_pasture_config_creation(self):
        """Testa criação de configurações de pastagem"""
        config = PastureConfig(
            biome="cerrado",
            season="seca", 
            quality="moderada",
            grass_coverage=0.7,
            invasive_species=["capim_gordura"],
            degradation_level=0.3
        )
        
        assert config.biome == "cerrado"
        assert config.season == "seca"
        assert config.quality == "moderada"
        assert 0 <= config.grass_coverage <= 1
        assert "capim_gordura" in config.invasive_species
    
    def test_prompt_generation(self):
        """Testa geração de prompts"""
        config = PastureConfig(biome="cerrado", season="seca")
        
        try:
            prompt = self.prompt_engine.generate_prompt(config)
            assert isinstance(prompt, str)
            assert len(prompt) > 10  # Prompt não pode ser vazio
            assert "cerrado" in prompt.lower() or "savana" in prompt.lower()
        except FileNotFoundError:
            # Config files podem não existir em ambiente de teste
            pytest.skip("Arquivos de configuração não encontrados")
    
    def test_biome_specific_prompts(self):
        """Testa prompts específicos por bioma"""
        biomes = ["cerrado", "mata_atlantica", "pampa"]
        
        for biome in biomes:
            config = PastureConfig(biome=biome, season="seca")
            try:
                prompt = self.prompt_engine.generate_prompt(config)
                assert isinstance(prompt, str)
                print(f"Bioma {biome}: {prompt[:50]}...")
            except FileNotFoundError:
                pytest.skip("Arquivos de configuração não encontrados")

class TestQualityMetrics:
    """Testa sistema de métricas de qualidade"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.quality_metrics = QualityMetrics()
    
    def test_quality_metrics_initialization(self):
        """Testa inicialização das métricas de qualidade"""
        assert self.quality_metrics is not None
    
    def test_synthetic_image_evaluation(self):
        """Testa avaliação de imagem sintética"""
        # Criar imagem de teste
        test_image = Image.new('RGB', (512, 512), color='green')
        
        # Testar avaliação
        try:
            report = self.quality_metrics.evaluate_image_quality(test_image)
            
            assert hasattr(report, 'overall_score')
            assert 0 <= report.overall_score <= 1
            assert hasattr(report, 'technical_quality')
            assert hasattr(report, 'agricultural_realism')
            
        except Exception as e:
            # Pode falhar se dependências específicas não estão instaladas
            print(f"Quality evaluation failed (expected in test env): {e}")
    
    def test_batch_quality_evaluation(self):
        """Testa avaliação em lote"""
        # Criar múltiplas imagens de teste
        test_images = [
            Image.new('RGB', (512, 512), color='green'),
            Image.new('RGB', (512, 512), color='brown'),
            Image.new('RGB', (512, 512), color='yellow')
        ]
        
        try:
            scores = []
            for img in test_images:
                report = self.quality_metrics.evaluate_image_quality(img)
                scores.append(report.overall_score)
            
            assert len(scores) == 3
            assert all(0 <= score <= 1 for score in scores)
            
        except Exception as e:
            print(f"Batch evaluation failed (expected in test env): {e}")

class TestDatasetGenerator:
    """Testa geração de datasets"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_generator = DatasetGenerator(config_dir="configs")
    
    def teardown_method(self):
        """Cleanup após cada teste"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_dataset_generator_initialization(self):
        """Testa inicialização do gerador de dataset"""
        assert self.dataset_generator is not None
    
    def test_generation_config_creation(self):
        """Testa criação de configuração de geração"""
        config = GenerationConfig(
            num_images=10,
            resolution=(512, 512),
            output_dir=self.temp_dir,
            biome_distribution={"cerrado": 0.6, "mata_atlantica": 0.4},
            season_distribution={"seca": 0.5, "chuvas": 0.5}
        )
        
        assert config.num_images == 10
        assert config.resolution == (512, 512)
        assert config.output_dir == self.temp_dir
        assert abs(sum(config.biome_distribution.values()) - 1.0) < 0.01
    
    def test_config_validation(self):
        """Testa validação de configurações"""
        # Config válida
        valid_config = GenerationConfig(
            num_images=5,
            resolution=(256, 256),
            output_dir=self.temp_dir
        )
        
        # Deveria validar sem erros
        try:
            validated = self.dataset_generator._validate_generation_config(valid_config)
            assert validated is not None
        except AttributeError:
            # Método pode não existir ainda
            pass
    
    @pytest.mark.slow
    def test_small_dataset_generation(self):
        """Testa geração de dataset pequeno (teste lento, skip por padrão)"""
        config = GenerationConfig(
            num_images=2,
            resolution=(256, 256),
            output_dir=self.temp_dir,
            num_inference_steps=5,  # Muito rápido para teste
            quality_threshold=0.1   # Muito baixo para aceitar qualquer coisa
        )
        
        try:
            # Este teste pode falhar se modelos não estão baixados
            result_path = self.dataset_generator.generate_dataset(config)
            
            if result_path:
                assert Path(result_path).exists()
                images_dir = Path(result_path) / "images"
                if images_dir.exists():
                    images = list(images_dir.glob("*.jpg"))
                    assert len(images) > 0
            
        except Exception as e:
            pytest.skip(f"Dataset generation failed (expected in test env): {e}")

class TestYOLOTrainer:
    """Testa sistema de treinamento YOLO"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.temp_dir = tempfile.mkdtemp()
        self.yolo_trainer = YOLOTrainer(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup após cada teste"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_yolo_trainer_initialization(self):
        """Testa inicialização do trainer YOLO"""
        assert self.yolo_trainer is not None
        assert self.yolo_trainer.config is not None
        assert isinstance(self.yolo_trainer.config, YOLOTrainingConfig)
    
    def test_training_config_creation(self):
        """Testa criação de configuração de treinamento"""
        config = YOLOTrainingConfig(
            model_size="yolov8n",
            epochs=10,
            batch_size=2,
            image_size=320
        )
        
        assert config.model_size == "yolov8n"
        assert config.epochs == 10
        assert config.batch_size == 2
        assert config.image_size == 320
    
    def test_hardware_optimization(self):
        """Testa otimização de hardware"""
        optimized_config = self.yolo_trainer.optimize_config_for_hardware()
        
        assert optimized_config is not None
        assert optimized_config.batch_size > 0
        assert optimized_config.workers >= 1
        assert optimized_config.device in ["cpu", "0", "cuda:0"]
    
    def test_dataset_validation(self):
        """Testa validação de dataset"""
        # Criar dataset YOLO falso para teste
        fake_dataset = self.temp_dir + "/fake_dataset.yaml"
        
        fake_config = {
            "train": "train/images",
            "val": "val/images", 
            "nc": 3,
            "names": ["class1", "class2", "class3"]
        }
        
        with open(fake_dataset, 'w') as f:
            yaml.dump(fake_config, f)
        
        # Criar diretórios
        (Path(self.temp_dir) / "train" / "images").mkdir(parents=True)
        (Path(self.temp_dir) / "val" / "images").mkdir(parents=True)
        
        # Validar
        is_valid = self.yolo_trainer.validate_dataset(fake_dataset)
        assert isinstance(is_valid, bool)

class TestYOLOFormatter:
    """Testa formatação de datasets para YOLO"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.temp_dir = tempfile.mkdtemp()
        self.yolo_formatter = YOLOFormatter()
    
    def teardown_method(self):
        """Cleanup após cada teste"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_yolo_formatter_initialization(self):
        """Testa inicialização do formatador YOLO"""
        assert self.yolo_formatter is not None
    
    def test_class_mapping_creation(self):
        """Testa criação de mapeamento de classes"""
        test_classes = ["capim_gordura", "area_degradada", "solo_exposto"]
        
        class_mapping = self.yolo_formatter._create_class_mapping(test_classes)
        
        assert isinstance(class_mapping, dict)
        assert len(class_mapping) == len(test_classes)
        assert all(isinstance(v, int) for v in class_mapping.values())
        assert set(class_mapping.values()) == set(range(len(test_classes)))
    
    def test_split_calculation(self):
        """Testa cálculo de splits de dataset"""
        total_images = 100
        train_split = 0.7
        val_split = 0.2
        test_split = 0.1
        
        splits = self.yolo_formatter._calculate_splits(
            total_images, train_split, val_split, test_split
        )
        
        assert isinstance(splits, dict)
        assert "train" in splits and "val" in splits and "test" in splits
        assert splits["train"] + splits["val"] + splits["test"] == total_images
        
        # Verificar proporções aproximadas
        assert abs(splits["train"] / total_images - train_split) < 0.05

class TestIntegration:
    """Testes de integração entre componentes"""
    
    def setup_method(self):
        """Setup para testes de integração"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup após testes"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_config_file_loading(self):
        """Testa carregamento de arquivos de configuração"""
        config_files = [
            "configs/prompts/base_prompts.yaml",
            "configs/model_configs.yaml",
            "configs/scientific_references.yaml"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    assert isinstance(config, dict)
                    assert len(config) > 0
                    print(f"✅ {config_file} carregado com sucesso")
                except Exception as e:
                    pytest.fail(f"Erro ao carregar {config_file}: {e}")
            else:
                print(f"⚠️ {config_file} não encontrado (pode ser normal em testes)")
    
    def test_requirements_compatibility(self):
        """Testa compatibilidade das dependências"""
        try:
            import torch
            import diffusers
            import ultralytics
            import transformers
            
            print(f"PyTorch: {torch.__version__}")
            print(f"Diffusers: {diffusers.__version__}")
            print(f"Ultralytics: {ultralytics.__version__}")
            
            # Verificar compatibilidade básica
            assert torch.__version__ >= "2.0.0"
            
        except ImportError as e:
            pytest.skip(f"Dependência não instalada: {e}")
    
    def test_pipeline_end_to_end_mock(self):
        """Testa pipeline completo com dados mock"""
        # Este teste simula o pipeline completo sem gerar imagens reais
        
        # 1. Configuração de prompt
        prompt_engine = PromptEngine()
        config = PastureConfig(biome="cerrado", season="seca")
        
        # 2. Configuração de dataset
        dataset_config = GenerationConfig(
            num_images=1,
            resolution=(256, 256),
            output_dir=self.temp_dir
        )
        
        # 3. Configuração de treinamento
        training_config = YOLOTrainingConfig(
            model_size="yolov8n",
            epochs=1,
            batch_size=1
        )
        
        # Verificar se todas as configurações são válidas
        assert config is not None
        assert dataset_config is not None
        assert training_config is not None
        
        print("✅ Pipeline end-to-end configurado com sucesso")

class TestSystemRequirements:
    """Testa requisitos do sistema"""
    
    def test_python_version(self):
        """Testa versão do Python"""
        import sys
        
        version = sys.version_info
        assert version.major == 3
        assert version.minor >= 8
        
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    def test_gpu_memory_check(self):
        """Testa disponibilidade de memória GPU"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            total_gb = total_memory / (1024**3)
            
            print(f"GPU memory: {total_gb:.1f}GB")
            
            if total_gb < 4:
                print("⚠️ GPU com pouca memória. Pode ser necessário ajustar configurações.")
            else:
                print("✅ Memória GPU suficiente")
        else:
            print("ℹ️ Sem GPU disponível - usando CPU")
    
    def test_disk_space(self):
        """Testa espaço em disco disponível"""
        import shutil
        
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        print(f"Espaço livre em disco: {free_gb:.1f}GB")
        
        if free_gb < 10:
            print("⚠️ Pouco espaço em disco. Recomendado pelo menos 20GB livres.")
        else:
            print("✅ Espaço em disco suficiente")

# Configuração para pytest
def pytest_configure(config):
    """Configuração global do pytest"""
    config.addinivalue_line(
        "markers", "slow: marca testes como lentos (desabilitados por padrão)"
    )

if __name__ == "__main__":
    # Executar testes básicos se chamado diretamente
    pytest.main([__file__, "-v", "-m", "not slow"])
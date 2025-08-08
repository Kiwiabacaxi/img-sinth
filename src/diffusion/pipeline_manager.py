"""
Gerenciador principal do pipeline Stable Diffusion otimizado para Google Colab
Especializado em geração de imagens de pastagens brasileiras
"""

import torch
import os
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler
)
from diffusers.optimization import get_scheduler
import xformers

logger = logging.getLogger(__name__)

class PipelineManager:
    """
    Gerenciador central dos pipelines Stable Diffusion
    Otimizado para execução em Google Colab com GPUs T4/V100
    """
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        cache_dir: str = "/content/model_cache",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        enable_optimizations: bool = True
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.enable_optimizations = enable_optimizations
        
        # Pipeline instances
        self.base_pipeline = None
        self.controlnet_pipeline = None
        self.controlnet_models = {}
        
        # Otimizações de memória
        self.memory_optimizations = {
            'attention_slicing': True,
            'cpu_offload': True, 
            'sequential_cpu_offload': False,
            'enable_xformers': True,
            'gradient_checkpointing': True
        }
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Configura logging para debugging"""
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Inicializando PipelineManager")
        logger.info(f"Device: {self.device}")
        logger.info(f"Dtype: {self.torch_dtype}")
        
    def load_base_pipeline(self) -> StableDiffusionPipeline:
        """
        Carrega pipeline base Stable Diffusion
        """
        if self.base_pipeline is not None:
            return self.base_pipeline
            
        logger.info(f"Carregando modelo base: {self.model_name}")
        
        try:
            # Determinar tipo de pipeline
            if "xl" in self.model_name.lower():
                pipeline_class = StableDiffusionXLPipeline
            else:
                pipeline_class = StableDiffusionPipeline
                
            # Carregar pipeline
            self.base_pipeline = pipeline_class.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.torch_dtype == torch.float16 else None
            )
            
            # Mover para device
            self.base_pipeline = self.base_pipeline.to(self.device)
            
            # Aplicar otimizações
            if self.enable_optimizations:
                self._apply_optimizations(self.base_pipeline)
                
            logger.info("✅ Pipeline base carregado com sucesso")
            return self.base_pipeline
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar pipeline base: {e}")
            raise
            
    def load_controlnet_pipeline(
        self, 
        controlnet_type: str = "canny"
    ) -> StableDiffusionControlNetPipeline:
        """
        Carrega pipeline ControlNet para controle preciso de geração
        
        Args:
            controlnet_type: Tipo do ControlNet ('canny', 'depth', 'seg', 'scribble')
        """
        if self.controlnet_pipeline is not None:
            return self.controlnet_pipeline
            
        logger.info(f"Carregando ControlNet: {controlnet_type}")
        
        try:
            # Mapear tipos de ControlNet
            controlnet_models = {
                'canny': 'lllyasviel/sd-controlnet-canny',
                'depth': 'lllyasviel/sd-controlnet-depth',
                'seg': 'lllyasviel/sd-controlnet-seg', 
                'scribble': 'lllyasviel/sd-controlnet-scribble',
                'openpose': 'lllyasviel/sd-controlnet-openpose'
            }
            
            if controlnet_type not in controlnet_models:
                raise ValueError(f"ControlNet type '{controlnet_type}' não suportado")
                
            # Carregar ControlNet model
            controlnet_model_id = controlnet_models[controlnet_type]
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype
            )
            
            # Carregar pipeline ControlNet
            self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_name,
                controlnet=controlnet,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.torch_dtype == torch.float16 else None
            )
            
            # Mover para device
            self.controlnet_pipeline = self.controlnet_pipeline.to(self.device)
            
            # Aplicar otimizações
            if self.enable_optimizations:
                self._apply_optimizations(self.controlnet_pipeline)
                
            # Cache do modelo ControlNet
            self.controlnet_models[controlnet_type] = controlnet
            
            logger.info(f"✅ ControlNet {controlnet_type} carregado com sucesso")
            return self.controlnet_pipeline
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar ControlNet: {e}")
            raise
            
    def _apply_optimizations(self, pipeline):
        """
        Aplica otimizações de memória e performance para Colab
        """
        logger.info("🔧 Aplicando otimizações de performance...")
        
        try:
            # Attention slicing para reduzir uso de VRAM
            if self.memory_optimizations['attention_slicing']:
                pipeline.enable_attention_slicing()
                logger.info("✅ Attention slicing habilitado")
                
            # CPU offload para componentes não ativos
            if self.memory_optimizations['cpu_offload']:
                pipeline.enable_model_cpu_offload()
                logger.info("✅ CPU offload habilitado")
                
            # Sequential CPU offload (mais agressivo)
            elif self.memory_optimizations['sequential_cpu_offload']:
                pipeline.enable_sequential_cpu_offload()
                logger.info("✅ Sequential CPU offload habilitado")
                
            # xFormers para otimização de atenção
            if self.memory_optimizations['enable_xformers']:
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ xFormers habilitado")
                except:
                    logger.warning("⚠️ xFormers não disponível")
                    
            # VAE slicing
            if hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()
                logger.info("✅ VAE slicing habilitado")
                
        except Exception as e:
            logger.warning(f"⚠️ Algumas otimizações falharam: {e}")
            
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        controlnet_image: Optional[torch.Tensor] = None,
        controlnet_conditioning_scale: float = 1.0
    ) -> Dict:
        """
        Gera uma imagem usando o pipeline apropriado
        
        Args:
            prompt: Prompt principal
            negative_prompt: Prompt negativo
            num_inference_steps: Número de steps de denoising
            guidance_scale: Força do guidance do prompt
            width, height: Dimensões da imagem
            seed: Seed para reprodutibilidade
            controlnet_image: Imagem de condicionamento (se usando ControlNet)
            controlnet_conditioning_scale: Força do condicionamento ControlNet
            
        Returns:
            Dict contendo imagem gerada e metadados
        """
        
        # Definir pipeline a usar
        if controlnet_image is not None:
            if self.controlnet_pipeline is None:
                self.load_controlnet_pipeline()
            pipeline = self.controlnet_pipeline
        else:
            if self.base_pipeline is None:
                self.load_base_pipeline()  
            pipeline = self.base_pipeline
            
        # Configurar generator para reprodutibilidade
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        logger.info(f"Gerando imagem: {prompt[:50]}...")
        
        try:
            # Preparar argumentos
            generation_args = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'width': width,
                'height': height,
                'generator': generator
            }
            
            # Adicionar argumentos ControlNet se necessário
            if controlnet_image is not None:
                generation_args.update({
                    'image': controlnet_image,
                    'controlnet_conditioning_scale': controlnet_conditioning_scale
                })
                
            # Gerar imagem
            with torch.inference_mode():
                result = pipeline(**generation_args)
                
            # Preparar resultado
            output = {
                'image': result.images[0],
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'seed': seed,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'width': width,
                'height': height,
                'controlnet_used': controlnet_image is not None
            }
            
            logger.info("✅ Imagem gerada com sucesso")
            return output
            
        except Exception as e:
            logger.error(f"❌ Erro na geração: {e}")
            raise
            
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 1,
        **generation_kwargs
    ) -> List[Dict]:
        """
        Gera múltiplas imagens em batch
        
        Args:
            prompts: Lista de prompts
            batch_size: Tamanho do batch (cuidado com VRAM)
            **generation_kwargs: Argumentos para generate_image
            
        Returns:
            Lista de resultados
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            logger.info(f"Processando batch {i//batch_size + 1}/{len(prompts)//batch_size + 1}")
            
            for prompt in batch_prompts:
                try:
                    result = self.generate_image(prompt, **generation_kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Erro no prompt '{prompt[:30]}...': {e}")
                    continue
                    
            # Limpar cache entre batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results
        
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Retorna informações sobre uso de memória GPU
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA não disponível"}
            
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
        }
        
    def clear_memory(self):
        """
        Limpa cache de memória GPU
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 Cache GPU limpo")
            
    def unload_models(self):
        """
        Descarrega modelos da memória
        """
        self.base_pipeline = None
        self.controlnet_pipeline = None
        self.controlnet_models = {}
        self.clear_memory()
        logger.info("🔄 Modelos descarregados")
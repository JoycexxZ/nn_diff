import numpy as np
import torch
from typing import Optional
# from diffusers.utils import is_accelerator_available
from diffusers import DiffusionPipeline
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import logging, BaseOutput
from dataclasses import dataclass

logger = logging.get_logger(__name__)


@dataclass
class DiffPipelineOutput(BaseOutput):
    model_weights: torch.Tensor
    
    
class WeightDiffPipeline(DiffusionPipeline):
    def __init__(self, vae, unet, scheduler):
        super().__init__()
        if vae is None:
            self.register_modules(
                unet=unet,
                scheduler=scheduler
            )
        else:
            self.register_modules(
                vae=vae,
                unet=unet,
                scheduler=scheduler
            )
        self.use_vae = vae is not None
    
    @torch.no_grad()
    def __call__(self,
                 num_inference_steps: int = 50,
                 latents: torch.Tensor = None,
                 batch_size: int = 1,
    ):
        device = self._execution_device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        if latents is None:
            latents = torch.randn(batch_size, self.unet.sample_size, device=device)
            
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(num_warmup_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                
                noise_pred = self.unet(latent_model_input, t)
                
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    
        return DiffPipelineOutput(model_weights=latents)
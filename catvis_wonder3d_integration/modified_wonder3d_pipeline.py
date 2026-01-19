"""
Modified Wonder3D Pipeline to accept external embeddings (e.g., from CATVis EEG)
"""
import torch
import torch.nn.functional as F
from typing import Optional, Union, List
from diffusers import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor


class ModifiedMVDiffusionPipeline(DiffusionPipeline):
    """
    Modified pipeline that accepts pre-computed embeddings instead of images
    """

    def __init__(self, vae, unet, scheduler):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler
        )

    def prepare_camera_embedding(self, camera_embedding):
        """
        Convert camera pose to sinusoidal encoding
        camera_embedding: [B, 3] -> [elevation_cond, elevation_target, azimuth_target]
        Returns: [B, 6] sinusoidal encoded
        """
        # Convert to radians
        camera_embedding = camera_embedding * torch.pi / 180.0

        # Sinusoidal encoding
        camera_embedding = torch.cat([
            torch.sin(camera_embedding),
            torch.cos(camera_embedding)
        ], dim=-1)

        return camera_embedding  # [B, 6]

    @torch.no_grad()
    def __call__(
        self,
        prompt_embeds: torch.Tensor,  # Pre-computed embeddings [B, 1, 768]
        camera_embedding: torch.Tensor,  # Camera poses [B, 3]
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
    ):
        """
        Generate multi-view images from pre-computed embeddings

        Args:
            prompt_embeds: [batch_size, 1, 768] - semantic embeddings (from EEG/CLIP)
            camera_embedding: [batch_size, 3] - camera poses
            height/width: Output image size
            num_inference_steps: Diffusion steps
            guidance_scale: CFG strength
            negative_prompt_embeds: Optional negative embeddings
            eta: DDIM eta parameter
            generator: Random seed generator
            output_type: 'pil' or 'pt'
        """
        device = self._execution_device
        batch_size = prompt_embeds.shape[0]

        # Classifier-free guidance setup
        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare embeddings
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Prepare camera embeddings
        camera_embedding = self.prepare_camera_embedding(camera_embedding)

        if do_classifier_free_guidance:
            negative_camera_embedding = torch.zeros_like(camera_embedding)
            camera_embedding = torch.cat([negative_camera_embedding, camera_embedding])

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = randn_tensor(
            (batch_size, num_channels_latents, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=prompt_embeds.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                class_labels=camera_embedding,  # Camera conditioning
                return_dict=False
            )[0]

            # Classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, eta=eta, generator=generator).prev_sample

        # Decode latents
        latents = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.decode(latents, return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)

        # Convert to output format
        if output_type == "pil":
            images = self.numpy_to_pil(images.cpu().permute(0, 2, 3, 1).numpy())
        elif output_type == "pt":
            pass  # Already tensor
        else:
            images = images.cpu()

        return type('Output', (), {'images': images})()

"""
EEG to Multi-View - Better approach using CATVis generation pipeline
Pipeline: EEG → CATVis (generate image) → Wonder3D (multi-view)
"""
import sys
import os
import torch
from pathlib import Path
import yaml
from PIL import Image
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set working directory early
os.chdir(project_root)

# CATVis imports
from src.models.eeg_classifier import EEGClassifier
from src.models.contrastive_encoder import ContrastiveEncoder
from src.data.data_loader import CATVisDataLoader
from src.data.preprocessor import DataPreprocessor
from src.pipeline.text_retrieval import TextRetrieval
from src.pipeline.image_generator import ImageGenerator

# Only basic imports here, Wonder3D imports inside functions
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


class EEGToMultiViewV2:
    def __init__(self, config_path, classifier_ckpt, contrastive_ckpt):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("Loading CATVis models...")
        # Load classifier for class prediction
        self.classifier = EEGClassifier(self.config).to(self.device)
        state_dict = torch.load(classifier_ckpt, map_location=self.device)
        self.classifier.model.load_state_dict(state_dict)
        self.classifier.eval()

        # Load contrastive for retrieval
        self.contrastive = ContrastiveEncoder(self.config).to(self.device)
        state_dict = torch.load(contrastive_ckpt, map_location=self.device)
        self.contrastive.model.load_state_dict(state_dict)
        self.contrastive.eval()

        # Image generator (Stable Diffusion)
        print("Loading Stable Diffusion...")
        self.image_gen = ImageGenerator(self.config, self.device)

        # Wonder3D pipeline
        print("Loading Wonder3D...")
        self._load_wonder3d()

        print("Setup complete!")

    def _load_wonder3d(self):
        """Load Wonder3D with CLIP image encoder"""
        # Add Wonder3D to path only inside this function
        wonder3d_path = str(Path(__file__).parent.parent / 'Wonder3D')
        if wonder3d_path not in sys.path:
            sys.path.append(wonder3d_path)

        # Now import Wonder3D modules
        from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
        from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

        base_model = "lambdalabs/sd-image-variations-diffusers"

        # CLIP image encoder
        self.feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model, subfolder="image_encoder"
        ).to(self.device)

        # VAE
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(self.device)

        # UNet (Wonder3D checkpoint)
        unet_path = "G:/Rabia-Salman/CATVis/Wonder3D/ckpts"
        try:
            unet = UNetMV2DConditionModel.from_pretrained(
                unet_path,
                subfolder="unet",
                num_views=6,
                multiview_attention=True,
                cross_attention_dim=1280
            ).to(self.device)
            print("  Loaded Wonder3D UNet")
        except Exception as e:
            print(f"  Warning: Could not load Wonder3D UNet: {e}")
            print("  Using base UNet (no multi-view)")
            from diffusers import UNet2DConditionModel
            unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet").to(self.device)

        # Scheduler
        scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")

        # Create pipeline
        self.wonder3d = MVDiffusionImagePipeline(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            image_encoder=self.image_encoder,
            feature_extractor=self.feature_extractor,
            safety_checker=None,
            requires_safety_checker=False
        )

    def generate_from_eeg(self, eeg_data, text_retrieval, num_sd_samples=1):
        """
        Complete pipeline: EEG → Image → Multi-view

        Args:
            eeg_data: [1, 128, 440]
            text_retrieval: TextRetrieval instance
            num_sd_samples: How many SD images to try (we'll use best)

        Returns:
            sd_image: Generated single image
            multiview_images: [6, 3, 256, 256] multi-view images
        """
        print("\n=== Step 1: EEG Classification ===")
        with torch.no_grad():
            outputs, _ = self.classifier(eeg_data)
            predicted_class_idx = outputs.argmax(dim=-1).item()

        predicted_class = self.config['class_prompts'][
            self.config['eeg_classification']['class_labels'][predicted_class_idx]
        ]
        print(f"Predicted class: {predicted_class}")

        print("\n=== Step 2: Caption Retrieval ===")
        retrieved_captions = text_retrieval.retrieve_top_k(
            self.contrastive, eeg_data, k=10
        )
        print(f"Top caption: {retrieved_captions[0]}")

        print("\n=== Step 3: Generate image with Stable Diffusion ===")
        # Use CATVis image generation
        sd_image = self.image_gen.generate_single_image(
            predicted_class,
            retrieved_captions[0]
        )

        # Save intermediate result
        sd_image_pil = Image.fromarray((sd_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        print(f"Generated SD image: {sd_image.shape}")

        print("\n=== Step 4: Generate multi-view with Wonder3D ===")
        # Prepare for Wonder3D (expects 256x256)
        sd_image_resized = torch.nn.functional.interpolate(
            sd_image.unsqueeze(0),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )[0]

        # Convert to PIL for Wonder3D
        sd_image_pil_256 = Image.fromarray(
            (sd_image_resized.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        )

        # Camera poses for 6 views
        camera_embeddings = torch.tensor([
            [0, 0, 0],        # front
            [0, 0, 30],       # front_right
            [0, 0, 90],       # right
            [0, 0, 180],      # back
            [0, 0, 270],      # left
            [0, 0, 330],      # front_left
        ], dtype=torch.float32, device=self.device)

        # Generate multi-view
        with torch.no_grad():
            output = self.wonder3d(
                image=sd_image_pil_256,
                camera_embedding=camera_embeddings,
                guidance_scale=3.0,
                num_inference_steps=50,
                output_type='pt'
            )

        multiview_images = output.images
        print(f"Generated {multiview_images.shape[0]} views")

        return sd_image_pil, multiview_images

    def save_results(self, sd_image, multiview_images, output_dir, prefix='eeg'):
        """Save all results"""
        from torchvision.utils import save_image

        os.makedirs(output_dir, exist_ok=True)

        # Save SD image
        sd_image.save(os.path.join(output_dir, f'{prefix}_sd_generated.png'))

        # Save multi-view images
        view_names = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        for i, view_name in enumerate(view_names):
            save_path = os.path.join(output_dir, f'{prefix}_view_{view_name}.png')
            save_image(multiview_images[i], save_path)

        print(f"All results saved to {output_dir}")


def main():
    # Already changed dir at top
    project_root = Path(__file__).parent.parent

    config_path = str(project_root / 'config' / 'config.yaml')
    classifier_ckpt = str(project_root / 'checkpoints' / 'eeg_classifier_best.pth')
    contrastive_ckpt = str(project_root / 'checkpoints' / 'contrastive_model_best.pth')

    # Initialize
    generator = EEGToMultiViewV2(config_path, classifier_ckpt, contrastive_ckpt)

    # Load test data
    print("\nLoading test data...")
    data_loader = CATVisDataLoader(generator.config)
    df = data_loader.get_dataset_dataframe()
    _, _, test_df = data_loader.get_train_val_test_splits(df)

    preprocessor = DataPreprocessor(generator.config)
    test_dataset = preprocessor.create_pipeline_dataset(test_df)

    # Setup text retrieval
    text_retrieval = TextRetrieval(generator.config, test_df)

    # Get test sample
    sample = test_dataset[0]
    eeg_data = sample['eeg'].unsqueeze(0).to(generator.device)

    print(f"\nProcessing EEG sample...")
    print(f"Ground truth label: {sample['label']}")

    # Generate
    sd_image, multiview_images = generator.generate_from_eeg(
        eeg_data, text_retrieval, num_sd_samples=1
    )

    # Save
    output_dir = str(project_root / 'outputs' / 'eeg_multiview_v2')
    generator.save_results(sd_image, multiview_images, output_dir, prefix='sample_0')

    print("\n✅ Done!")


if __name__ == '__main__':
    main()

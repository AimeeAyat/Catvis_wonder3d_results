"""
EEG to Multi-View Image Generation using CATVis + Wonder3D
"""
import sys
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import yaml

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'Wonder3D'))

from src.models.contrastive_encoder import ContrastiveEncoder
from src.data.data_loader import CATVisDataLoader
from src.data.preprocessor import DataPreprocessor

# Wonder3D imports
from diffusers import AutoencoderKL, DDIMScheduler

# Use our modified pipeline
from modified_wonder3d_pipeline import ModifiedMVDiffusionPipeline


class EEGToMultiViewGenerator:
    def __init__(self, catvis_config_path, wonder3d_config_path, contrastive_checkpoint):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load CATVis config and model
        with open(catvis_config_path, 'r') as f:
            self.catvis_config = yaml.safe_load(f)

        print("Loading CATVis contrastive model...")
        self.catvis_model = ContrastiveEncoder(self.catvis_config).to(self.device)
        checkpoint = torch.load(contrastive_checkpoint, map_location=self.device)
        self.catvis_model.model.load_state_dict(checkpoint)
        self.catvis_model.eval()

        # Load Wonder3D config
        with open(wonder3d_config_path, 'r') as f:
            self.wonder3d_config = yaml.safe_load(f)

        print("Loading Wonder3D pipeline...")
        self._load_wonder3d_pipeline()

    def _load_wonder3d_pipeline(self):
        """Load Wonder3D UNet, VAE, and create pipeline"""
        # Load pre-trained models
        base_model = "lambdalabs/sd-image-variations-diffusers"

        print("  Loading VAE...")
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")

        print("  Loading Wonder3D UNet...")
        # Load UNet from checkpoint
        unet_path = "G:/Rabia-Salman/CATVis/Wonder3D/ckpts"
        try:
            from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
            unet = UNetMV2DConditionModel.from_pretrained(
                unet_path,
                subfolder="unet",
                num_views=6,
                multiview_attention=True,
                cross_attention_dim=1280
            )
        except:
            print("  Warning: Could not load Wonder3D checkpoint, using base model")
            # Fallback: use standard UNet (won't have multi-view attention)
            from diffusers import UNet2DConditionModel
            unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")

        print("  Loading scheduler...")
        scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")

        # Create our modified pipeline
        self.wonder3d_pipeline = ModifiedMVDiffusionPipeline(
            vae=vae,
            unet=unet,
            scheduler=scheduler
        ).to(self.device)

        print("  Wonder3D pipeline loaded!")

    def get_eeg_embedding(self, eeg_data):
        """
        Extract 768-d CLIP-aligned embedding from EEG

        Args:
            eeg_data: torch.Tensor [1, 128, 440]

        Returns:
            embedding: torch.Tensor [1, 1, 768] for cross-attention
        """
        with torch.no_grad():
            eeg_embedding = self.catvis_model(eeg_data)  # [1, 768]
            eeg_embedding = F.normalize(eeg_embedding, dim=-1)  # Normalize to unit sphere
            eeg_embedding = eeg_embedding.unsqueeze(1)  # [1, 1, 768] for cross-attention
        return eeg_embedding

    def prepare_camera_embeddings(self, num_views=6):
        """
        Prepare camera pose embeddings for multi-view generation

        Args:
            num_views: 6 (front, front_right, right, back, left, front_left)

        Returns:
            camera_embeddings: torch.Tensor [6, 3]
        """
        # Camera poses: [elevation_cond, elevation_target, azimuth_target]
        if num_views == 6:
            poses = torch.tensor([
                [0, 0, 0],        # front (conditioning view)
                [0, 0, 30],       # front_right
                [0, 0, 90],       # right
                [0, 0, 180],      # back
                [0, 0, 270],      # left
                [0, 0, 330],      # front_left
            ], dtype=torch.float32)
        elif num_views == 4:
            poses = torch.tensor([
                [0, 0, 0],        # front
                [0, 0, 90],       # right
                [0, 0, 180],      # back
                [0, 0, 270],      # left
            ], dtype=torch.float32)
        else:
            raise ValueError(f"num_views={num_views} not supported")

        return poses.to(self.device)

    def generate_multiview_from_eeg(self, eeg_data, num_views=6, guidance_scale=3.0, num_inference_steps=50):
        """
        Generate multi-view images from EEG data

        Args:
            eeg_data: torch.Tensor [1, 128, 440]
            num_views: int (4 or 6)
            guidance_scale: float (classifier-free guidance strength)
            num_inference_steps: int (diffusion steps)

        Returns:
            images: torch.Tensor [num_views, 3, 256, 256]
        """
        # Get EEG embedding
        eeg_embedding = self.get_eeg_embedding(eeg_data)  # [1, 1, 768]

        # Prepare camera poses
        camera_embeddings = self.prepare_camera_embeddings(num_views)  # [num_views, 3]

        # Replicate EEG embedding for all views
        eeg_embedding = eeg_embedding.repeat(num_views, 1, 1)  # [num_views, 1, 768]

        # Generate with Wonder3D
        # Note: We bypass the image encoding step and directly provide embeddings
        with torch.no_grad():
            output = self.wonder3d_pipeline(
                prompt_embeds=eeg_embedding,  # Use EEG embeddings instead of CLIP image
                camera_embedding=camera_embeddings,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type='pt'
            )

        images = output.images  # [num_views, 3, 256, 256]
        return images

    def save_multiview_images(self, images, output_dir, prefix='eeg_view'):
        """Save generated multi-view images"""
        from torchvision.utils import save_image

        os.makedirs(output_dir, exist_ok=True)

        view_names = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

        for i, view_name in enumerate(view_names[:len(images)]):
            save_path = os.path.join(output_dir, f'{prefix}_{view_name}.png')
            save_image(images[i], save_path)
            print(f"Saved: {save_path}")


def main():
    # Paths (relative to project root)
    project_root = Path(__file__).parent.parent

    # Change working directory to project root (for data loading)
    os.chdir(project_root)

    catvis_config = project_root / 'config' / 'config.yaml'
    wonder3d_config = project_root / 'Wonder3D' / 'configs' / 'mvdiffusion-joint-ortho-6views.yaml'
    contrastive_checkpoint = project_root / 'checkpoints' / 'contrastive_model_best.pth'

    # Initialize generator
    generator = EEGToMultiViewGenerator(
        catvis_config_path=str(catvis_config),
        wonder3d_config_path=str(wonder3d_config),
        contrastive_checkpoint=str(contrastive_checkpoint)
    )

    # Load test EEG sample
    print("\nLoading test EEG data...")
    data_loader = CATVisDataLoader(generator.catvis_config)
    df = data_loader.get_dataset_dataframe()
    _, _, test_df = data_loader.get_train_val_test_splits(df)

    preprocessor = DataPreprocessor(generator.catvis_config)
    test_dataset = preprocessor.create_pipeline_dataset(test_df)

    # Get first test sample
    sample = test_dataset[0]
    eeg_data = sample['eeg'].unsqueeze(0).to(generator.device)  # [1, 128, 440]

    print(f"EEG data shape: {eeg_data.shape}")
    print(f"Ground truth label: {sample['label']}")

    # Generate multi-view images
    print("\nGenerating 6-view images from EEG...")
    images = generator.generate_multiview_from_eeg(
        eeg_data,
        num_views=6,
        guidance_scale=3.0,
        num_inference_steps=50
    )

    print(f"Generated images shape: {images.shape}")

    # Save outputs
    output_dir = project_root / 'outputs' / 'eeg_multiview'
    generator.save_multiview_images(images, str(output_dir), prefix='sample_0')

    print(f"\nDone! Multi-view images saved to {output_dir}")


if __name__ == '__main__':
    main()

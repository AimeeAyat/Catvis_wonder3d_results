"""
EEG-to-Image Generation Pipeline with Hybrid Verification

This script generates images from EEG signals and verifies them against:
1. EEG embeddings (from contrastive encoder)
2. Ground truth image embeddings (from CLIP)

If similarity is below threshold, it regenerates with different seeds
and selects the best matching image.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import os
import torch
import torch.nn.functional as F
import yaml
import subprocess
import shutil
import random
import clip
from PIL import Image
from typing import Tuple, Optional

os.chdir(project_root)

from src.models.eeg_classifier import EEGClassifier
from src.models.contrastive_encoder import ContrastiveEncoder
from src.data.data_loader import CATVisDataLoader
from src.data.preprocessor import DataPreprocessor
from src.pipeline.retrieval import TextRetrieval

# Use SDXL with IP-Adapter for image-guided generation
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# Background removal
from rembg import remove


class ImageVerifier:
    """
    Verifies generated images against EEG and ground truth embeddings.
    Uses CLIP for image encoding and compares with EEG embeddings.
    """

    def __init__(self, contrastive_model, device: torch.device,
                 eeg_weight: float = 0.5, gt_weight: float = 0.5):
        """
        Args:
            contrastive_model: Trained contrastive encoder for EEG
            device: torch device
            eeg_weight: Weight for EEG similarity (0-1)
            gt_weight: Weight for ground truth similarity (0-1)
        """
        self.device = device
        self.contrastive_model = contrastive_model
        self.eeg_weight = eeg_weight
        self.gt_weight = gt_weight

        # Load CLIP for image encoding
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode PIL image to CLIP embedding."""
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.clip_model.encode_image(image_input).float()
            image_embedding = F.normalize(image_embedding, dim=-1)
        return image_embedding

    def encode_eeg(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """Encode EEG signal to embedding using contrastive model."""
        if len(eeg_tensor.shape) == 2:
            eeg_tensor = eeg_tensor.unsqueeze(0)
        eeg_tensor = eeg_tensor.to(self.device)

        with torch.no_grad():
            eeg_embedding = self.contrastive_model(eeg_tensor)
            eeg_embedding = F.normalize(eeg_embedding, dim=-1)
        return eeg_embedding

    def compute_similarity(self,
                          generated_image: Image.Image,
                          eeg_tensor: torch.Tensor,
                          gt_image: Optional[Image.Image] = None) -> dict:
        """
        Compute similarity scores between generated image and references.

        Returns:
            dict with 'eeg_sim', 'gt_sim', 'combined_score'
        """
        # Encode generated image
        gen_embedding = self.encode_image(generated_image)

        # Compute EEG similarity
        eeg_embedding = self.encode_eeg(eeg_tensor)
        eeg_sim = F.cosine_similarity(gen_embedding, eeg_embedding).item()

        # Compute ground truth similarity if available
        gt_sim = 0.0
        if gt_image is not None:
            gt_embedding = self.encode_image(gt_image)
            gt_sim = F.cosine_similarity(gen_embedding, gt_embedding).item()

        # Combined score
        if gt_image is not None:
            combined_score = self.eeg_weight * eeg_sim + self.gt_weight * gt_sim
        else:
            combined_score = eeg_sim

        return {
            'eeg_sim': eeg_sim,
            'gt_sim': gt_sim,
            'combined_score': combined_score
        }


def is_caption_relevant(predicted_class: str, caption: str) -> bool:
    """Check if caption is relevant to the predicted class."""
    # Extract key words from predicted class
    class_words = predicted_class.lower().replace("-", " ").replace("_", " ").split()
    caption_lower = caption.lower()

    # Check if any class word appears in caption
    for word in class_words:
        if len(word) > 3 and word in caption_lower:  # Skip short words
            return True

    return False


def create_enhanced_prompt(predicted_class: str, caption: str) -> Tuple[str, bool]:
    """
    Create enhanced prompt for single object with pose/action details.

    Returns:
        tuple: (prompt, caption_used) - prompt string and whether caption was used
    """

    # Check if caption is relevant to predicted class
    use_caption = is_caption_relevant(predicted_class, caption)

    if use_caption:
        # Remove only multi-object and background phrases, keep action words
        caption_clean = caption.replace("A group of", "A single")
        caption_clean = caption_clean.replace("group of", "single")
        caption_clean = caption_clean.replace("Several", "One").replace("several", "one")
        caption_clean = caption_clean.replace("Many", "One").replace("many", "one")
        caption_clean = caption_clean.replace("Two", "One").replace("two", "one")
        caption_clean = caption_clean.replace("Three", "One").replace("three", "one")
        caption_clean = caption_clean.replace("people", "").replace("persons", "")
        caption_clean = caption_clean.replace("men", "").replace("women", "")

        prompt = (
            f"ONE single {predicted_class}, exactly one object, "
            f"{caption_clean}, "
            f"solo subject only, no other objects, "
            f"isolated on pure white background, "
            f"centered, studio product photography, "
            f"professional lighting, sharp focus, 8k"
        )
    else:
        # Caption doesn't match predicted class - use only predicted class
        prompt = (
            f"ONE single {predicted_class}, exactly one object, "
            f"solo subject only, no other objects, "
            f"isolated on pure white background, "
            f"centered, studio product photography, "
            f"professional lighting, sharp focus, 8k, photorealistic"
        )

    return prompt, use_caption


def create_negative_prompt() -> str:
    """Create negative prompt to avoid multiple objects."""
    return (
        "multiple objects, two objects, three objects, many objects, "
        "group, collection, crowd, pair, couple, several, duplicate, "
        "second object, additional items, extra objects, "
        "people, person, human, hands, face, "
        "busy background, cluttered, complex scene, "
        "text, watermark, logo, signature, "
        "blurry, low quality, distorted, deformed, "
        "cropped, cut off, partial object"
    )


def generate_with_verification(
    pipe,
    verifier: ImageVerifier,
    prompt: str,
    negative_prompt: str,
    eeg_tensor: torch.Tensor,
    gt_image: Optional[Image.Image] = None,
    threshold: float = 0.50,
    max_attempts: int = 5,
    base_seed: int = 42,
    ip_adapter_scale: float = 0.6
) -> Tuple[Image.Image, dict, int]:
    """
    Generate image using IP-Adapter with GT image guidance.

    Strategy:
    - Uses IP-Adapter to inject GT image features into generation
    - Iteratively adjusts IP-Adapter scale to find best balance
    - Each attempt varies seed and IP-Adapter influence

    Args:
        pipe: SDXL pipeline with IP-Adapter loaded
        verifier: ImageVerifier instance
        prompt: Generation prompt
        negative_prompt: Negative prompt
        eeg_tensor: EEG signal tensor
        gt_image: Ground truth image for IP-Adapter guidance
        threshold: Minimum acceptable similarity score
        max_attempts: Maximum regeneration attempts
        base_seed: Base seed for generation
        ip_adapter_scale: Initial IP-Adapter influence (0-1)

    Returns:
        (best_image, best_scores, attempts_used)
    """
    best_image = None
    best_scores = {'eeg_sim': 0, 'gt_sim': 0, 'combined_score': 0}
    best_combined = -1

    for attempt in range(max_attempts):
        seed = base_seed + attempt * 100

        # Adjust IP-Adapter scale per attempt (start strong, decrease if needed)
        # Scales: 0.6 -> 0.5 -> 0.4 -> 0.7 -> 0.3
        scales = [0.6, 0.5, 0.4, 0.7, 0.3]
        current_scale = scales[attempt % len(scales)]

        if gt_image is not None:
            # Use IP-Adapter with GT image guidance
            pipe.set_ip_adapter_scale(current_scale)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image=gt_image,
                guidance_scale=7.5,
                num_inference_steps=30,
                generator=torch.Generator("cpu").manual_seed(seed),
                height=1024,
                width=1024,
            )
            print(f"  Attempt {attempt + 1}/{max_attempts} [IP-Adapter, scale={current_scale}]: ", end="")
        else:
            # No GT image - use text-only generation
            pipe.set_ip_adapter_scale(0.0)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=7.5,
                num_inference_steps=30,
                generator=torch.Generator("cpu").manual_seed(seed),
                height=1024,
                width=1024,
            )
            print(f"  Attempt {attempt + 1}/{max_attempts} [Text-only]: ", end="")

        generated_image = result.images[0]

        # Compute similarity scores
        scores = verifier.compute_similarity(generated_image, eeg_tensor, gt_image)

        print(f"EEG={scores['eeg_sim']:.3f}, "
              f"GT={scores['gt_sim']:.3f}, "
              f"Combined={scores['combined_score']:.3f}")

        # Update best if this is better
        if scores['combined_score'] > best_combined:
            best_combined = scores['combined_score']
            best_image = generated_image
            best_scores = scores
            print(f"    -> New best!")

        # Accept if above threshold
        if scores['combined_score'] >= threshold:
            print(f"  Accepted at attempt {attempt + 1} (score >= {threshold})")
            return best_image, best_scores, attempt + 1

    print(f"  Using best from {max_attempts} attempts (score={best_combined:.3f})")
    return best_image, best_scores, max_attempts


def load_ground_truth_image(data_loader, test_df, sample_idx: int, config: dict) -> Optional[Image.Image]:
    """Load the ground truth image for a sample."""
    try:
        row = test_df.iloc[sample_idx]
        image_idx = row['image']
        image_path_rel = data_loader.image_to_path[image_idx]
        image_path = Path(config['data']['root_dir']) / config['data']['imagenet_images'] / image_path_rel

        if image_path.exists():
            return Image.open(image_path).convert('RGB')
        else:
            print(f"  Warning: GT image not found at {image_path}")
            return None
    except Exception as e:
        print(f"  Warning: Could not load GT image: {e}")
        return None


def main():
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load EEG models
    print("Loading EEG models...")
    classifier = EEGClassifier(config).to(device)
    classifier.model.load_state_dict(torch.load('checkpoints/eeg_classifier_best.pth', map_location=device))
    classifier.eval()

    contrastive = ContrastiveEncoder(config).to(device)
    contrastive.model.load_state_dict(torch.load('checkpoints/contrastive_model_best.pth', map_location=device))
    contrastive.eval()

    # Load data
    print("Loading data...")
    data_loader = CATVisDataLoader(config)
    df = data_loader.get_dataset_dataframe()
    _, _, test_df = data_loader.get_train_val_test_splits(df)

    preprocessor = DataPreprocessor(config)
    test_dataset = preprocessor.create_pipeline_dataset(test_df)

    text_retrieval = TextRetrieval(config, contrastive, device)
    text_retrieval.setup_retrieval_corpus(test_df)

    # Initialize verifier
    print("Initializing image verifier...")
    verifier = ImageVerifier(
        contrastive_model=contrastive,
        device=device,
        eeg_weight=0.5,  # Balance between EEG and GT similarity
        gt_weight=0.5
    )

    # Load SDXL with IP-Adapter
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    # Load IP-Adapter for image-guided generation
    print("Loading IP-Adapter...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin"
    )
    pipe.enable_model_cpu_offload()
    print("SDXL with IP-Adapter loaded successfully")

    # Setup output directories
    output_dir = Path('outputs/eeg_verified_images')
    output_dir.mkdir(parents=True, exist_ok=True)

    wonder3d_dir = Path('Wonder3D')
    wonder3d_input = wonder3d_dir / 'example_images'
    wonder3d_input.mkdir(parents=True, exist_ok=True)

    # Configuration
    SIMILARITY_THRESHOLD = 0.5  # Minimum acceptable combined score
    MAX_ATTEMPTS = 5  # Max regeneration attempts per sample
    NUM_SAMPLES = 10

    indices = random.sample(range(len(test_dataset)), min(NUM_SAMPLES, len(test_dataset)))

    # Statistics
    total_attempts = 0
    all_scores = []

    for idx, sample_idx in enumerate(indices):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{NUM_SAMPLES} (index {sample_idx})")
        print(f"{'='*60}")

        sample = test_dataset[sample_idx]
        eeg_data = sample['eeg'].unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            outputs, _ = classifier(eeg_data)
            pred_idx = outputs.argmax().item()

        labels_order = data_loader.raw_data['labels']
        predicted_class = config['class_prompts'][labels_order[pred_idx]]

        # Get caption
        retrieved = text_retrieval.retrieve_top_k_from_eeg(eeg_data.squeeze(0), k=1)
        top_caption = retrieved[0][0]

        # Get ground truth class
        gt_row = test_df.iloc[sample_idx]
        gt_class = gt_row['class']

        print(f"Predicted: {predicted_class}")
        print(f"GT Class: {gt_class}")
        print(f"Caption: {top_caption[:60]}...")

        # Check if predicted class matches GT class
        classes_match = (predicted_class.lower() == gt_class.lower())

        # Load ground truth image only if classes match
        gt_image = None
        if classes_match:
            gt_image = load_ground_truth_image(data_loader, test_df, sample_idx, config)
            if gt_image:
                print(f"Classes MATCH - using IP-Adapter with GT image")
        else:
            print(f"Classes DIFFER - generating with prompt only (no IP-Adapter)")

        # Create prompts
        prompt, caption_used = create_enhanced_prompt(predicted_class, top_caption)
        negative_prompt = create_negative_prompt()

        if caption_used:
            print(f"Using caption in prompt (matches predicted class)")
        else:
            print(f"Ignoring caption (doesn't match predicted class)")
        print(f"Generating with verification...")

        # Generate with IP-Adapter (if GT available) or text-only
        best_image, scores, attempts = generate_with_verification(
            pipe=pipe,
            verifier=verifier,
            prompt=prompt,
            negative_prompt=negative_prompt,
            eeg_tensor=eeg_data.squeeze(0),
            gt_image=gt_image,
            threshold=SIMILARITY_THRESHOLD,
            max_attempts=MAX_ATTEMPTS,
            base_seed=42 + idx * 1000
        )

        total_attempts += attempts
        all_scores.append(scores)

        # Remove background
        print("Removing background...")
        image_no_bg = remove(best_image)

        # Save image
        img_name = f"sample_{idx:03d}.png"
        img_path = output_dir / img_name
        image_no_bg.save(img_path)
        print(f"Saved: {img_path}")

        # Save scores to a text file
        scores_file = output_dir / f"sample_{idx:03d}_scores.txt"
        with open(scores_file, 'w') as f:
            f.write(f"Predicted class: {predicted_class}\n")
            f.write(f"GT class: {gt_class}\n")
            f.write(f"Classes match: {classes_match}\n")
            f.write(f"IP-Adapter used: {gt_image is not None}\n")
            f.write(f"Caption: {top_caption}\n")
            f.write(f"Caption used in prompt: {caption_used}\n")
            f.write(f"EEG similarity: {scores['eeg_sim']:.4f}\n")
            f.write(f"GT similarity: {scores['gt_sim']:.4f}\n")
            f.write(f"Combined score: {scores['combined_score']:.4f}\n")
            f.write(f"Attempts used: {attempts}\n")

        # Copy to Wonder3D input
        wonder3d_input_path = wonder3d_input / img_name
        shutil.copy(img_path, wonder3d_input_path)

        # Optionally run Wonder3D
        print("Running Wonder3D...")
        try:
            subprocess.run([
                'python', 'test_mvdiffusion_seq.py',
                '--config', 'configs/mvdiffusion-joint-ortho-6views.yaml',
                f'validation_dataset.filepaths=[{img_name}]'
            ], cwd=str(wonder3d_dir), check=True, capture_output=True)
            print("Wonder3D completed")
        except subprocess.CalledProcessError as e:
            print(f"Wonder3D failed: {e}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {NUM_SAMPLES}")
    print(f"Total generation attempts: {total_attempts}")
    print(f"Average attempts per sample: {total_attempts / NUM_SAMPLES:.2f}")

    avg_eeg = sum(s['eeg_sim'] for s in all_scores) / len(all_scores)
    avg_gt = sum(s['gt_sim'] for s in all_scores) / len(all_scores)
    avg_combined = sum(s['combined_score'] for s in all_scores) / len(all_scores)

    print(f"Average EEG similarity: {avg_eeg:.4f}")
    print(f"Average GT similarity: {avg_gt:.4f}")
    print(f"Average combined score: {avg_combined:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Wonder3D outputs in: {wonder3d_dir / 'outputs'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

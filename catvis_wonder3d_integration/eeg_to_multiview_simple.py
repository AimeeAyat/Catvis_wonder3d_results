"""
Simplified EEG to Multi-View
Uses subprocess to call Wonder3D's original test script
"""
import sys
import os
import torch
from pathlib import Path
import yaml
from PIL import Image
import subprocess

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# CATVis imports only
from src.models.eeg_classifier import EEGClassifier
from src.models.contrastive_encoder import ContrastiveEncoder
from src.data.data_loader import CATVisDataLoader
from src.data.preprocessor import DataPreprocessor
from src.pipeline.text_retrieval import TextRetrieval


def generate_image_from_eeg(config, classifier, contrastive, eeg_data, text_retrieval):
    """Generate single image using CATVis pipeline"""
    import clip
    from diffusers import StableDiffusionPipeline
    import torch.nn.functional as F
    from torch.distributions import Beta

    device = eeg_data.device

    # Step 1: Classify
    with torch.no_grad():
        outputs, _ = classifier(eeg_data)
        predicted_idx = outputs.argmax().item()

    predicted_class = config['class_prompts'][
        list(config['class_prompts'].keys())[predicted_idx]
    ]

    # Step 2: Retrieve caption
    retrieved_captions = text_retrieval.retrieve_top_k(contrastive, eeg_data, k=1)
    caption = retrieved_captions[0]

    print(f"Predicted: {predicted_class}")
    print(f"Caption: {caption}")

    # Step 3: Generate with Stable Diffusion
    sd_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # Load CLIP for interpolation
    clip_model, _ = clip.load("ViT-L/14", device=device)

    # Encode prompts
    class_tokens = clip.tokenize([predicted_class]).to(device)
    caption_tokens = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        class_embeds = clip_model.encode_text(class_tokens).float()
        caption_embeds = clip_model.encode_text(caption_tokens).float()

    class_embeds = F.normalize(class_embeds, dim=-1)
    caption_embeds = F.normalize(caption_embeds, dim=-1)

    # Interpolate
    alpha = Beta(10, 10).sample().item()

    # Just use caption for simplicity (Wonder3D works better with concrete descriptions)
    prompt = caption

    # Generate image
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    return image, predicted_class, caption


def run_wonder3d_on_image(input_image_path, output_dir):
    """Call Wonder3D test script via subprocess"""
    wonder3d_dir = Path(__file__).parent.parent / 'Wonder3D'

    # Wonder3D expects images in a specific format
    cmd = [
        'python',
        str(wonder3d_dir / 'test_mvdiffusion_seq.py'),
        '--config', str(wonder3d_dir / 'configs' / 'mvdiffusion-joint-ortho-6views.yaml'),
        '--input_image', str(input_image_path),
        '--output_dir', str(output_dir)
    ]

    print(f"\nRunning Wonder3D...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(wonder3d_dir))

    if result.returncode != 0:
        print(f"Error running Wonder3D:")
        print(result.stderr)
        return False

    print("Wonder3D completed successfully")
    return True


def main():
    # Load config
    config_path = str(project_root / 'config' / 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CATVis models
    print("Loading CATVis models...")
    classifier = EEGClassifier(config).to(device)
    classifier_ckpt = torch.load('checkpoints/eeg_classifier_best.pth', map_location=device)
    classifier.model.load_state_dict(classifier_ckpt)
    classifier.eval()

    contrastive = ContrastiveEncoder(config).to(device)
    contrastive_ckpt = torch.load('checkpoints/contrastive_model_best.pth', map_location=device)
    contrastive.model.load_state_dict(contrastive_ckpt)
    contrastive.eval()

    # Load test data
    print("Loading test data...")
    data_loader = CATVisDataLoader(config)
    df = data_loader.get_dataset_dataframe()
    _, _, test_df = data_loader.get_train_val_test_splits(df)

    preprocessor = DataPreprocessor(config)
    test_dataset = preprocessor.create_pipeline_dataset(test_df)

    text_retrieval = TextRetrieval(config, test_df)

    # Get sample
    sample = test_dataset[0]
    eeg_data = sample['eeg'].unsqueeze(0).to(device)

    print(f"\n=== Processing EEG Sample ===")
    print(f"Ground truth: {sample['label']}")

    # Generate image
    print("\n=== Step 1: Generate image from EEG ===")
    sd_image, predicted_class, caption = generate_image_from_eeg(
        config, classifier, contrastive, eeg_data, text_retrieval
    )

    # Save intermediate
    output_dir = project_root / 'outputs' / 'eeg_multiview_simple'
    output_dir.mkdir(parents=True, exist_ok=True)

    sd_image_path = output_dir / 'generated_sd_image.png'
    sd_image.save(sd_image_path)
    print(f"Saved SD image: {sd_image_path}")

    # Run Wonder3D
    print("\n=== Step 2: Generate multi-view with Wonder3D ===")
    success = run_wonder3d_on_image(sd_image_path, output_dir)

    if success:
        print(f"\n✅ Complete! Check: {output_dir}")
    else:
        print("\n❌ Wonder3D failed. You may need to run it manually:")
        print(f"   cd Wonder3D")
        print(f"   python test_mvdiffusion_seq.py --input_image {sd_image_path}")


if __name__ == '__main__':
    main()

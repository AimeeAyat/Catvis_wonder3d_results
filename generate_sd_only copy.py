import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import os
import torch
import yaml
import subprocess
import shutil
import random
from PIL import Image
from torch.distributions import Beta

os.chdir(project_root)

from src.models.eeg_classifier import EEGClassifier
from src.models.contrastive_encoder import ContrastiveEncoder
from src.data.data_loader import CATVisDataLoader
from src.data.preprocessor import DataPreprocessor
from src.pipeline.retrieval import TextRetrieval
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline


def main():
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    print("Loading models...")
    classifier = EEGClassifier(config).to(device)
    classifier.model.load_state_dict(torch.load('checkpoints/eeg_classifier_best.pth', map_location=device))
    classifier.eval()

    contrastive = ContrastiveEncoder(config).to(device)
    contrastive.model.load_state_dict(torch.load('checkpoints/contrastive_model_best.pth', map_location=device))
    contrastive.eval()

    print("Loading data...")
    data_loader = CATVisDataLoader(config)
    df = data_loader.get_dataset_dataframe()
    _, _, test_df = data_loader.get_train_val_test_splits(df)

    preprocessor = DataPreprocessor(config)
    test_dataset = preprocessor.create_pipeline_dataset(test_df)

    text_retrieval = TextRetrieval(config, contrastive, device)
    text_retrieval.setup_retrieval_corpus(test_df)

    print("Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    output_dir = Path('outputs/eeg_multiview_batch')
    output_dir.mkdir(parents=True, exist_ok=True)

    wonder3d_dir = Path('Wonder3D')
    wonder3d_input = wonder3d_dir / 'example_images'

    indices = random.sample(range(len(test_dataset)), min(10, len(test_dataset)))

    for idx, sample_idx in enumerate(indices):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/10 (index {sample_idx})")
        print(f"{'='*60}")

        sample = test_dataset[sample_idx]
        eeg_data = sample['eeg'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs, _ = classifier(eeg_data)
            pred_idx = outputs.argmax().item()

        # Use the actual label order from raw data (not config keys which have different order)
        labels_order = data_loader.raw_data['labels']
        predicted_class = config['class_prompts'][labels_order[pred_idx]]

        retrieved = text_retrieval.retrieve_top_k_from_eeg(eeg_data.squeeze(0), k=1)
        top_caption = retrieved[0][0]

        print(f"Predicted: {predicted_class}")
        print(f"Caption: {top_caption}")
        print("Generating SD image...")

        prompt = f"{predicted_class}, {top_caption}"
        result = pipe(prompt, num_inference_steps=50, guidance_scale=7.5)

        img_name = f"sample_{idx:03d}.png"
        img_path = output_dir / img_name
        img_rgba = result.images[0].convert('RGBA')
        img_rgba.save(img_path)
        print(f"Saved: {img_path}")

        wonder3d_input_path = wonder3d_input / img_name
        shutil.copy(img_path, wonder3d_input_path)

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

    print(f"\n{'='*60}")
    print(f"Completed! Results in: {output_dir}")
    print(f"Wonder3D outputs in: {wonder3d_dir / 'outputs'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

"""
EEG Classification model for CATVis.
"""

import torch
import torch.nn as nn
from braindecode.models import EEGConformer
from typing import Dict, Any, Tuple


class EEGClassifier(nn.Module):
    """
    EEG Classifier based on EEGConformer from braindecode.
    Modified to produce CLIP-aligned 768-dimensional features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(EEGClassifier, self).__init__()
        self.config = config
        
        # Extract model parameters from config
        model_config = config['eeg_classification']
        model_params = model_config['model_params']
        
        # Create base EEGConformer model
        self.model = EEGConformer(
            n_outputs=model_config['n_outputs'],
            n_chans=model_config['n_chans'],
            n_times=model_config['n_times'],
            n_filters_time=model_params['n_filters_time'],
            filter_time_length=model_params['filter_time_length'],
            pool_time_length=model_params['pool_time_length'],
            pool_time_stride=model_params['pool_time_stride'],
            final_fc_length=model_params['final_fc_length'],
            return_features=False
        )

        # Modify final layers to produce 768-dimensional features (CLIP-aligned)
        # fc.fc[3] is Linear(256 -> 32), replace with Linear(256 -> 768)
        self.model.fc.fc[3] = nn.Linear(in_features=256, out_features=768, bias=True)
        # final_layer is Linear(32 -> 40), replace with Linear(768 -> n_outputs)
        self.model.final_layer = nn.Linear(in_features=768, out_features=model_config['n_outputs'], bias=True)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input EEG tensor of shape [batch_size, n_chans, n_times]

        Returns:
            Tuple of (classification_outputs, eeg_embeddings)
            - classification_outputs: [batch_size, n_classes]
            - eeg_embeddings: [batch_size, 768] feature embeddings
        """
        # Add channel dimension: [batch, n_chans, n_times] -> [batch, 1, n_chans, n_times]
        x = x.unsqueeze(1)

        # Forward through patch embedding and transformer
        x = self.model.patch_embedding(x)
        x = self.model.transformer(x)

        # Forward through fc to get 768-d embeddings
        embeddings = self.model.fc(x)  # [batch, 768]

        # Forward through final layer for classification
        outputs = self.model.final_layer(embeddings)  # [batch, n_classes]

        return outputs, embeddings
        
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get only the 768-dimensional embeddings.
        
        Args:
            x: Input EEG tensor of shape [batch_size, n_chans, n_times]
            
        Returns:
            eeg_embeddings: [batch_size, 768] feature embeddings
        """
        _, embeddings = self.forward(x)
        return embeddings
        
    def load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {checkpoint_path}")
        
    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}") 
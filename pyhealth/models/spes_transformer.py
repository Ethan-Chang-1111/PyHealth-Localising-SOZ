"""
Contributor: Ethan Chang
NetID: eango
Paper Title: Localising the Seizure Onset Zone from Single-Pulse Electrical Stimulation Responses with a CNN Transformer
Paper Link: https://proceedings.mlr.press/v252/norris24a.html
Description: Convolutional-Transformer encoder model implementation for SPES Seizure Onset Zone Localisation.

SPES CNN-Transformer model.

Paper: Localising the Seizure Onset Zone from Single-Pulse Electrical 
Stimulation Responses with a CNN Transformer (Norris et al. 2024).
https://proceedings.mlr.press/v252/norris24a.html

Original Code: https://github.com/norrisjamie23/Localising_SOZ_from_SPES/
"""

import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
# from torcheeg.transforms import RandomNoise
class RandomNoise:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, eeg, **kwargs):
        if self.std > 0:
            return {'eeg': eeg + torch.randn_like(eeg) * self.std}
        return {'eeg': eeg}

from pyhealth.models.base_model import BaseModel
from pyhealth.models.spes_resnet import MSResNet


class SPESResponseEncoder(nn.Module):
    """
    A neural network model for classifying responses to Single Pulse Electrical Stimulation (SPES).
    The full model incorporates both convolutional and MLP embeddings, with a transformer encoder for the final classification.
    """
    def __init__(
        self,
        mean: bool,
        std: bool,
        conv_embedding: bool = True,
        mlp_embedding: bool = True,
        dropout_rate: float = 0.5,
        num_layers: int = 2,
        embedding_dim: int = 64,
        random_channels=None,
        noise_std: float = 0.1,
        include_distance: bool = True,
    ):
        """
        Initialize the SPESResponseEncoder class.

        Args:
            mean (bool): Flag indicating whether to include mean in embedding.
            std (bool): Flag indicating whether to include standard deviation in embedding.
            conv_embedding (bool, optional): Flag indicating whether to use convolutional embedding. Defaults to True.
            mlp_embedding (bool, optional): Flag indicating whether to use MLP embedding. Defaults to True.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.5.
            num_layers (int, optional): Number of transformer encoder layers. Defaults to 2.
            embedding_dim (int, optional): Dimension of the embedding. Defaults to 64.
            random_channels (None, optional): Random channels. Defaults to None.
            noise_std (float, optional): Standard deviation of the noise to be added to the input. Defaults to 0.1.
        """
        super(SPESResponseEncoder, self).__init__()

        assert mean or std, "Either mean or std (or both) must be True for embedding."

        self.mean = mean
        self.std = std
        self.conv_embedding = conv_embedding
        self.mlp_embedding = mlp_embedding
        self.random_channels = random_channels
        self.noise_std = noise_std
        self.include_distance = include_distance
        
        self.noise = RandomNoise(std=self.noise_std)

        # Distances are optionally stripped (1 padding unit reduction)
        offset = 0 if self.include_distance else 1

        if conv_embedding:
            input_channels = self.mean + self.std
            self.msresnet = MSResNet(input_channel=input_channels, num_classes=1)
            # MSResNet output size is 768
            embedding_in = 768 + (self.mean + self.std) * (155 - offset) * mlp_embedding
        else:
            embedding_in = (self.mean + self.std) * (509 - offset)

        self.patch_to_embedding = nn.Linear(embedding_in, embedding_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.class_token = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, 1, embedding_dim)))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=embedding_dim // 8, 
            dim_feedforward=embedding_dim * 2, 
            dropout=dropout_rate, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, modes, chans, timesteps].
                              Modes corresponds to mean/std.
        """
        if self.training:
            x = self.apply_noise_and_zero_channels(x)
        
        if self.random_channels:
            distances = x[:, 0, :, 0]
            all_x = []

            for sample_idx, (single_sample, distance) in enumerate(zip(x, distances)):
                valid_rows = torch.where(distance != 0)[0]
                if len(valid_rows) == 0:
                    ts_std = single_sample[1, :, 1:]
                    valid_rows = torch.where(ts_std.sum(dim=-1) != 0)[0]

                if len(valid_rows) == 0:
                    valid_rows = torch.arange(single_sample.shape[1], device=x.device)

                if len(valid_rows) < self.random_channels:
                    idx = torch.randint(0, len(valid_rows), (self.random_channels,), device=x.device)
                
                else:
                    idx = torch.randperm(len(valid_rows), device=x.device)[:self.random_channels]
                random_channels_idx = valid_rows[idx].sort()[0]
                all_x.append(single_sample[:, random_channels_idx, :])
            # Stack processed samples and pass them through the MSResNet and the final layer.    
            x = torch.stack(all_x, dim=0)

        # Distances from the first mode 
        distances = x[:, 0, :, 0]
        key_padding_mask = self.create_key_padding_mask(distances)

        all_output = self.prepare_channels(x)
        x = self.dropout(self.patch_to_embedding(all_output))

        weight = self.class_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((weight, x), dim=1)

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # Return the class token output
        return x[:, 0]

    def apply_noise_and_zero_channels(self, x):
        """Applies noise to EEG data and zeros out random channels."""
        # Implementation of noise application and zeroing random channels
        non_zero_indices = torch.nonzero(x[:, 0, :, 0].sum(axis=0), as_tuple=False).squeeze(-1)

        if len(non_zero_indices) > 0:
            # Step 1: Uniformly sample a number from 0 to the length of non_zero_indices
            sample_size = random.randint(0, len(non_zero_indices) // 2)
            if sample_size > 0:
                # Step 2: Select a random sample of this number from non_zero_indices without replacement
                random_indices = torch.randperm(len(non_zero_indices))[:sample_size]
                random_sample = non_zero_indices[random_indices]
                # Set these to zero
                x[:, :, random_sample] = 0

        if self.noise_std > 0:
            x[:, :, :, 1:] = self.noise(eeg=x[:, :, :, 1:])['eeg']

        return x

    def create_key_padding_mask(self, distances):
        """Creates a key padding mask based on the distances tensor."""
        key_padding_mask = (distances == 0)

        # Prepend a false column for the class token
        false_column = torch.zeros(distances.size(0), 1, dtype=torch.bool, device=distances.device)
        key_padding_mask = torch.cat([false_column, key_padding_mask], dim=1)

        return key_padding_mask

    def prepare_channels(self, x):
        """
        Prepares each channel prior to embedding.
        Args:
            x (torch.Tensor): [batch_size, modes, chans, timesteps]
        """
        start_idx = 0 if self.include_distance else 1
        if self.conv_embedding:
            if self.mean:
                if self.std:
                    conv_input = x[:, :, :, 1:]
                else:
                    conv_input = x[:, :1, :, 1:]
            else:
                conv_input = x[:, 1:, :, 1:]

            batch_size, modes, chans, timesteps = conv_input.shape
            conv_input = conv_input.swapaxes(1, 2).reshape(-1, modes, timesteps)

            late_output = self.msresnet(conv_input)
            late_output = late_output.reshape(batch_size, chans, -1)

            if self.mlp_embedding:
                if self.mean:
                    if self.std:
                        all_output = torch.cat([x[:, 0, :, start_idx:155], x[:, 1, :, start_idx:155], late_output], dim=-1)
                    else:
                        all_output = torch.cat([x[:, 0, :, start_idx:155], late_output], dim=-1)
                else:
                    all_output = torch.cat([x[:, 1, :, start_idx:155], late_output], dim=-1)
            else:
                all_output = late_output
        elif self.mlp_embedding:
            if self.mean:
                if self.std:
                    all_output = torch.cat([x[:, 0, :, start_idx:], x[:, 1, :, start_idx:]], dim=-1)
                else:
                    all_output = x[:, 0, :, start_idx:]
            else:
                all_output = x[:, 1, :, start_idx:]
        
        return all_output


class SPESTransformer(BaseModel):
    """
    SPES_Transformer model for Seizure Onset Zone localisation.
    Integrates the SPESResponseEncoder directly.
    """
    def __init__(
        self,
        dataset,
        feature_keys=None,
        label_key=None,
        mode=None,
        mean=True,
        std=True,
        conv_embedding=True,
        mlp_embedding=True,
        dropout_rate=0.5,
        num_layers=2,
        embedding_dim=64,
        random_channels=None,
        noise_std=0.0,
        include_distance=True,
        **kwargs
    ):
        super(SPESTransformer, self).__init__(
            dataset=dataset,
        )
        self.feature_keys = feature_keys or ["spes_responses"]
        self.label_key = label_key or "soz_label"
        if mode is not None:
            self.mode = mode
        
        num_classes = 1

        self.encoder = SPESResponseEncoder(
            mean=mean,
            std=std,
            conv_embedding=conv_embedding,
            mlp_embedding=mlp_embedding,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            random_channels=random_channels,
            noise_std=noise_std,
            include_distance=include_distance
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, num_classes)
        )

        nn.init.xavier_uniform_(self.fc[1].weight)
        if self.fc[1].bias is not None:
            nn.init.zeros_(self.fc[1].bias)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            **kwargs: Contains 'spes_responses' tensor of shape [batch, max_C, 2, T+1]
        """
        # [batch_size, max_C, 2, T+1] -> [batch_size, 2, max_C, T+1]
        x = kwargs["spes_responses"].transpose(1, 2)

        features = self.encoder(x)
        logit = self.fc(features)

        if self.mode == "binary" and logit.shape[-1] == 1:
            logit = logit.squeeze(-1)
            
        y_true = kwargs.get(self.label_key)
        if self.mode == "binary" and y_true is not None and y_true.ndim > 1:
            y_true = y_true.squeeze(-1)

        loss_fn = self.get_loss_function()
        loss = loss_fn(logit, y_true.float() if self.mode == "binary" else y_true)
        y_prob = self.prepare_y_prob(logit)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logit,
        }

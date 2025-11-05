from torch import nn
import torch


class AdderNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class TransformerAdderNetwork(nn.Module):
    def __init__(self, input_dim=16, model_dim=32,
                 num_heads=2, num_layers=2, output_dim=8):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x).unsqueeze(1)
        x = self.input_proj(x) + self.pos_embedding

        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)

        x = x[0]
        logits = self.output_proj(x)
        return logits

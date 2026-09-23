from torch import nn


class AudioProjector(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, hidden):
        return self.layers(hidden)

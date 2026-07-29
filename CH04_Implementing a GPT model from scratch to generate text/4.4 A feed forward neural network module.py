import modulefinder
import torch
import torch.nn as nn
from gptPlaceholder import *


class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)  # Creates sample input with batch dimension 2
out = ffn(x)
print()
print(out.shape)

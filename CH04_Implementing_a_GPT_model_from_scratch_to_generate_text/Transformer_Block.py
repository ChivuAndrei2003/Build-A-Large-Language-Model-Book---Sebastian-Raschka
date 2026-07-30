import torch
import torch.nn as nn

from CH03_Coding_attention_mechanisms.multiHeadAttention import MultiHeadAttention
from CH04_Implementing_a_GPT_model_from_scratch_to_generate_text.Layer_Norm import LayerNorm
from CH04_Implementing_a_GPT_model_from_scratch_to_generate_text.feed_forward_module import FeedForward
from CH04_Implementing_a_GPT_model_from_scratch_to_generate_text.gptPlaceholder import GPT_CONFIG_124M


class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            #
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        shortcut = x  # Shortcut connection for feed forward block
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Adds the original input back

        return x


# Using the GPT_CONFIG_124M dictionary we defined earlier,
# let’s instantiate a transformer block and feed it some sample data:


if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.rand(
        2, 4, 768
    )  # Creates sample input of shape [batch_size, num_tokens, emb_dim]
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    print("\nInput shape:", x.shape)
    print("Output shape:", output.shape)

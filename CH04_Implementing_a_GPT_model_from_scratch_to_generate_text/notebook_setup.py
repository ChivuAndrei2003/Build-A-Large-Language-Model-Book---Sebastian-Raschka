"""Imports shared by the chapter 4 notebooks."""

from pathlib import Path
import sys

import torch

current_dir = Path.cwd().resolve()
project_root = next(
    (
        path
        for path in (current_dir, *current_dir.parents)
        if (path / "CH03_Coding_attention_mechanisms").is_dir()
    ),
    None,
)

if project_root is None:
    raise RuntimeError("Could not locate the project root.")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from CH03_Coding_attention_mechanisms.multiHeadAttention import MultiHeadAttention

from CH04_Implementing_a_GPT_model_from_scratch_to_generate_text.feed_forward_module import (
    FeedForward,
)
from CH04_Implementing_a_GPT_model_from_scratch_to_generate_text.gptPlaceholder import (
    GPT_CONFIG_124M,
)
from CH04_Implementing_a_GPT_model_from_scratch_to_generate_text.Layer_Norm import LayerNorm
from CH04_Implementing_a_GPT_model_from_scratch_to_generate_text.Transformer_Block import (
    TransformerBlock,
)
from CH04_Implementing_a_GPT_model_from_scratch_to_generate_text.tokenizer import (
    batch,
    tokenizer,
)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


__all__ = [
    "FeedForward",
    "GPT_CONFIG_124M",
    "LayerNorm",
    "MultiHeadAttention",
    "TransformerBlock",
    "batch",
    "generate_text_simple",
    "tokenizer",
]

from importlib.metadata import version
import tiktoken

print()
print("tiktoken version:", version("tiktoken"))


tokenizer = tiktoken.get_encoding("gpt2")

print()

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)  # prints token IDs

strings = tokenizer.decode(integers)
print(strings)

print("\n Exercise : ")
text2 = "Akwirw ier "
integers2 = tokenizer.encode(text2, allowed_special={"<|endoftext|>"})
print(integers2)

strings2 = tokenizer.decode(integers2)
print(strings2)

print()

# 2.6 Data sampling with a sliding window
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

print()

enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]
print(f"x : {x}")
print(f"y : {y}")


for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

print()
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# Listing 2.5 A dataset for batched inputs and targets

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert (
            len(token_ids) > max_length
        ), "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# Listing 2.6 A data loader to generate batches with input-with pairs


# The following code uses the GPTDatasetV1 to load the inputs in batches via a PyTorch
# DataLoader :
def create_dataloader_v1(
    txt, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0
):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


# Let's test the dataloader with a batch size of 1 for an LLM with a context size of 4:
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)
data_iter = iter(dataloader)
# The first_batch variable contains two tensors: the first tensor stores the input token
# IDS
first_batch = next(data_iter)
print(first_batch)
# and the second tensor stores the target token IDs.
second_batch = next(data_iter)
print(second_batch)

# A stride of 1 moves/shifts the input field by 1 position

print("""
      \n
      Exercise 2.2 Data loaders with different strides and context sizes :
       try to run it with different
      settings such as max_length=2 and stride=2, and max_length=8 and stride=2.
      \n""")

print("\nMax_length=2 and stride=2 : \n")
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=2, stride=2, shuffle=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)

print("\nMax_length=8 and stride=2 : \n")
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=8, stride=2, shuffle=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)


# Let’s look briefly at how we can use the data loader to sample with a batch size
# greater than 1 :

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("\nInputs:\n", inputs)
print("\nTargets:\n", targets)

# Note that we increase the stride to 4 to utilize the data set fully (we don’t skip
# a single word).
# This avoids any overlap between the batches since more overlap could lead to
# increased overfitting.


# 2.7 Creating token embeddings

# Suppose we have the following four input examples with input
# ids 2, 3, 5, and 1 (after tokenization):
input_ids = torch.tensor([2, 3, 5, 1])
# For the sake of simplicity, suppose we have a small vocabulary
# of only 6 words and we want to create embeddings of size 3:
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# This would result in a 6x3 weight matrix:
print(
    "\n", embedding_layer.weight
)  # The print statement prints the embedding layer’s underlying weight matrix:

print("\nNow, let’s apply it to a token ID to obtain the embedding vector:")
print(embedding_layer(torch.tensor([3])))

# In other words, the embedding layer is essen-
# tially a lookup operation that retrieves rows from the embedding layer’s weight matrix
# via a token ID

print("""\nLet’s now convert all four input IDs (torch.tensor([2, 3, 5, 1]))
    into a three-dimensional embedding vector : """)
print("\n", embedding_layer(input_ids))


# 2.8 Encoding word positions

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# NOTE As we can see, the token ID tensor is 8 × 4 dimensional,
# meaning that the data batch consists of eight text samples with four tokens each.
print()
# Let’s now use the embedding layer to embed these token IDs into 256-dimensionalvectors
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
print(
    """\nThe 8 × 4 × 256–dimensional tensor output shows that each token ID is now embed-
ded as a 256-dimensional vector."""
)

print("""\nFor a GPT model’s absolute embedding approach, we just need to create another
embedding layer that has the same embedding dimension as the token_embedding_
layer:""")
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("\n", pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

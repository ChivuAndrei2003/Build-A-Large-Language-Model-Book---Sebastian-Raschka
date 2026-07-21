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

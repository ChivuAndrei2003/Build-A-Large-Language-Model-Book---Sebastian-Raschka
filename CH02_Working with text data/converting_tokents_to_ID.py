from tokenize import String
import urllib.request
import re



url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")

file_path="the-verdict.txt"
urllib.request.urlretrieve(url,file_path)

# Next, we can load the the-verdict.txt file using
#    Python’s standard file reading utilities
print()
with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed)) 

print()

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# After determining that the vocabulary size is 1,130 via this code, we create the 
# vocabulary and print its first 51 entries for illustration purposes.
print()

vocab = {token:integer for integer,token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
    
print("/n Implementing a simple text tokenizer : ")

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    

print()
# Let’s instantiate a new tokenizer object 
# from the SimpleTokenizerV1 class and
# tokenize a passage from Edith Wharton’s short story 
# to try it out in practice:
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# Next, let’s see whether we can turn these token IDs back into text using the decode
# method:
print(tokenizer.decode(ids))

print()
# text = "Hello, do you like tea?"
# print(tokenizer.encode(text))


# 2.4 Adding special context tokens

"""Let’s now modify the vocabulary to include these two special tokens, <unk> and
<|endoftext|>, by adding them to our list of all unique words:"""

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["< | endoftext| >","< |unk| >"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print()
print(len(vocab.items()))
print()
# As an additional quick check, let’s 
# print the last five entries of the updated vocabulary:
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
    
print()    
# Listing 2.4 A simple text tokenizer that handles unknown words
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed
                        if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "< |unk| >" for item in preprocessed ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " < |endoftext| > ".join((text1, text2))
print(text)

print()

# Next, let’s tokenize the sample text using the
# SimpleTokenizerV2 on the vocab we
# previously created in listing 2.2:
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

# detokenize again : 
print(tokenizer.decode(tokenizer.encode(text)))
import urllib.request
import re
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")

file_path="the-verdict.txt"
urllib.request.urlretrieve(url,file_path)

# Next, we can load the the-verdict.txt file using
#    Python’s standard file reading utilities
print("\n")
with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed)) 

print("\n")
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# After determining that the vocabulary size is 1,130 via this code, we create the 
# vocabulary and print its first 51 entries for illustration purposes.
print("\n")
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
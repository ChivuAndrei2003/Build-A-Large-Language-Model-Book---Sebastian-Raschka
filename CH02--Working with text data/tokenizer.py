import urllib.request
import re
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")

file_path="the-verdict.txt"
urllib.request.urlretrieve(url,file_path)

# Next, we can load the the-verdict.txt file using
#    Python’s standard file reading utilities.
print("\n")
with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    
    
print("\nTokenizing words using standard python : ")
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)

print("\nLet’s modify the regular expression splits on whitespaces (\s), commas, and peri-ods ([,.]) : ") # type: ignore
result = re.split(r'([,.]|\s)', text)
print(result)


print("\nA small remaining problem is that the list still includes whitespace characters. \nOption-ally, we can remove these redundant characters safely as follows:: ")
result = [item for item in result if item.strip()]
print(result)

print("\n")

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

print("\nNow  that  we  have  a  basic  tokenizer  working,  let’s  apply  it  to  Edith  Wharton’s  entireshort story : ")

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed)) 


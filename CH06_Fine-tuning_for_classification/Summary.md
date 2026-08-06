- There are different strategies for fine-tuning LLMs, including classification
  fine-tuning and instruction fine-tuning.
- Classification fine-tuning involves replacing the output layer of an LLM via a
  small classification layer.
- In the case of classifying text messages as “spam” or “not spam,” the new classifi-
  cation layer consists of only two output nodes. Previously, we used the number
  of output nodes equal to the number of unique tokens in the vocabulary
  (i.e., 50,256).
- Instead of predicting the next token in the text as in pretraining, classification
  fine-tuning trains the model to output a correct class label—for example,
  “spam” or “not spam.”
- The model input for fine-tuning is text converted into token IDs, similar to
  pretraining.
- Before fine-tuning an LLM, we load the pretrained model as a base model.
- Evaluating a classification model involves calculating the classification accuracy
  (the fraction or percentage of correct predictions).
- Fine-tuning a classification model uses the same cross entropy loss function as
  when pretraining the LLM.

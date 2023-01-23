from transformers import AutoTokenizer as AT
from transformers import AutoModelForSequenceClassification as AM
import torch

tokenizer = AT.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AM.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokens = tokenizer.encode('This is sample text', return_tensors='pt')
result = model(tokens)
print(int(torch.argmax(result.logits)) + 1)

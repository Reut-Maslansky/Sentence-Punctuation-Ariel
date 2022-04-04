import os

os.add_dll_directory("C:\Dev\Sentence Punctuation\server\\venv\Lib\site-packages")
import torch

model = torch.load("model/punctuation_model.pth", map_location=torch.device('cpu'))
model.eval()

tokenizer = torch.load("model/tokenizer.pth")
# tokenizer.eval()

print(torch.cuda.is_available())

def punc(text):
    tokenized_sentence = tokenizer.encode(text)
    return torch.tensor([tokenized_sentence]).cuda
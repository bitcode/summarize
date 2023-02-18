import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

import sys
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
#from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the tokenizer
#tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
#tokenizer = BartTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

# load the model
#model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large").to(device)
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
#model = BartForConditionalGeneration.from_pretrained("philschmid/bart-large-cnn-samsum").to(device)

# move the model to the GPU
model = model.to(device)

# wrap the model with DataParallel
model = torch.nn.DataParallel(model)

# Define a custom dataset
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length):
        self.text = text
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.text, max_length=self.max_length, truncation=True, padding=True, return_tensors='pt').to(device)
        attention_mask = torch.ones(inputs["input_ids"].shape, dtype=torch.long).to(device)
        return {"input_ids": inputs["input_ids"], "attention_mask": attention_mask}

# get the text file name from the command line argument
file_name = sys.argv[1]

# read the text from the file
with open(file_name, 'r') as file:
    text = file.read()

# set the maximum length of the model
max_length = 2048
    
# Create the custom dataset instance
dataset = TextDataset(text, tokenizer, max_length)

# Define a DataLoader with the appropriate batch size
data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

# extract the original model from inside the DataParallel object
model = model.module

for data in data_loader:
    attention_mask = data['attention_mask'].view(-1, max_length)

    # summarize the text
    summary = model.generate(data['input_ids'], attention_mask=attention_mask, max_length=2048)

    # move the summary back to the CPU
    summary = summary.to('cpu')

    # decode the summary
    summary_text = tokenizer.decode(summary[0], skip_special_tokens=True)

    # write the summary to a file with the same name as the input file
    output_file_name = file_name.replace('.txt', '') + '_summary.txt'
    with open(output_file_name, 'w') as output_file:
        output_file.write(summary_text)

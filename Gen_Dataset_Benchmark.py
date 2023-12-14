import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

from Dataset import Dataset


haikuDF = pd.read_parquet('./haiku_dataset/train.parquet', engine='fastparquet')
summarizeDF = pd.read_parquet('./arxiv_dataset/train.parquet', engine = 'fastparquet')

print('Data Loaded!')

tokenizer = T5Tokenizer.from_pretrained('t5-large')

print('Tokenizer Initialized!')

summarizeDataset = Dataset(dataframe = summarizeDF,
                           prefix = 'summarize: ',
                           inputCol = 'summaries',
                           targetCol = 'titles')

summarizeDataLoader = DataLoader(summarizeDataset, batch_size = 8, shuffle = False)

haikuDataset = Dataset(dataframe = haikuDF,
                       prefix = 'write a haiku about ',
                       inputCol = 'keywords',
                       targetCol = 'text')

haikuDataLoader = DataLoader(haikuDataset, batch_size = 32, shuffle = False)

print('DataLoaders Created!')

model = T5ForConditionalGeneration.from_pretrained('t5-large')
model.to(device)
model.eval()

print('T5 Model Initialized!')

# Haiku Generation

allPredictions = []
allTargets = []

for _, data in enumerate(haikuDataLoader, 0):


    inputText = data['input_text']
    targetText = data['target_text']

    inputEncodings = tokenizer(
        inputText,
        padding = True,
        truncation = True,
        return_tensors = 'pt'
    )

    inputIDs = inputEncodings['input_ids'].squeeze().to(device, dtype = torch.long)
    inputMask = inputEncodings['attention_mask'].squeeze().to(device, dtype = torch.long)

    predIDs = model.generate(
        input_ids = inputIDs,
        attention_mask = inputMask
    )

    decodedPred = tokenizer.batch_decode(predIDs, skip_special_tokens = True)

    allPredictions.extend(decodedPred)
    allTargets.extend(targetText)


allHaikus = pd.DataFrame({'Generated Haikus': allPredictions, 'Reference Haikus': allTargets})
allHaikus.to_csv('./Results/Benchmark_Data/Gen_Ref_Haikus.csv')

print('Haiku Generation Completed!')

# Scientific Abstract Generation

allPredictions = []
allTargets = []

for _, data in enumerate(summarizeDataLoader, 0):

    inputText = data['input_text']
    targetText = data['target_text']

    inputEncodings = tokenizer(
        inputText,
        padding = True,
        truncation = True,
        return_tensors = 'pt'
    )

    inputIDs = inputEncodings['input_ids'].squeeze().to(device, dtype = torch.long)
    inputMask = inputEncodings['attention_mask'].squeeze().to(device, dtype = torch.long)

    predIDs = model.generate(
        input_ids = inputIDs,
        attention_mask = inputMask
    ).to('cpu')
    
    del inputIDs, inputMask
    torch.cuda.empty_cache()
    gc.collect()

    decodedPred = tokenizer.batch_decode(predIDs, skip_special_tokens = True)

    allPredictions.extend(decodedPred)
    allTargets.extend(targetText)

allAbstracts = pd.DataFrame({'Generated Titles': allPredictions, 'Reference Titles': allTargets})
allAbstracts.to_csv('./Results/Benchmark_Data/Gen_Ref_Titles.csv')

print('Summarization Completed!')

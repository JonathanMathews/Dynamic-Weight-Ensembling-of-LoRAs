import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
from tqdm.notebook import tqdm_notebook
from ContextGatedMOE import loralib as lora

import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

from Dataset import Dataset

# Parameters
EPOCHS = 25
LR = 3e-3
BS = 4
LORA_RANK = 16
FIGURE_PATH = './Results/CGM_Lora/'
MODEL_PATH = './Results/CGM_Lora/Models/'
BASE_PATH = './Results/CGM_Lora/Base_Model/'
BASE_CHECKPOINT = BASE_PATH + 'base_model.pt'
LORA_PATH = './Results/Lora_Finetune/Models/'
HAIKU_LORA_CHECKPOINT = LORA_PATH + 'haiku_lora_finetune.pt'
SUM_LORA_CHECKPOINT = LORA_PATH + 'title_lora_finetune.pt'
CGM_CHECKPOINT = MODEL_PATH + 'haiku_title_cgm_lora.pt'
T5_MODEL = 't5-large'

def applyLora(model, numBlocks, linearDim, rank):

    for i in range(numBlocks):

        model.encoder.block[i].layer[0].SelfAttention.q = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.encoder.block[i].layer[0].SelfAttention.k = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.encoder.block[i].layer[0].SelfAttention.v = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.encoder.block[i].layer[0].SelfAttention.o = lora.Linear(linearDim, linearDim, r = rank, bias = False)

        model.decoder.block[i].layer[0].SelfAttention.q = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[i].layer[0].SelfAttention.k = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[i].layer[0].SelfAttention.v = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[i].layer[0].SelfAttention.o = lora.Linear(linearDim, linearDim, r = rank, bias = False)

        model.decoder.block[i].layer[1].EncDecAttention.q = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[i].layer[1].EncDecAttention.k = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[i].layer[1].EncDecAttention.v = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[i].layer[1].EncDecAttention.o = lora.Linear(linearDim, linearDim, r = rank, bias = False)

    lora.mark_only_lora_as_trainable(model)

def apply_lora_dwm(model, numBlocks, haikuLoraWeights, sumLoraWeights):

    # Apply LoRA to all attention matrices in the transformer block: q,k,v,o
    for i in range(numBlocks):
        
        lora_A_weights = [haikuLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.q.lora_A'],
                          sumLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.q.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.q.lora_B'],
                          sumLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.q.lora_B']]
        model.encoder.block[i].layer[0].SelfAttention.q = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

        lora_A_weights = [haikuLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.k.lora_A'],
                          sumLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.k.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.k.lora_B'],
                          sumLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.k.lora_B']]
        model.encoder.block[i].layer[0].SelfAttention.k = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

        lora_A_weights = [haikuLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.v.lora_A'],
                          sumLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.v.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.v.lora_B'],
                          sumLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.v.lora_B']]
        model.encoder.block[i].layer[0].SelfAttention.v = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

        lora_A_weights = [haikuLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.o.lora_A'],
                          sumLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.o.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.o.lora_B'],
                          sumLoraWeights[f'encoder.block.{i}.layer.0.SelfAttention.o.lora_B']]
        model.encoder.block[i].layer[0].SelfAttention.o = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

        lora_A_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.q.lora_A'],
                          sumLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.q.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.q.lora_B'],
                          sumLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.q.lora_B']]
        model.decoder.block[i].layer[0].SelfAttention.q = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

        lora_A_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.k.lora_A'],
                          sumLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.k.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.k.lora_B'],
                          sumLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.k.lora_B']]
        model.decoder.block[i].layer[0].SelfAttention.k = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

        lora_A_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.v.lora_A'],
                          sumLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.v.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.v.lora_B'],
                          sumLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.v.lora_B']]
        model.decoder.block[i].layer[0].SelfAttention.v = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

        lora_A_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.o.lora_A'],
                          sumLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.o.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.o.lora_B'],
                          sumLoraWeights[f'decoder.block.{i}.layer.0.SelfAttention.o.lora_B']]
        model.decoder.block[i].layer[0].SelfAttention.o = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

        lora_A_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.q.lora_A'],
                          sumLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.q.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.q.lora_B'],
                          sumLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.q.lora_B']]
        model.decoder.block[i].layer[1].EncDecAttention.q = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

        lora_A_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.k.lora_A'],
                          sumLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.k.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.k.lora_B'],
                          sumLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.k.lora_B']]
        model.decoder.block[i].layer[1].EncDecAttention.k = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

        lora_A_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.v.lora_A'],
                          sumLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.v.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.v.lora_B'],
                          sumLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.v.lora_B']]
        model.decoder.block[i].layer[1].EncDecAttention.v = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

        lora_A_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.o.lora_A'],
                          sumLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.o.lora_A']]
        lora_B_weights = [haikuLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.o.lora_B'],
                          sumLoraWeights[f'decoder.block.{i}.layer.1.EncDecAttention.o.lora_B']]
        model.decoder.block[i].layer[1].EncDecAttention.o = lora.DWMLinear(lora_A_weights, lora_B_weights, bias=False)

def runOneEpoch(trainFlag, haikuDL, sumDL, model, optimizer):

    torch.set_grad_enabled(trainFlag)
    model.train() if trainFlag else model.eval()
    
    losses = []

    for multiDataBatch in tqdm_notebook(zip(haikuDL, sumDL), desc="Batches", total=len(sumDL)):
        
        for batch in multiDataBatch:

            inputText = batch['input_text']
            targetText = batch['target_text']

            inputEncodings = tokenizer(
                inputText,
                padding = True,
                truncation = True,
                return_tensors = 'pt'
            )

            targetEncodings = tokenizer(
                text_target = targetText,
                padding = True,
                truncation = True,
                return_tensors = 'pt'
            )

            inputIDs = inputEncodings['input_ids'].squeeze().to(device, dtype = torch.long)
            inputMask = inputEncodings['attention_mask'].squeeze().to(device, dtype = torch.long)

            targetIDs = targetEncodings['input_ids'].squeeze().to(device, dtype = torch.long)
            targetIDs[targetIDs == tokenizer.pad_token_id] = -100

            del inputEncodings, targetEncodings
            gc.collect()
            
            loss = model(
                input_ids = inputIDs,
                attention_mask = inputMask,
                labels = targetIDs
            ).loss

            del inputIDs, inputMask, targetIDs
            torch.cuda.empty_cache()
            gc.collect()

            if trainFlag:
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            losses.append(loss.item())

    return np.mean(losses)

def train(haikuTrain, haikuVal, sumTrain, sumVal, model, optimizer, checkpoint):
    
    bestValLoss = np.inf
    patience = 10
    patienceCounter = patience

    trainLosses = []
    valLosses = []

    for epoch in range(EPOCHS):

        print('Epoch', epoch + 1)

        trainLoss = runOneEpoch(True, haikuTrain, sumTrain, model, optimizer)
        valLoss = runOneEpoch(False, haikuVal, sumVal, model, optimizer)

        trainLosses.append(trainLoss)
        valLosses.append(valLoss)
        
        if valLoss <= bestValLoss:
            
            torch.save(lora.cgm_state_dict(model), checkpoint)
            bestValLoss = valLoss
            patienceCounter = patience

        else:
            
            patienceCounter -= 1

            if patienceCounter <= 0:
                
                break

        print(f'Training Loss = {trainLoss} | Validation Loss = {valLoss}')
    
    return trainLosses, valLosses

def test(dataloader, model):

    torch.set_grad_enabled(False)
    model.eval()

    allPredictions = []
    allTargets = []

    for _, data in enumerate(dataloader, 0):

        inputText = data['input_text']
        targetText = data['target_text']

        inputEncodings = tokenizer(
            inputText,
            padding = True,
            truncation = True,
            return_tensors = 'pt'
        )

        inputIDs = inputEncodings['input_ids'].to(device, dtype = torch.long)
        inputMask = inputEncodings['attention_mask'].to(device, dtype = torch.long)

        predIDs = model.generate(
            input_ids = inputIDs,
            attention_mask = inputMask
        )

        del inputIDs, inputMask
        torch.cuda.empty_cache()
        gc.collect()

        decodedPred = tokenizer.batch_decode(predIDs, skip_special_tokens = True)

        allPredictions.extend(decodedPred)
        allTargets.extend(targetText)

    predTargetDict = {
        'Predictions': allPredictions,
        'Targets': allTargets
    }

    return predTargetDict

# Create tokenizer for T5-base
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)

print('Tokenizer Created!')

# Read Data and equalize number of samples across datasets
haikuDF = pd.read_parquet('./haiku_dataset/train.parquet', engine='fastparquet')[:4096]
summarizeDF = pd.read_parquet('./arxiv_dataset/train.parquet', engine = 'fastparquet')[:4096]

print('Data Loaded!')

haikuTrainDF, haikuValDF = train_test_split(haikuDF, test_size = 0.3, shuffle = True)
haikuValDF, haikuTestDF = train_test_split(haikuValDF, test_size = 0.5, shuffle = True)

haikuTrainDF.reset_index(drop = True, inplace = True)
haikuValDF.reset_index(drop = True, inplace = True)
haikuTestDF.reset_index(drop = True, inplace = True)

sumTrainDF, sumValDF = train_test_split(summarizeDF, test_size = 0.3, shuffle = True)
sumValDF, sumTestDF = train_test_split(sumValDF, test_size = 0.5, shuffle = True)

sumTrainDF.reset_index(drop = True, inplace = True)
sumValDF.reset_index(drop = True, inplace = True)
sumTestDF.reset_index(drop = True, inplace = True)

print('Split into Train/Val/Test (0.7/0.15/0.15) Sets!')

haikuTrain = Dataset(dataframe = haikuTrainDF,
                     prefix = 'write a haiku about ',
                     inputCol = 'keywords',
                     targetCol = 'text')

haikuVal = Dataset(dataframe = haikuValDF,
                   prefix = 'write a haiku about ',
                   inputCol = 'keywords',
                   targetCol = 'text')

haikuTest = Dataset(dataframe = haikuTestDF,
                    prefix = 'write a haiku about ',
                    inputCol = 'keywords',
                    targetCol = 'text')

sumTrain = Dataset(dataframe = sumTrainDF,
                   prefix = 'summarize: ',
                   inputCol = 'summaries',
                   targetCol = 'titles')

sumVal = Dataset(dataframe = sumValDF,
                   prefix = 'summarize: ',
                   inputCol = 'summaries',
                   targetCol = 'titles')

sumTest = Dataset(dataframe = sumTestDF,
                    prefix = 'summarize: ',
                    inputCol = 'summaries',
                    targetCol = 'titles')

haikuTrainLoader = DataLoader(dataset = haikuTrain, batch_size = BS, shuffle = True)
haikuValLoader = DataLoader(dataset = haikuVal, batch_size = BS, shuffle = False)
haikuTestLoader = DataLoader(dataset = haikuTest, batch_size = BS, shuffle = False)

sumTrainLoader = DataLoader(dataset = sumTrain, batch_size = BS, shuffle = True)
sumValLoader = DataLoader(dataset = sumVal, batch_size = BS, shuffle = False)
sumTestLoader = DataLoader(dataset = sumTest, batch_size = BS, shuffle = False)

print('Data Loaders Created!')

model = T5ForConditionalGeneration.from_pretrained(T5_MODEL)
torch.save(model.state_dict(), BASE_CHECKPOINT)

model.to(device)

numBlocks = len(model.encoder.block)

print('Base Model Loaded!')

haikuLoraWeights = torch.load(HAIKU_LORA_CHECKPOINT)
sumLoraWeights = torch.load(SUM_LORA_CHECKPOINT)

print('Haiku and Summarization LoRA Weights Loaded')

apply_lora_dwm(model, numBlocks, haikuLoraWeights, sumLoraWeights)
model.load_state_dict(torch.load(BASE_CHECKPOINT), strict=False)

model.to(device)

for name, param in model.named_parameters():

    if 'context_gated_mixing' in name:

        param.requires_grad = True

    else:

        param.requires_grad = False

print('Context-Gated Mixing Layers Initialized!')

optimizer = optim.AdamW(params = model.parameters(), lr=LR, eps=1e-8, weight_decay=0.0)

print('AdamW Optimizer Created!')

trainLosses, valLosses = train(haikuTrainLoader,
                               haikuValLoader,
                               sumTrainLoader,
                               sumValLoader,
                               model,
                               optimizer,
                               CGM_CHECKPOINT)

print('CGM Model Trained and Saved!')

haikuPred = test(haikuTestLoader, model)

allHaikus = pd.DataFrame({'Generated Haikus': haikuPred['Predictions'], 
                          'Reference Haikus': haikuPred['Targets']})
allHaikus.to_csv(FIGURE_PATH + 'CGM_Lora_Haikus.csv')

print('Haiku Generation Complete!')

sumPred = test(sumTestLoader, model)

allSum = pd.DataFrame({'Generated Titles': sumPred['Predictions'],
                       'Reference Titles': sumPred['Targets']})
allSum.to_csv(FIGURE_PATH + 'CGM_Lora_Titles.csv')

print('Summarization Complete!')

plt.figure()

plt.plot(trainLosses, label = 'Training')
plt.plot(valLosses, label = 'Validation')

plt.title('Context-Gated Mixing - Haiku & Title Generation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(FIGURE_PATH + 'CGM_Lora_Losses.png')
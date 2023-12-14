import numpy as np
import pandas as pd
import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
from ContextGatedMOE import loralib as lora

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

from Dataset import Dataset

# Parameters
EPOCHS = 25
LR = 1e-3
HAIKU_BS = 16
SUM_BS = 2
HAIKU_STEPS = 4
SUM_STEPS = 32
LORA_RANK = 16
FIGURE_PATH = './Results/Lora_Finetune/'
MODEL_PATH = './Results/Lora_Finetune/Models/'
BASE_MODEL_PATH = './Results/Lora_Finetune/Base_Models/'
HAIKU_BASE_CHECKPOINT = BASE_MODEL_PATH + 'haiku_base.pt'
SUM_BASE_CHECKPOINT = BASE_MODEL_PATH + 'title_base.pt'
HAIKU_CHECKPOINT = MODEL_PATH + 'haiku_lora_finetune.pt'
SUM_CHECKPOINT = MODEL_PATH + 'title_lora_finetune.pt'
T5_MODEL = 't5-large'

def applyLora(model, numBlocks, linearDim, rank):

    for block in range(numBlocks):

        model.encoder.block[block].layer[0].SelfAttention.q = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.encoder.block[block].layer[0].SelfAttention.k = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.encoder.block[block].layer[0].SelfAttention.v = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.encoder.block[block].layer[0].SelfAttention.o = lora.Linear(linearDim, linearDim, r = rank, bias = False)

        model.decoder.block[block].layer[0].SelfAttention.q = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[block].layer[0].SelfAttention.k = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[block].layer[0].SelfAttention.v = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[block].layer[0].SelfAttention.o = lora.Linear(linearDim, linearDim, r = rank, bias = False)

        model.decoder.block[block].layer[1].EncDecAttention.q = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[block].layer[1].EncDecAttention.k = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[block].layer[1].EncDecAttention.v = lora.Linear(linearDim, linearDim, r = rank, bias = False)
        model.decoder.block[block].layer[1].EncDecAttention.o = lora.Linear(linearDim, linearDim, r = rank, bias = False)

    lora.mark_only_lora_as_trainable(model)

    return model

def runOneEpoch(trainFlag, dataloader, model, optimizer, accumulateSteps):

    torch.set_grad_enabled(trainFlag)
    model.train() if trainFlag else model.eval()
    
    losses = []

    for index, data in enumerate(dataloader, 0):

        inputText = data['input_text']
        targetText = data['target_text']

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
            
            loss = loss / accumulateSteps
            loss.backward()

            if ((index + 1) % accumulateSteps == 0) or (index + 1 == len(dataloader)):

                optimizer.step()
                optimizer.zero_grad()
        
        losses.append(loss.item())

    return np.mean(losses)

def train(trainLoader, valLoader, model, optimizer, accumulateSteps, baseCheckpoint, checkpoint):
    
    bestValLoss = np.inf
    patience = 5
    patienceCounter = patience

    trainLosses = []
    valLosses = []

    for epoch in range(EPOCHS):

        print('Epoch', epoch + 1)

        trainLoss = runOneEpoch(True, trainLoader, model, optimizer, accumulateSteps)
        valLoss = runOneEpoch(False, valLoader, model, optimizer, accumulateSteps)

        trainLosses.append(trainLoss)
        valLosses.append(valLoss)
        
        if valLoss <= bestValLoss:
            
            torch.save(lora.lora_state_dict(model), checkpoint)
            bestValLoss = valLoss
            patienceCounter = patience

        else:
            
            patienceCounter -= 1

            if patienceCounter <= 0:
                
                model.load_state_dict(torch.load(baseCheckpoint), strict = False)
                model.load_state_dict(torch.load(checkpoint), strict = False)
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
haikuDF = pd.read_parquet('./haiku_dataset/train.parquet', engine='fastparquet')
summarizeDF = pd.read_parquet('./arxiv_dataset/train.parquet', engine = 'fastparquet')

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

haikuTrainLoader = DataLoader(dataset = haikuTrain, batch_size = HAIKU_BS, shuffle = True)
haikuValLoader = DataLoader(dataset = haikuVal, batch_size = HAIKU_BS, shuffle = False)
haikuTestLoader = DataLoader(dataset = haikuTest, batch_size = HAIKU_BS, shuffle = False)

sumTrainLoader = DataLoader(dataset = sumTrain, batch_size = SUM_BS, shuffle = True)
sumValLoader = DataLoader(dataset = sumVal, batch_size = SUM_BS, shuffle = False)
sumTestLoader = DataLoader(dataset = sumTest, batch_size = SUM_BS, shuffle = False)

print('Data Loaders Created!')

haikuBaseModel = T5ForConditionalGeneration.from_pretrained(T5_MODEL)

torch.save(haikuBaseModel.state_dict(), HAIKU_BASE_CHECKPOINT)

modelBlocks = len(haikuBaseModel.encoder.block)
linearFeat = haikuBaseModel.encoder.block[0].layer[0].SelfAttention.q.in_features

haikuModel = applyLora(haikuBaseModel,
                       numBlocks = modelBlocks,
                       linearDim = linearFeat,
                       rank = LORA_RANK)
haikuModel.to(device)

print('Haiku T5-Model Created')

haikuOptim = optim.Adafactor(params = haikuModel.parameters(), lr = LR)

print('Haiku AdaFactor Optimizer Created!')

haikuTrainLosses, haikuValLosses = train(haikuTrainLoader,
                                         haikuValLoader,
                                         haikuModel,
                                         haikuOptim,
                                         HAIKU_STEPS,
                                         HAIKU_BASE_CHECKPOINT,
                                         HAIKU_CHECKPOINT)

print('Haiku model finetuned and saved!')

haikuPred = test(haikuTestLoader, haikuModel)

allHaikus = pd.DataFrame({'Generated Haikus': haikuPred['Predictions'], 
                          'Reference Haikus': haikuPred['Targets']})
allHaikus.to_csv(FIGURE_PATH + 'Lora_Haikus.csv')

print('Haiku Generation Complete!')

del haikuModel
torch.cuda.empty_cache()
gc.collect()

sumBaseModel = T5ForConditionalGeneration.from_pretrained(T5_MODEL)

torch.save(sumBaseModel.state_dict(), SUM_BASE_CHECKPOINT)

modelBlocks = len(sumBaseModel.encoder.block)
linearFeat = sumBaseModel.encoder.block[0].layer[0].SelfAttention.q.in_features

sumModel = applyLora(sumBaseModel,
                     numBlocks = modelBlocks,
                     linearDim = linearFeat,
                     rank = LORA_RANK)

sumModel.to(device)

print('Summarization T5-Model Created')

sumOptim = optim.Adafactor(params = sumModel.parameters(), lr = LR)

print('Summarization AdaFactor Optimizer Created!')

sumTrainLosses, sumValLosses = train(sumTrainLoader,
                                     sumValLoader,
                                     sumModel,
                                     sumOptim,
                                     SUM_STEPS,
                                     SUM_BASE_CHECKPOINT,
                                     SUM_CHECKPOINT)

print('Summarization model finetuned and saved!')

sumPred = test(sumTestLoader, sumModel)

allSum = pd.DataFrame({'Generated Titles': sumPred['Predictions'],
                       'Reference Titles': sumPred['Targets']})
allSum.to_csv(FIGURE_PATH + 'Lora_Titles.csv')

print('Summarization Complete!')

del sumModel
torch.cuda.empty_cache()
gc.collect()

# Plot losses of both finetuned models

plt.figure()

plt.plot(haikuTrainLosses, label = 'Training')
plt.plot(haikuValLosses, label = 'Validation')

plt.title('Finetuned LoRA Haiku Generation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(FIGURE_PATH + 'Lora_Haiku_Losses.png')

plt.figure()

plt.plot(sumTrainLosses, label = 'Training')
plt.plot(sumValLosses, label = 'Validation')

plt.title('Finetuned LoRA Title Generation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(FIGURE_PATH + 'Lora_Titles_Losses.png')
import numpy as np
import pandas as pd
from ignite.metrics import Rouge
from torchmetrics.text.bert import BERTScore
import torch
import syllables

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def count_syllables(predHaikus):

    totalHaikus = len(predHaikus)

    sentenceSplitHaikus = predHaikus.apply(lambda x: x.split('/'))

    threeSentences = sentenceSplitHaikus.apply(lambda x: len(x) == 3)
    fullHaikus  = sentenceSplitHaikus[threeSentences].reset_index(drop = True)

    firstSentence = fullHaikus.apply(lambda x: x[0].split())
    secondSentence = fullHaikus.apply(lambda x: x[1].split())
    thirdSentence = fullHaikus.apply(lambda x: x[2].split())

    firstSentenceList = firstSentence.to_list()
    firstSentSyllables = []
    for sentence in firstSentenceList:

        totalSyllables = 0

        for word in sentence:

            totalSyllables += syllables.estimate(word)

        firstSentSyllables.append(totalSyllables)
    
    secondSentenceList = secondSentence.to_list()
    secSentSyllables = []
    for sentence in secondSentenceList:

        totalSyllables = 0

        for word in sentence:

            totalSyllables += syllables.estimate(word)

        secSentSyllables.append(totalSyllables)

    thirdSentenceList = thirdSentence.to_list()
    thirdSentSyllables = []
    for sentence in thirdSentenceList:

        totalSyllables = 0

        for word in sentence:

            totalSyllables += syllables.estimate(word)

        thirdSentSyllables.append(totalSyllables)

    allHaikuSyllables = pd.DataFrame({'First': firstSentSyllables, 'Second': secSentSyllables, 'Third': thirdSentSyllables})

    totalCorrect = 0
    for index, i in enumerate(allHaikuSyllables.itertuples(index = False, name = None)):

        if i == (5, 7, 5):
            
            totalCorrect += 1

    percentCorrect = totalCorrect / totalHaikus * 100
    
    return totalCorrect, percentCorrect


def evaluate(csvFile, predictCol, targetCol, haiku):

    metric = Rouge(variants = [1, 2, 'L'], device = device)
    allGenerations = pd.read_csv(csvFile)
    allGenerations = allGenerations.dropna(how = 'any', axis = 0) 

    predGen = allGenerations[predictCol]
    targetGen = allGenerations[targetCol]

    predictions = predGen.apply(str).apply(str.split).to_list()
    targets = targetGen.apply(str).apply(str.split).to_list()

    metric.update((predictions, targets))

    bertScore = BERTScore(verbose = True).to(device)

    allF1 = bertScore(predGen.to_list(), targetGen.to_list())['f1']
    f1 = torch.mean(allF1).item()

    if haiku:

        numCorrect, percentCorrect = count_syllables(predGen)

        return metric.compute(), (numCorrect, percentCorrect), f1

    return metric.compute(), f1


benchmarkHaiku, correctBench, bHaikuBS = evaluate(csvFile = './Results/Benchmark_Data/Gen_Ref_Haikus.csv', 
                                        predictCol = 'Generated Haikus', 
                                        targetCol = 'Reference Haikus', 
                                        haiku = True)

benchmarkSum, bSumBS = evaluate(csvFile = './Results/Benchmark_Data/Gen_Ref_Titles.csv',
                                predictCol = 'Generated Titles',
                                targetCol = 'Reference Titles',
                                haiku = False)

finetuneHaiku, correctFine, fHaikuBS = evaluate(csvFile = './Results/Finetune/Finetuned_Haikus.csv',
                                    predictCol = 'Generated Haikus',
                                    targetCol = 'Reference Haikus',
                                    haiku = True)

finetuneSum, fSumBS = evaluate(csvFile = './Results/Finetune/Finetuned_Titles.csv',
                                predictCol = 'Generated Titles',
                                targetCol = 'Reference Titles',
                                haiku = False)

loraHaiku, correctLora, LHaikuBS = evaluate(csvFile = './Results/Lora_Finetune/Lora_Haikus.csv',
                                    predictCol = 'Generated Haikus',
                                    targetCol = 'Reference Haikus',
                                    haiku = True)

loraSum, LSumBS = evaluate(csvFile = './Results/Lora_Finetune/Lora_Titles.csv',
                                predictCol = 'Generated Titles',
                                targetCol = 'Reference Titles',
                                haiku = False)

cgmHaiku, correctCGM, CGMHaikuBS = evaluate(csvFile = './Results/CGM_Lora/CGM_Lora_Haikus.csv',
                                    predictCol = 'Generated Haikus',
                                    targetCol = 'Reference Haikus',
                                    haiku = True)

cgmSum, CGMSumBS = evaluate(csvFile = './Results/CGM_Lora/CGM_Lora_Titles.csv',
                                predictCol = 'Generated Titles',
                                targetCol = 'Reference Titles',
                                haiku = False)

scoresDF = pd.DataFrame([benchmarkHaiku, finetuneHaiku, loraHaiku, cgmHaiku],
                        index = ['B-Haiku', 'F-Haiku', 'L-Haiku', 'CGM-Haiku'])
scoresDF = scoresDF[['Rouge-1-F', 'Rouge-2-F', 'Rouge-L-F']]
scoresDF = scoresDF.mul(100)

scoresDF.to_csv('./Results/Rouge_Haiku.csv')

scoresDF = pd.DataFrame([benchmarkSum, finetuneSum, loraSum, cgmSum],
                        index = ['B-Title', 'F-Title', 'L-Title', 'CGM-Title'])
scoresDF = scoresDF[['Rouge-1-F', 'Rouge-2-F', 'Rouge-L-F']]
scoresDF = scoresDF.mul(100)

scoresDF.to_csv('./Results/Rouge_Title.csv')

bertScoresSeries = pd.Series(data = [bHaikuBS, fHaikuBS, LHaikuBS, CGMHaikuBS],
                       index = ['B-Haiku', 'F-Haiku', 'L-Haiku', 'CGM-Haiku'])

bertScoresSeries.to_csv('./Results/BertScore_Haiku.csv')

bertScoresSeries = pd.Series(data = [bSumBS, fSumBS, LSumBS, CGMSumBS],
                       index = ['B-Title', 'F-Title', 'L-Title', 'CGM-Title'])

bertScoresSeries.to_csv('./Results/BertScore_Title.csv')

correctSyllables = [correctBench[0], correctFine[0], correctLora[0], correctCGM[0]]
percentSyllables = [correctBench[1], correctFine[1], correctLora[1], correctCGM[1]]

syllablesDF = pd.DataFrame({'Correct Number of Syllables': correctSyllables,
                            'Percentage Correct Number Syllables': percentSyllables})

syllablesDF.to_csv('./Results/Syllables_Haiku.csv')
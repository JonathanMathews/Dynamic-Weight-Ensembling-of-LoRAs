# Rouge-1/2/L

class Dataset:

    def __init__(self, dataframe, prefix, inputCol, targetCol):

        self.dataframe = dataframe
        self.inputCol = inputCol
        self.targetCol = targetCol

        self.inputSeries = self.dataframe[inputCol].map(lambda x: prefix + x)
        self.targetSeries = self.dataframe[targetCol]

    def __len__(self):

        return len(self.inputSeries)

    def __getitem__(self, index):

        inputText = str(self.inputSeries[index])
        inputText = " ".join(inputText.split())

        targetText = str(self.targetSeries[index])
        targetText = " ".join(targetText.split())

        textDict = {
            'input_text': inputText,
            'target_text': targetText
        }

        return textDict
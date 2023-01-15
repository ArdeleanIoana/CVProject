import torch
import json
class FeedData:
    def __init__(self):
        self.currentTrainRead = 0
        self.currentTestRead = 0
        self.trainMaxim = 600  # 600 is the number of training samples
        self.epochs = 0
    def getEpochs(self):
        return self.epochs

    # [1 , 0 ] safe
    # [0, 1 ] unsafe
    def feedBatchFromFile(self,batchSize):
        f = open('trainingData.json')
        jsonTrainData = json.load(f)
        batch = jsonTrainData[self.currentTrainRead:(self.currentTrainRead+batchSize)]
        gifs = [torch.tensor([[y] for y in x["frames"]]) for x in batch]
        gifs = torch.stack(gifs)
        categories = [x["category"] for x in batch]
        categories = [ torch.Tensor([1,0]) if x == "safe" else torch.Tensor([0,1]) for x in categories]
        categories = torch.stack(categories)
        self.currentTrainRead += batchSize
        if self.currentTrainRead >= self.trainMaxim:
            self.currentTrainRead = self.trainMaxim - self.currentTrainRead
            self.epochs += 1
        f.close()
        return gifs, categories
    def feedNextTest(self):
        f = open('testData.json')
        jsonTestData = json.load(f)
        gif = jsonTestData[self.currentTestRead]["frames"]
        category = jsonTestData[self.currentTestRead]["category"]
        self.currentTestRead += 1
        f.close()
        return category, torch.tensor([[[x] for x in gif]])
import cv2
import numpy as np
import torch
import pandas as pd


class FeedData:
    def __init__(self):
        self.max_nr_frames = 316 #last result from the countFrame functions, represents the the maximum number of frames a gif has in our dataset
        self.trainCSV = pd.read_csv("train.csv")
        self.testCSV = pd.read_csv("test.csv")
        self.currentTrainRead = 0
        self.currentTestRead = 0
        self.trainMaxim = 600 #number of training samples

    def gifToListOfAvgLumi(self,gif):
        fps = gif.get(cv2.CAP_PROP_FPS)
        frames = []
        ret, frame = gif.read()  # ret=True if it finds a frame else False.
        while ret:
            # read next frame
            ret, frame = gif.read()
            if isinstance(frame, (np.ndarray, np.generic)):
                frames.append(self.frameToAVGLuminance(frame))
        return frames, fps
    def add_padding(self, framesLumiList):
        required_pad = self.max_nr_frames - len(framesLumiList)
        last_frame_lumi = framesLumiList[-1]
        padding = [last_frame_lumi for x in range(required_pad)]
        return framesLumiList + padding

    def gifToTensor(self, gif):
        framesLumi , fps = self.gifToListOfAvgLumi(gif)
        #framesLumi = self.add_padding(framesLumi)
        list = [fps] + framesLumi
        list = [[[int(x)]] for x in list]
        tens = torch.Tensor(list)
        return tens

    def rgbToLuminance(self, red,green,blue):
        return int(0.2126 * red + 0.7152 * green + 0.0722 * blue)
    #BGR
    def frameToHistogram(self, frame):
        dict = {}
        listFrame = np.ndarray.tolist(frame)
        for row in listFrame:
            for col in row:
                lumi = self.rgbToLuminance(col[2], col[1], col[0])
                if lumi not in dict.keys():
                    dict.update({lumi: 1})
                else:
                    dict[lumi] = dict[lumi] + 1
        return dict

    def frameToAVGLuminance(self, frame):
        avg = 0
        listFrame = np.ndarray.tolist(frame)
        for row in listFrame:
            for col in row:
                lumi = self.rgbToLuminance(col[2], col[1], col[0])
                avg = avg + lumi

        avg = avg / (len(listFrame) * len(listFrame[0]))
        return avg

    # [1 , 0 ] safe
    # [0, 1 ] unsafe
    def feedNextTrain(self):
        if self.currentTrainRead == self.trainMaxim:
            self.currentTrainRead = 0
        path = self.trainCSV["video_name"].values.tolist()[self.currentTrainRead]
        category = self.trainCSV["tag"].values.tolist()[self.currentTrainRead]
        if category == "safe":
            category_tensor = torch.Tensor([[1,0]])
        else:
            category_tensor = torch.Tensor([[0,1]])
        self.currentTrainRead += 1
        gif = cv2.VideoCapture(path)
        return category, path,category_tensor,  self.gifToTensor(gif)

    def feedNextTest(self):
        path = self.testCSV["video_name"].values.tolist()[self.currentTestRead]
        category = self.testCSV["tag"].values.tolist()[self.currentTestRead]
        if category == "safe":
            category_tensor = torch.Tensor([[1,0]])
        else:
            category_tensor = torch.Tensor([[0,1]])
        self.currentTestRead += 1
        gif = cv2.VideoCapture(path)
        return category, self.gifToTensor(gif)



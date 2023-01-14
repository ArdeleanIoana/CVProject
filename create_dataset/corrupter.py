import random

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

class Corrupter:
    def __init__(self, filenameTemp, sourceFolder, destinationFolder):
        self.filenameTemp = filenameTemp
        self.destinationFolder = destinationFolder
        self.sourceFolder = sourceFolder
    def corruptNGifs(self, count):
        for x in range(count):
            gif = cv2.VideoCapture(self.sourceFolder + "\\" + self.filenameTemp + str(x) + ".gif")
            self.pickCorruption(gif,x)

    def pickCorruption(self,gif, count):
        choice = random.choice([0,1])
        if choice == 0:
            self.redCorruption(gif,count)
        else:
            self.blackWhiteCorruption(gif,count)
    def redCorruption(self,gif, count):
        try:
            fps = gif.get(cv2.CAP_PROP_FPS)
            frames = self.gifAsListOfFrames(gif)
            framesCount = len(frames)
            distance = random.randrange(1,(fps - 3)//2 + 1)
            redLength = random.randrange(1,(fps - distance * 2)//3 + 1)
            repetition = random.randrange(3,framesCount // (redLength + distance) + 1)
            whereStart = random.randrange(0, framesCount-(distance+redLength)*repetition + 1)
        except:
            print("fps too low to randomize, setting default parameters")
            repetition = 3
            distance = 1
            redLength = 1
            whereStart = 0
        finally:
            for i in range(repetition):
                for j in range(redLength):
                    frames.insert(whereStart,self.redFrame(frames[0].shape))
                    whereStart +=1
                whereStart = whereStart + distance
            self.saveGif(frames,count,fps)


    def blackWhiteCorruption(self,gif,count):
        frames = self.gifAsListOfFrames(gif)
        framesCount = len(frames)
        where = random.randrange(0,framesCount)
        fps = gif.get(cv2.CAP_PROP_FPS)
        try:
            flickerCount = random.randrange(3,10)
            repetitionBlack = random.randrange(1,fps//3 + 1)
            repetitionWhite = random.randrange(1, fps//(3 * repetitionBlack) + 1)
        except:
            print("fps too low to randomize, setting default parameters - blackWhite")
            flickerCount = 3
            repetitionWhite = 1
            repetitionBlack = 1
        finally:
            corruption = self.blackWhiteFilcker(flickerCount,repetitionWhite,repetitionBlack,frames[0].shape)
            corruptedframes = frames[:where] + corruption + frames[where:]
            self.saveGif(corruptedframes,count,fps)
    def gifAsListOfFrames(self, gif):
        #fps = gif.get(cv2.CAP_PROP_FPS)
        frames = []
        ret, frame = gif.read()  # ret=True if it finds a frame else False.
        while ret:
            # read next frame
            ret, frame = gif.read()
            if isinstance(frame, (np.ndarray, np.generic)):
                frames.append(frame)
        return frames
    def saveGif(self, frames,count,originalFps):
        originalFps += 1
        print("Saving GIF file")
        with imageio.get_writer(self.destinationFolder + "\\" + self.filenameTemp + str(count) + ".gif",
                                mode="I",fps=originalFps) as writer:
            for idx, frame in enumerate(frames):
                if isinstance(frame, (np.ndarray, np.generic)):
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(rgb_frame)
    #BGR
    def redFrame(self,frameShape):
        redFrame = [[[0,0,255] for x in range(frameShape[1])] for y in range(frameShape[0])]
        npRedFrame = np.array(redFrame, dtype=np.uint8)
        return npRedFrame

    def whiteFrame(self,frameShape):
        whiteFrame = [[[255, 255, 255] for x in range(frameShape[1])] for y in range(frameShape[0])]
        npwhiteFrame = np.array(whiteFrame, dtype=np.uint8)
        return npwhiteFrame
    def blackFrame(self,frameShape):
        blackFrame = [[[0, 0, 0] for x in range(frameShape[1])] for y in range(frameShape[0])]
        npblackFrame = np.array(blackFrame, dtype=np.uint8)
        return npblackFrame
    def blackWhiteFilcker(self,flickerCount,repetition1,repetition2, frameShape):
        blackFrame = self.blackFrame(frameShape)
        whiteFrame = self.whiteFrame(frameShape)
        flicker =[blackFrame for _ in range(repetition1)]+ [whiteFrame for _ in range(repetition2)]
        return flicker * flickerCount

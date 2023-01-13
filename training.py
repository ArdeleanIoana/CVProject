import torch
import cv2
from RNN import *
from frameProcessing import FeedData
from utils import category_from_output
from torch import optim
# inspiration: https://www.youtube.com/watch?v=WEV61GmmPrk&list=PLiDmKRJhglti6HwdDP9kEItTlHMZCDPk_&index=4&t=184s


device = "cuda" if torch.cuda.is_available() else "cpu"
def trainingLoop(hidden_size, learning_rate, epochs, sequence_length, num_layers, batch_size):
    input_size = 1  # size of one seq input in rnn
    num_classes = 2  # number of classification ouput
    loader = FeedData()
    rnn = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

    while loader.getEpochs() < epochs:
        print(str(loader.getEpochs()) + " out of " + str(epochs))
        gif_batch, category_batch = loader.feedBatch(batch_size)
        scores = rnn(gif_batch)
        loss = criterion(scores, category_batch)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # gradient descent update step/adam step
        optimizer.step()
        print("loss: " + str(loss))
    return rnn

def manualTest(RNN):
    loader = FeedData()
    while(True):
        x = input("path: ")
        if x == "x":
            break
        else:
            gif = cv2.VideoCapture(x)
            tens = loader.gifToTensor(gif)
            hidden = RNN.init_hidden()
            for i in range(tens.size()[0]):
                output, hidden = RNN(tens[i], hidden)

            guess = category_from_output(output)
            print(guess)

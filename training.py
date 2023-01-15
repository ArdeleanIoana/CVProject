import torch
import cv2
from RNN import *
from FeedData import FeedData
from utils import category_from_output
from torch import optim
# inspiration: https://www.youtube.com/watch?v=WEV61GmmPrk&list=PLiDmKRJhglti6HwdDP9kEItTlHMZCDPk_&index=4&t=184s


device = "cuda" if torch.cuda.is_available() else "cpu"
def trainingLoop(hidden_size, learning_rate, epochs, num_layers, batch_size):
    input_size = 1  # size of one seq input in rnn
    num_classes = 2  # number of classification ouput
    loader = FeedData()
    rnn = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

    while loader.getEpochs() < epochs:
        print(str(loader.getEpochs()) + " out of " + str(epochs))
        gif_batch, category_batch = loader.feedBatchFromFile(batch_size)
        scores = rnn(gif_batch)
        loss = criterion(scores, category_batch)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # gradient descent update step/adam step
        optimizer.step()
        print("loss: " + str(loss))
    return rnn, optimizer


import torch
import cv2
from RNN import *
from frameProcessing import FeedData
from utils import category_from_output
# inspiration: https://www.youtube.com/watch?v=WEV61GmmPrk&list=PLiDmKRJhglti6HwdDP9kEItTlHMZCDPk_&index=4&t=184s
#problems with loss

def train(line_tensor, rnn):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(torch.FloatTensor(line_tensor[i]), hidden)
    return output

def trainingLoop(n_hidden, learning_rate,n_iters):
    n_input = 1  # size of one seq input in rnn
    n_categories = 2  # number of classification ouput
    loader = FeedData()
    rnn = RNN(n_input, n_hidden, n_categories)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    for i in range(n_iters):
        print(str(i) + " out of " + str(n_iters))
        category, line, category_tensor, line_tensor = loader.feedNextTrain()
        output = train(line_tensor, rnn)
        loss = criterion(output, category_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(" loss " + str(loss.item()))
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

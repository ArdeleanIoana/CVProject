import torch
import cv2
from RNN import *
from frameProcessing import FeedData
from utils import category_from_output
# inspiration: https://www.youtube.com/watch?v=WEV61GmmPrk&list=PLiDmKRJhglti6HwdDP9kEItTlHMZCDPk_&index=4&t=184s
# params from name classification example
n_input = 1 # number of all posibile letter. might mean number of all possbible extracted frames for me
n_categories = 2 # number of classification ouput, obvi 2 for me, safe unsafe
n_hidden = 128 #hyper parameter
rnn = RNN(n_input, n_hidden, n_categories)
criterion = nn.L1Loss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
current_loss = 0
all_losses = []
plot_steps, print_steps = 10, 10
n_iters = 100
loader = FeedData()



# def train : one gif aka one training step
#line_tensor: whole name, for me whole list of frames
def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(torch.FloatTensor(line_tensor[i]), hidden)
    loss = criterion(output,category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()

def trainingLoop():
    current_loss = 0
    for i in range(n_iters):
        print(str(i) + " out of " + str(n_iters))
        category, line, category_tensor, line_tensor = loader.feedNextTrain()
        output, loss = train(line_tensor, category_tensor)
        current_loss += loss
        print(" loss " + str(current_loss))

    while(True):
        x = input("path: ")
        if x == "x":
            break
        else:
            gif = cv2.VideoCapture(x)
            tens = loader.gifToTensor(gif)
            hidden = rnn.init_hidden()
            for i in range(tens.size()[0]):
                output, hidden = rnn(tens[i], hidden)

            guess = category_from_output(output)
            print(guess)

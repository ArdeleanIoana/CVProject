import torch

from RNN import *
from frameProcessing import FeedData
# inspiration: https://www.youtube.com/watch?v=WEV61GmmPrk&list=PLiDmKRJhglti6HwdDP9kEItTlHMZCDPk_&index=4&t=184s
# params from name classification example
n_frames = 317 # number of all posibile letter. might mean number of all possbible extracted frames for me
n_categories = 2 # number of classification ouput, obvi 2 for me, safe unsafe
n_hidden = 128 #hyper parameter
rnn = RNN(n_frames, n_hidden, n_categories)
criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000
loader = FeedData()

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return category_idx

# def train : one gif aka one training step
#line_tensor: whole name, for me whole list of frames
def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    for i in range(len(line_tensor.size())):
        output, hidden = rnn(torch.FloatTensor([line_tensor[i]]), hidden)
    loss = criterion(output,category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()

def trainingLoop():
    for i in range(n_iters):
        category, line, category_tensor, line_tensor = loader.feedNextTrain()

        output, loss = train(line_tensor, category_tensor)
        current_loss += loss

        if (i + 1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0

        if (i + 1) % print_steps == 0:
            guess = category_from_output(output)
            correct = "CORRECT" if guess == category else f"WRONG ({category})"
            print(f"{i + 1} {(i + 1) / n_iters * 100} {loss:.4f} {line} / {guess} {correct}")
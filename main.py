
from training import trainingLoop
from performance import runPerformance
import torch
from RNN import RNN
#316 last result

def saveModel(rnn,optimizer,filename):
    done = {'state_dict': rnn.state_dict(), 'optimizer_dict': optimizer.state_dict()}
    torch.save(done, "try1.pth.tar")
def main():
    print("main function run")
    hidden_size = 10
    num_layers = 2
    sequence_length = 317
    learning_rate = 0.005
    batch_size = 30
    num_epochs = 2
    rnn, optimizer = trainingLoop(hidden_size, learning_rate,num_epochs, num_layers, batch_size)
    test_count = 400#400 is maximum
    runPerformance(rnn, test_count, "results\\resultsLSTM21.txt", learning_rate, hidden_size, num_epochs, num_layers, batch_size)

if  __name__ == '__main__':
    main()
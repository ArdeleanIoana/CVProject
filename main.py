
from training import trainingLoop
from performance import runPerformance
from RNN import RNN
#316 last result

def main():
    print("main function run")
    hidden_size = 250
    num_layers = 2
    sequence_length = 317
    learning_rate = 0.005
    batch_size = 15
    num_epochs = 1
    rnn = trainingLoop(hidden_size, learning_rate,num_epochs, sequence_length, num_layers, batch_size)
    test_count = 30 #400 is maximum
    runPerformance(rnn, test_count, "resultsRNNsmall.txt", learning_rate, hidden_size, num_epochs, num_layers, batch_size)

if  __name__ == '__main__':
    main()
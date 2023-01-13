
from training import trainingLoop
from performance import runPerformance
from RNN import RNN
#316 last result

def main():
    print("main function run")
    n_hidden = 125  # hyper parameters
    learning_rate = 0.001
    n_iters = 600
    rnn = trainingLoop(n_hidden, learning_rate,n_iters)
    test_count = 400 #400 is maximum
    runPerformance(rnn, test_count, "results4.txt", learning_rate, n_hidden, n_iters)
main()
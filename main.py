
from training import trainingLoop
from performance import runPerformance
from RNN import RNN
#316 last result

def main():
    print("main function run")
    n_hidden = 128  # hyper parameter

    learning_rate = 0.005
    n_iters = 1
    rnn = trainingLoop(n_hidden, learning_rate,n_iters)
    runPerformance(rnn)
main()
import sys
from FeedData import FeedData
from utils import category_from_output
from RNN import RNN
import cv2
# positive = unsafe
# negative = safe

def getResults(rnn, test_count):
    parmDic = {"fp": 0, "tp": 0, "fn": 0, "tn": 0}
    feeder = FeedData()
    for i in range(test_count):
        print(i)
        category, tens = feeder.feedNextTest()
        output = rnn(tens)
        guess = category_from_output(output)
        print("guess:" + guess + " true: " + category)
        if guess == "safe":
            if category == "safe":
                parmDic["tn"] += 1
                print("tn case")
            else:
                parmDic["fn"] += 1
                print("fn case")
        else:
            if category == "safe":
                parmDic["fp"] += 1
                print("fp case")
            else:
                parmDic["tp"] += 1
                print("tp case")
    return parmDic

def saveToFile(filename, precision, accuracy, recall, learning_rate, n_hidden, n_iters,test_count, num_layers, batch_size):
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("-------------HYPER-PARAMS-----------------")
        print("learning rate: " + str(learning_rate))
        print("hidden layers:" + str(n_hidden))
        print("number of epochs: " + str(n_iters))
        print("number of gifs tested: " + str(test_count))
        print("number of layers: " + str(num_layers))
        print("batch size: " + str(batch_size))
        print("-------------RESULTS---------------------")
        print("accuracy: " + str(accuracy))
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        sys.stdout = original_stdout  # Reset the standard output to its original value

def getPrecision(dict):
    if dict["tp"] == 0 and  dict["fp"] == 0:
        return 0
    return dict["tp"] / (dict["tp"] + dict["fp"])

def getAccuracy(dict):
    return (dict["tp"] + dict["tn"]) / (dict["tp"] + dict["fp"] + dict["tn"] + dict["fn"])


def getRecall(dict):
    return dict["tp"] / (dict["tp"] + dict["fn"])


def runPerformance(rnn, test_count,filename, learning_rate, n_hidden, n_iters, num_layers, batch_size):
    dic = getResults(rnn, test_count)
    print(dic)
    saveToFile(filename,getPrecision(dic),getAccuracy(dic),getRecall(dic), learning_rate, n_hidden, n_iters,test_count, num_layers, batch_size)

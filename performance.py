import sys
from frameProcessing import FeedData
from utils import category_from_output
from RNN import RNN
import cv2
# positive = safe
# negative = unsafe

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
                parmDic["tp"] += 1
                print("tp case")
            else:
                parmDic["fp"] += 1
                print("fp case")
        else:
            if category == "safe":
                parmDic["fn"] += 1
                print("fn case")
            else:
                parmDic["tn"] += 1
                print("tn case")
    return parmDic

def saveToFile(filename, precision, accuracy, recall, learning_rate, n_hidden, n_iters,test_count):
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("-------------HYPER-PARAMS-----------------")
        print("learning rate: " + str(learning_rate))
        print("hidden layers:" + str(n_hidden))
        print("number of iterations: " + str(n_iters))
        print("number of gifs tested: " + str(test_count))
        print("-------------RESULTS---------------------")
        print("accuracy: " + str(accuracy))
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        sys.stdout = original_stdout  # Reset the standard output to its original value

def getPrecision(dict):
    return dict["tp"] / (dict["tp"] + dict["fp"])
def getAccuracy(dict):
    return (dict["tp"] + dict["tn"]) / (dict["tp"] + dict["fp"] + dict["tn"] + dict["fn"])
def getRecall(dict):
    return dict["tp"] / (dict["tp"] + dict["fn"])

def runPerformance(rnn, test_count,filename, learning_rate, n_hidden, n_iters):
    dic = getResults(rnn, test_count)
    print(dic)
    saveToFile(filename,getPrecision(dic),getAccuracy(dic),getRecall(dic), learning_rate, n_hidden, n_iters,test_count)

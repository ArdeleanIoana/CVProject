import sys
from frameProcessing import FeedData
from utils import category_from_output
from RNN import RNN
import cv2
# positive = safe
# negative = unsafe

def getResults(rnn):
    parmDic = {"fp": 0, "tp": 0, "fn": 0, "tn": 0}
    feeder = FeedData()
    for i in range(400):
        print(i)
        category, tens = feeder.feedNextTest()
        hidden = rnn.init_hidden()
        for i in range(tens.size()[0]):
            output, hidden = rnn(tens[i], hidden)
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

def saveToFile(filename, precision, accuracy, recall):
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
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

def runPerformance(rnn):
    dic = getResults(rnn)
    print(dic)
    saveToFile("results.txt",getPrecision(dic),getAccuracy(dic),getRecall(dic))

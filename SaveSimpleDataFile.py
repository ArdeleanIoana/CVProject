from frameProcessing import *
import sys


def dictToString(dict):
    list = dict['frames']
    category = dict['category']
    return "{ \"frames\":" + str(list) + ", \"category\":" +" \""+ category + "\"}"
def saveArraysMain(filename, source):
    data = FeedData()
    trainCSV = pd.read_csv(source)

    writeToFile(filename, "[")
    for iter in range(600):
        path = trainCSV["video_name"].values.tolist()[iter]
        category = trainCSV["tag"].values.tolist()[iter]
        gif = cv2.VideoCapture(path)
        frames, fps = data.gifToListOfAvgLumi(gif)
        frames = [fps] + frames
        frames = data.add_padding(frames)
        dict = { "frames": frames, "category": category}
        print(dict)
        writeToFile(filename,dictToString(dict))
        if iter!=599:
            writeToFile(filename, ",")
    writeToFile(filename, "]")


def writeToFile(filename,result):
    try:
        original_stdout = sys.stdout
        with open(filename, 'a') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print(result)
            sys.stdout = original_stdout  # Reset the standard output to its original value
    except:
        original_stdout = sys.stdout
        with open(filename, 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print(result)
            sys.stdout = original_stdout  # Reset the standard output to its original value

if __name__ == '__main__':
    #saveArraysMain("trainingData.json","train.csv")
    saveArraysMain("testData.json", "test.csv")
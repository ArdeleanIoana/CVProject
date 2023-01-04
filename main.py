from prepareData import *
from frameProcessing import *
from training import *
#316 last result
def maxNumberOfFrames():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    video_paths_train = train_df["video_name"].values.tolist()
    video_paths_test = test_df["video_name"].values.tolist()
    max = 0
    for path in video_paths_train:
        gif = cv2.VideoCapture(path)
        count = countFrames(gif)
        if count > max:
            max = count
    for path in video_paths_test:
        gif = cv2.VideoCapture(path)
        count = countFrames(gif)
        if count > max:
            max = count
    return max

def main():
    print("main function run")
    trainingLoop()

main()
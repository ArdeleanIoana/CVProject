import pandas as pd
import cv2
import torch
def countFrames(gif):
    count = 0
    ret, frame = gif.read()  # ret=True if it finds a frame else False.
    while ret:
        # read next frame
        ret, frame = gif.read()
        count += 1
    return count

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

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return category_idx
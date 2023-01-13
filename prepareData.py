import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# source
# https://www.youtube.com/watch?v=ezjnySXqdTo&list=PLiDmKRJhglti6HwdDP9kEItTlHMZCDPk_&index=3&t=681s


def trainDataAsCSV():
    dataset_path = os.listdir('dataset/train')
    label_types = os.listdir('dataset/train')
    rooms = []
    for item in dataset_path:
        # Get all the file names
        all_rooms = os.listdir('dataset/train' + '/' + item)

        # Add them to the list
        for room in all_rooms:
            rooms.append((item, str('dataset/train' + '/' + item) + '/' + room))
    random.shuffle(rooms)
    # Build a dataframe
    train_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
    df = train_df.loc[:, ['video_name', 'tag']]
    df.to_csv('train.csv')

def testDataAsCSV():
    dataset_path = os.listdir('dataset/test')
    label_types = os.listdir('dataset/test')
    rooms = []
    for item in dataset_path:
        # Get all the file names
        all_rooms = os.listdir('dataset/test' + '/' + item)

        # Add them to the list
        for room in all_rooms:
            rooms.append((item, str('dataset/test' + '/' + item) + '/' + room))

    # Build a dataframe
    test_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
    df = test_df.loc[:, ['video_name', 'tag']]
    df.to_csv('test.csv')


if  __name__ == '__main__':
    trainDataAsCSV()
    testDataAsCSV()
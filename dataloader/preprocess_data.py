import pandas as pd
import numpy as np
import torchvision
import os
import glob

def label2str():
    label_dict = {}
    with open("data/CUB_200_2011/classes.txt", encoding="utf-8", mode="r") as f:
        for line in f.readlines():
            label_idx, label_str = line.split()
            label_dict[label_idx] = label_str
        return label_dict

if __name__ == "__main__":
    # Parse data to csv file
    # Get image path and index
    with open("data/CUB_200_2011/images.txt", mode="r", encoding="utf-8") as f:
        index_lst = []
        path_lst = []
        for line in f.readlines():
            index, path = line.split()
            index_lst.append(index)
            path_lst.append(path)
    
    # Get image class
    with open("data/CUB_200_2011/image_class_labels.txt", mode="r", encoding="utf-8") as f:
        label_lst = []
        for line in f.readlines():
            index, label = line.split()
            label_lst.append(label)

    # Get image tag (train, test)
    with open("data/CUB_200_2011/train_test_split.txt", mode="r", encoding="utf-8") as f:
        tag_lst = []
        for line in f.readlines():
            index, tag = line.split()
            tag_lst.append(tag)

    # Get image bounding boxes for segmentation
    with open("data/CUB_200_2011/bounding_boxes.txt", mode="r", encoding="utf-8") as f:
        x_lst = []
        y_lst = []
        w_lst = []
        h_lst = []
        for line in f.readlines():
            index, x, y, w, h = line.split()
            x_lst.append(x)
            y_lst.append(y)
            w_lst.append(w)
            h_lst.append(h)

    dataframe = pd.DataFrame(data=zip(index_lst, path_lst, label_lst, tag_lst, x_lst, y_lst, w_lst, h_lst),
                             columns=["index", "filepath", "label", "tag", "x", "y", "w", "h"]
                             )
    
    train_dataframe = dataframe[dataframe["tag"]=='1']
    test_dataframe = dataframe[dataframe["tag"]=='0']
    
    assert len(train_dataframe) + len(test_dataframe) == len(dataframe), "Somethings wrong bro"

    train_dataframe.to_csv("data/train.csv")
    test_dataframe.to_csv("data/test.csv")


    

    

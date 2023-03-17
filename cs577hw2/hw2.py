import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


fileTrain = "WiC_dataset/train/train.data.txt"
fileTest = "WiC_dataset/test/test.data.txt"
trainList = []

with open(fileTrain, "r") as f:

    while True:
        line = f.readline()
        trainList.append(line.split())
        #print(line)
        if not line:
            break

print(trainList)
            

# targets_numpy = train.label.values
# features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

# # train test split. Size of train data is 80% and size of test data is 20%. 
# features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
#                                                                              targets_numpy,
#                                                                              test_size = 0.2,
#                                                                              random_state = 42) 

# # create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
# featuresTrain = torch.from_numpy(features_train)
# targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# # create feature and targets tensor for test set.
# featuresTest = torch.from_numpy(features_test)
# targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long


# # batch_size, epoch and iteration
# batch_size = 100
# n_iters = 10000
# num_epochs = n_iters / (len(features_train) / batch_size)
# num_epochs = int(num_epochs)

# # Pytorch train and test sets
# train = TensorDataset(featuresTrain,targetsTrain)
# test = TensorDataset(featuresTest,targetsTest)

# # data loader
# train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
# test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

# # visualize one of the images in data set
# plt.imshow(features_numpy[10].reshape(28,28))
# plt.axis("off")
# plt.title(str(targets_numpy[10]))
# plt.savefig('graph.png')
# plt.show()
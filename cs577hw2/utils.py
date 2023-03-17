from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import gensim.downloader as api


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class WiCDataset(Dataset):

    def __init__(self, data):
        #data = np.loadtxt("./WiC_dataset/train/train.gold.txt", delimiter='\t')

        self.mydata = pd.read_csv(data, sep='\t', header=None)

        #print(self.mydata[0]) ## print first column
        #print(self.mydata[1]) ## print second column

        #print(self.mydata.loc[0,[3]]) ## the first sentence is 3rd column
        #print(self.mydata.loc[0,[4]]) ## the second sentence is 4th column

    def __len__(self):
        
        shape = self.mydata.shape
        #print("my shape is", shape[0], shape)
        return shape[0]

    def __getitem__(self, rowidx, colidx):
        
        currendata = self.mydata.loc[rowidx, [colidx]]    
        
        return currendata


myinstance = WiCDataset("./WiC_dataset/train/train.data.txt")
myinstance.__len__()
rowidx, colidx = 0, 3
mysentence = myinstance.__getitem__(rowidx, colidx)
print(mysentence.apply)
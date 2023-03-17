from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import csv
from nltk.tokenize import word_tokenize
import gensim.downloader as api





# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class WiCDataset(Dataset):

    def __init__(self, data, label):
        # Open the CSV file and read its contents
        self.mydata = []
        self.ylabel = []
        with open(data, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                self.mydata.append(row)

        with open(label, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:

                if row == "F":
                    self.ylabel.append(0)
                else:
                    self.ylabel.append(1)


    def __len__(self):
        
        numrow, numcol = len(self.mydata), len(self.mydata[0])
        #print("my shape is", shape[0], shape)
        return numrow, numcol  ### this will return the number of sentence and number of words in the sentence

    def __getitem__(self, rowidx, colidx):
        
        currendata = self.mydata[rowidx][colidx]    
        
        return currendata


    def preprocess(self, text):
        
        text = text.rstrip(".")
        tokens = word_tokenize(text)
        tokensList = []
        glove_embs = api.load("glove-wiki-gigaword-50")
        sentence_embs = []
        for token in tokens:

            if token.lower() in  glove_embs:
                print(token) ### the type is string
                embedding_vector = glove_embs[token.lower()]
                sentence_embs.append(embedding_vector)
            else: continue

        mean = np.mean(np.array(sentence_embs), axis=0)

        # print(len(mean)) ### this will print the number of dimension of the word vectors; the size of the vector is 50
        return mean   ### this will return the sentence embedding
# myinstance = WiCDataset("./WiC_dataset/train/train.data.txt", "./WiC_dataset/train/train.gold.txt")
# print(myinstance.__len__())
# rowidx, colidx = 0, 3
# mysentence = myinstance.__getitem__(rowidx, colidx)
# print(mysentence)
# print(myinstance.preprocess(mysentence))
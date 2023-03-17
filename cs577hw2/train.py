import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from utils2 import WiCDataset
import torch.nn as nn

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
torch.set_default_tensor_type(torch.FloatTensor)
#torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cpu")

'''
NOTE: Do not change any of the statements above regarding random/numpy/pytorch.
You can import other built-in libraries (e.g. collections) or pre-specified external libraries
such as pandas, nltk and gensim below. 
Also, if you'd like to declare some helper functions, please do so in utils.py and
change the last import statement below.
'''

import gensim.downloader as api

from neural_archs import DAN, RNN, LSTM

if __name__ == "__main__":

    myinstance = WiCDataset("./WiC_dataset/train/train.data.txt", "./WiC_dataset/train/train.gold.txt")


    parser = argparse.ArgumentParser()

    # TODO: change the `default' attribute in the following 3 lines with the choice
    # that achieved the best performance for your case
    parser.add_argument('--neural_arch', choices=['dan', 'rnn', 'lstm'], default='dan', type=str)
    parser.add_argument('--rnn_bidirect', default=False, action='store_true')
    parser.add_argument('--init_word_embs', choices=['scratch', 'glove'], default='scratch', type=str)

    args = parser.parse_args()

    train_size = 5428

    if args.init_word_embs == "glove":
        # TODO: Feed the GloVe embeddings to NN modules appropriately
        # for initializing the embeddings
        #glove_embs = api.load("glove-wiki-gigaword-50")
        #embedding_vector = glove_embs['example']

        myinstance.__len__()
        colidx1, colidx2 = 2, 3
        twosentenceEmbeddingX = []
        for rowidx in range(train_size):
            mysentence1 = myinstance.__getitem__(rowidx, colidx1)
            mysentence2 = myinstance.__getitem__(rowidx, colidx2)

            onesentenceEmbedding1 = myinstance.preprocess(mysentence1)
            onesentenceEmbedding2 = myinstance.preprocess(mysentence2)


            twosentenceEmbeddingX.append(onesentenceEmbedding1 + onesentenceEmbedding2)
        


    # TODO: Freely modify the inputs to the declaration of each module below
    if args.neural_arch == "dan":

        input_size = 100
        hidden_size = 10
        output_size = 1
        dan = DAN(input_size, hidden_size, output_size).to(torch_device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(dan.parameters(), lr=0.01)

        # Train the model
        for epoch in range(100):
            # Generate some dummy data
            inputs = torch.tensor(twosentenceEmbeddingX)
            targets = torch.tensor(myinstance.ylabel)
            # if epoch == 0:
            #     print("#########################")
            #     print(inputs)
            #     print(type(inputs))
            #     print(targets)
            #     print(type(targets))
            #     print("#########################")
            #     break
            #     pass


            # Forward pass
            outputs = dan(inputs)  ### so why it does not need to explictly call dan.forward ???
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            if (epoch+1) % 10 == 0:
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == targets).sum().item() / targets.size(0)
                print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch+1, 100, loss.item(), accuracy*100))
            elif args.neural_arch == "rnn":
                if args.rnn_bidirect:
                    model = RNN().to(torch_device)
                else:
                    model = RNN().to(torch_device)
            elif args.neural_arch == "lstm":
                if args.rnn_bidirect:
                    model = LSTM().to(torch_device)
                else:
                    model = LSTM().to(torch_device)

    # TODO: Read off the WiC dataset files from the `WiC_dataset' directory
    # (will be located in /homes/cs577/WiC_dataset/(train, dev, test))
    # and initialize PyTorch dataloader appropriately
    # Take a look at this page
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # and implement a PyTorch Dataset class for the WiC dataset in
    # utils.py

    # TODO: Training and validation loop here

    # TODO: Testing loop
    # Write predictions (F or T) for each test example into test.pred.txt
    # One line per each example, in the same order as test.data.txt.



'''


input_size = 100
output_size = 2
hidden_size = 32

# Create the DAN model


'''
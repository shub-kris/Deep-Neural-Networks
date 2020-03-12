import unidecode
import string
import random
import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable


'''
Reading the text file
'''
def read_file(filename):
    file = unidecode.unidecode(open(filename).read())    # Returns a unicode file/string
    file = file.lower()
    # print(file[:5], len(file))
    #remove all special characters
    # print(f'before: {text[:100]} ')
    # processed_text = re.sub(r'[^a-z]'," ", text)
    # print(processed_text[:100])
    return file

'''
Function for mapping each character to numbers using dictionary
'''

def create_mapping(file):                 # Defines one-to-one mapping between characters and integers by using Dictionary
    chars = tuple(set(file))
    # print(chars)
    int2char = {i:ch for i, ch in enumerate(chars)}
    char2int = {ch:i for  i, ch in enumerate(chars)}
    return int2char, char2int

'''
Using One-hot Encoding for the input , Word2Vec can also be used or other Word Embeddings
'''

def one_hot_encode(arr, n_chars):
    one_hot = np.zeros((arr.size, n_chars), dtype = np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    one_hot = one_hot.reshape((*arr.shape, n_chars))
    return one_hot
    

'''
We need to prepare batches of data for training
input will be a single character and output will be a character shifted by 1
Ex: Word 'Hello' , ip: H , op: e

At each time step we are feeding an input of size seq_length and since we have a batchsize bs
So, we have bs*seq_length characters in each batch
mini_bs = Size of each mini_batch

Creating a Genrator to iterate through batches
'''    
    
def batchify(arr, batch_size, seq_length):

    tbatch_size = batch_size * seq_length   #Seq-length tells at one time how many characters you want to use as i/p
    n_batches = len(arr) // tbatch_size
    
    arr = arr[:n_batches * tbatch_size]   # We want only enough characters to have full batches and rest we leave
    arr = arr.reshape((batch_size , -1))
    
    for i in range(0, arr.shape[1], seq_length):
        x = arr[:, i:i + seq_length]
        y = np.zeros_like(x)                # Targets, shifted by one
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, i+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y



class RNN(nn.Module):
    def __init__(self, ip_dim, hid_dim = 256, n_hid_layer = 2):
        super(RNN, self).__init__()
        self.ip_dim = ip_dim
        self.hid_dim = hid_dim
        self.hid_layer = n_hid_layer
        self.lstm = nn.LSTM(ip_dim, hid_dim, n_hid_layer, batch_first = True)
        self.fc = nn.Linear(hid_dim, ip_dim)
    
    def forward(self, x, hidden):
        op, hidden = self.lstm(x, hidden)
        # print(op.size)
        op = op.contiguous().view(-1, self.hid_dim)   #Becuase it may happen that pytorch just change the indexes, contigous to make sure it doesn't happen
        op = self.fc(op)
        return op, hidden
        
    def weight_init(self, batch_size):
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.hid_layer, batch_size, self.hid_dim).zero_().cuda(),
                  weight.new(self.hid_layer, batch_size, self.hid_dim).zero_().cuda())
        return hidden
    

def train(model, data, device, epochs = 10, batch_size = 10, seq_length = 50, 
                lr = 0.001, clip = 5, val_frac = 0.1, print_every = 10):
    
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    model.to(device)
    
    cnt = 0
    for i in range(epochs):
        h = model.weight_init(batch_size)
        for x, y in batchify(data, batch_size, seq_length):
            cnt += 1
            x = one_hot_encode(x, model.ip_dim)
            # print(x.shape)
            x, y = torch.from_numpy(x), torch.from_numpy(y).to(torch.long)
            ip, target = Variable(x).to(device), Variable(y).to(device)
            
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([Variable(each.data) for each in h])
            
            model.zero_grad()
            
            op, h = model.forward(ip, h)
            loss = criterion(op, target.view(batch_size * seq_length))
            loss.backward()
            #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm(model.parameters(), clip)
            
            optimizer.step()
            
            if cnt % print_every == 0:
            # Get validation loss
                val_h = model.weight_init(batch_size)
                val_losses = []
                model.eval()
                for x, y in batchify(val_data, batch_size, seq_length):
                # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, model.ip_dim)
                    x, y = torch.from_numpy(x), torch.from_numpy(y).to(torch.long)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([Variable(each.data) for each in val_h])

                    inputs, targets = Variable(x).to(device), Variable(y).to(device)

                    output, val_h = model.forward(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length))

                    val_losses.append(val_loss.item())
                    
                model.train()
                print("Epoch: {}/{}...".format(i+1, epochs),
                "Step: {}...".format(cnt),
                "Loss: {:.4f}...".format(loss.item()),
                "Val Loss: {:.4f}".format(np.mean(val_losses)))

 
def main():
    data = read_file('shakespeare.txt')
    # print(data)
    int2char, char2int = create_mapping(data)

    encoded = np.array([char2int[ch] for ch in data])   
    # test = one_hot_encode(np.array([[1,2,3]]), 4)
    # print(test)
    hid_dim = 256
    n_hlayer = 2
    ip_dim = len(int2char)
    batch_size = 10 
    seq_length = 50
    lr = 0.001
    print_every = 10
    model = RNN(ip_dim, hid_dim, n_hlayer)
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # train(model, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)
    train(model, encoded, device, epochs = 10, batch_size = 10, seq_length = 50, 
                lr = 0.001, clip = 5, val_frac = 0.1, print_every = 10)
    
    
if __name__ == '__main__':
    main()
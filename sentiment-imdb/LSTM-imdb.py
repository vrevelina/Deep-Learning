### Training a Bidirectional LSTM for Sentiment Analysis on IMDB Dataset
### Vania Revelina

import torch
import torch.nn as nn
import torch.optim as optim
import random
import utils
import time

# using GPU
device= torch.device("cuda")

# download IMDB dataset
train_data = torch.load('imdb/train_data.pt')
train_label = torch.load('imdb/train_label.pt')
test_data = torch.load('imdb/test_data.pt')
test_label = torch.load('imdb/test_label.pt')

print('num review in train = ', len(train_data))
print('num review in test = ', len(test_data))

# Create the network using a bidirectional LSTM
class rec_neural_net(nn.Module):
    
    def __init__(self, vocab_size, hidden_size, output_size, num_layers):
        super().__init__()
        
        self.emb_layer = nn.Embedding(vocab_size, hidden_size)
        self.rec_layer = nn.LSTM(hidden_size,hidden_size,num_layers=num_layers,bidirectional=True)
        self.lin_layer = nn.Linear(hidden_size*2,output_size)
        
        
    def forward(self, input_seq):
        
        input_seq_emb = self.emb_layer(input_seq)
    
        output_seq,(h_last,c_last) = self.rec_layer(input_seq_emb)
        
        h_direc_1  = h_last[2,:,:]
        h_direc_2  = h_last[3,:,:]
        h_direc_12 = torch.cat( (h_direc_1, h_direc_2)  , dim=1) 
        
        scores = self.lin_layer(h_direc_12)
            
        return scores

# instantiate the neural net
vocab_size = 25002
num_layers = 2
hid_size = 50
out_size = 2

net = rec_neural_net(vocab_size,hid_size,out_size,num_layers)

# SEND NETWORK TO GPU:
net = net.to(device)

print(net)
utils.display_num_param(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1)
bs = 64

# Function to evaluate the network on the test set
def eval_on_test_set():

    running_error=0
    num_batches=0

    with torch.no_grad():

        for i in range(0,25000-bs,bs):

            # extract the minibatch
            indices = torch.arange(i,i+bs)            
            minibatch_data, minibatch_label =  utils.make_minibatch(indices, test_data, test_label) 
            
            # truncate if review longer than 500:
            if minibatch_data.size(0)>500:
                minibatch_data = minibatch_data[0:499]  
                
            # send to GPU    
            minibatch_data = minibatch_data.to(device)
            minibatch_label = minibatch_label.to(device) 
            
            # feed it to the network
            scores=net(minibatch_data) 

            # compute the error made on this batch
            error = utils.get_error( scores , minibatch_label)

            # add it to the running error
            running_error += error.item()

            num_batches+=1

    # compute error rate on the full test set
    total_error = running_error/num_batches

    print( 'error rate on test set =', total_error*100 ,'percent')


# Do 16 passes through the training set
start=time.time()

for epoch in range(16):
    
    running_loss=0
    running_error=0
    num_batches=0
    
    shuffled_indices=torch.randperm(25000)
 
    for count in range(0,25000-bs,bs):
      
        # Set the gradients to zeros
        optimizer.zero_grad()
        
        # get the minibatch
        indices = shuffled_indices[count:count+bs]
        minibatch_data, minibatch_label =  utils.make_minibatch(indices, train_data, train_label) 
        
        # truncate if review longer than 500:
        if minibatch_data.size(0)>500:
            minibatch_data = minibatch_data[0:500]  
            
        # send to GPU    
        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device) 

        # forward the minibatch through the net        
        scores = net(minibatch_data)

        # Compute the average of the losses of the data points in the minibatch
        loss = criterion( scores , minibatch_label) 
        
        # backward pass    
        loss.backward()
        
        # clip the gradient
        nn.utils.clip_grad_norm_(net.parameters(), 5)

        # do one step of stochastic gradient descent
        optimizer.step()     

        
        # computing stats
        num_batches+=1
        with torch.no_grad():
            running_loss += loss.item()
            error = utils.get_error( scores , minibatch_label)
            running_error += error.item()
                          
    # epoch finished:  compute and display stats for the full training set
    total_loss = running_loss/num_batches
    total_error = running_error/num_batches
    elapsed = time.time()-start
    print('epoch=',epoch, '\t time=', elapsed/60, '\t loss=', total_loss , '\t error=', total_error*100 ,'percent')
    
    # compute error on the test set:
    eval_on_test_set() 
    print(" ")

# Save the trained parameters
torch.save( net.state_dict() , 'trained_parameters_LSTM.pt'  )
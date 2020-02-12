import torch
import torch.nn.functional as F
import numpy as np
import math
from collections import defaultdict


word2idx  =  torch.load('../../data/ptb/word2idx.pt')
idx2word  =  torch.load('../../data/ptb/idx2word.pt')


word2idx_temp=defaultdict(lambda:26)
for w, idx in word2idx.items():
     word2idx_temp[w]=idx
word2idx = word2idx_temp


def text2tensor(text):
    text=text.lower()
    list_of_words = text.split()
    list_of_int = [ word2idx[w] for w in list_of_words ]
    x=torch.LongTensor(list_of_int)
    return x

def tensor2text(x):
    list_of_words = [ idx2word[idx.item()] for idx in x]
    text = ' '.join(list_of_words)
    return text


def normalize_gradient(net):

    grad_norm_sq=0

    for p in net.parameters():
        grad_norm_sq += p.grad.data.norm()**2

    grad_norm=math.sqrt(grad_norm_sq)
   
    if grad_norm<1e-4:
        net.zero_grad()
        print('grad norm close to zero')
    else:    
        for p in net.parameters():
             p.grad.data.div_(grad_norm)

    return grad_norm


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param/1e6)
         )




def show_most_likely_words(prob):
    num_word_display=30
    p=prob.view(-1)
    p,word_idx = torch.topk(p,num_word_display)

    for i,idx in enumerate(word_idx):
        percentage= p[i].item()*100
        word=  idx2word[idx.item()]
        print(  "{:.1f}%\t".format(percentage),  word ) 

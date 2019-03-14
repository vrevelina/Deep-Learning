import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


word2idx = torch.load('../../data/imdb/word2idx.pt')
idx2word = torch.load('../../data/imdb/idx2word.pt')


def text2tensor(text):
    list_of_words = text.split()
    list_of_int = [ word2idx[w] for w in list_of_words ]
    x=torch.LongTensor(list_of_int)
    return x

def tensor2text(x):
    list_of_words = [ idx2word[idx.item()] for idx in x]
    text = ' '.join(list_of_words)
    return text



def reorder_minibatch(minibatch_data, minibatch_label):
    temp = sorted( zip(minibatch_data, minibatch_label) , key = lambda x: len(x[0]), reverse=True )
    list1,list2 = map(list, zip(*temp) )
    return list1, list2


def make_minibatch(indices, data, label):
        
        minibatch_data = [ data[idx.item()]  for idx in indices  ]
        minibatch_label= [ label[idx.item()] for idx in indices  ] 
        
        minibatch_data , minibatch_label = reorder_minibatch(minibatch_data, minibatch_label) 
        minibatch_data = nn.utils.rnn.pad_sequence(minibatch_data, padding_value=1)
        minibatch_label = torch.stack( minibatch_label,dim=0)
        
        return minibatch_data, minibatch_label    


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param/1e6)
         )

def get_error( scores , labels ):

    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    
    return 1-num_matches.float()/bs    




def show_prob_imdb(p):


    p=p.data.squeeze().cpu().numpy()

    ft=15
    label = ('negative', 'positive' )
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)



    plt.show()




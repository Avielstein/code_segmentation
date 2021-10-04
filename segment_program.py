#########################################################
# Imports
#########################################################


#stanard
import json
import random as ra
import numpy as np
import math

#display
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence

#specialized
import sentencepiece as spm
import torch
import torch.nn as nn


#########################################################
# Tokenizer
#########################################################

#from code segmentation file

class Tokenizer:

    def __init__(self, filepath='python_tokenizer_30k.model'):
        self.sp = spm.SentencePieceProcessor(model_file=filepath)

    def encode(self, text, t=int):
        return self.sp.encode(text, out_type=t)

    def decode(self, pieces):
        return self.sp.decode(pieces)

    @staticmethod
    def train(input_file='data/raw_sents.txt', model_prefix='sp_model', vocab_size=30522):
        spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix, vocab_size=vocab_size,
                                       #input_sentence_size=2 ** 16, shuffle_input_sentence=True)
                                       input_sentence_size=number_of_lines, shuffle_input_sentence=True)
        
        #instantiate tokenizer model
tokenizer = Tokenizer('python_tokenizer.model')

#########################################################
# BiLSTM - Segmenetation Model
#########################################################


###
# adapted from the PyTorch examples. for the full PyTorch LM example, see: 
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
###
class LSTM_LM(nn.Module):
    """Model feeds pre-trained embeddings through a series of biLSTM
       layers, followed by a linear vocabulary decoder."""
    
    def __init__(self, in_dim, hidden_dim, lstm_layers, word_vectors, 
                 dropout=0.05, bidirectional = True):
        super(LSTM_LM, self).__init__()

        self.vocab_size = word_vectors.shape[0]
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # blank embed layer starting from GloVe pre-trained vectors
        self._embed = nn.Embedding.from_pretrained(word_vectors, freeze=False)        
        self._drop = nn.Dropout(dropout)

        self._lstm = nn.LSTM(in_dim, hidden_dim, num_layers = lstm_layers, dropout = dropout,
                             bidirectional = bidirectional, batch_first=True)
        self._ReLU = nn.ReLU()
        self._pred = nn.Linear((2 if bidirectional else 1)*hidden_dim, 
                               #self.vocab_size)
                               1) #only 1 or zeros here 

    def forward(self, x):
        e = self._drop(self._embed(x))
        z, h = self._lstm(e)
        z_drop = self._drop(z)
        s = self._pred(self._ReLU(z_drop))
        #s = s.view(-1, self.vocab_size)
        s = s.squeeze()
        return s, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.lstm_layers, batch_size, self.hidden_dim)

#########################################################
# BiLSTM - Segmenetation Model
#########################################################

########
## use saved model
#######

def initalize_model():
    torch.manual_seed(691)

    #vocab size from sentence peice
    #vocab dim???? 
    vocab_size = 10000 #same as sentence peice
    vocab_dim = 50  # the size of our pre-trained word vectors

    # randomly initialize our word vectors!
    vocab_dim = 256
    word_vectors = torch.randn(vocab_size, vocab_dim)
    word_vectors.shape, word_vectors

    #set up model
    hidden_dim = 200
    lstm_layers = 2
    LSTM_LM_net_trained = LSTM_LM(word_vectors.shape[1], hidden_dim,lstm_layers, word_vectors)

    #[TODO]: fix so it works
    #https://pytorch.org/tutorials/beginner/saving_loading_models.html
    #https://stackoverflow.com/questions/61242966/pytorch-attributeerror-function-object-has-no-attribute-copy

    #load weights into model
    LSTM_LM_net_trained.load_state_dict(torch.load('biLSTM_LM.pt'))
    LSTM_LM_net_trained.eval()

    return LSTM_LM_net_trained


#########################################################
# BiLSTM - Segmenetation Model
#########################################################

# Python code to
# demonstrate readlines()
#get all python files and associated task
def get_GCJ_code():
    path = 'example.py'
    code = ''
    # Using readlines()
    file1 = open(path, 'r')
    Lines = file1.readlines()
 
    count = 0
    # Strips the newline character
    for line in Lines:
        code+=line
    return code



#########################################################
# BiLSTM - Segmenetation Model
#########################################################

def centered_sliding_window(token_list, window_diamiter,encode=False,PAD='unk'):
    windows = []
    for i in range(len(token_list)):
        
        #print(token_list)
        #input()
        
        window = []
        
        #if we have to pad the begining
        if i < window_diamiter:
            before_len = window_diamiter-i
            before = [PAD]*before_len+token_list[0:i]
        else:
            before = token_list[i-window_diamiter:i]
        
        #if we have to pad the end
        if i+window_diamiter>=len(token_list):
            after_len = (i+1+window_diamiter)-len(token_list)
            after = token_list[i+1:i+1+window_diamiter]+[PAD]*after_len

        else:
            after = token_list[i+1:i+1+window_diamiter]
        
        #put it togeather
        #print('------')
        #print('before:',before)
        #print('center:',token_list[i])
        #print('after:',after)
        window = before + [token_list[i]] + after
        #for encoding code if we want
        if encode:
            new_window = []
            #print(window)
            #input()
            for i in window:
                encoded = tokenizer.encode(i)
                if len(encoded)>1:       
                    x=encoded[1]
                    if type(x)==list:
                        new_window.append(x[0])
                    else:
                        new_window.append(x)
                elif len(encoded)==1:
                    if type(encoded)==list:
                        new_window.append(encoded[0])
                    else:
                        new_window.append(encoded)
                else:
                    #for some reason it finds the unicode stuff __
                    pass
                    #print(window)
                    #print(i)
                    #print(encoded)
                    #input()
            #print(window)
            #print(len(window))
            #print(len(tokenizer.decode(window)))
            #print(tokenizer.decode(window))
            #print(len(tokenizer.encode(window)))
            #window = tokenizer.encode(tokenizer.decode(window))
            window = new_window
        #print(window)
        #print(len(window))
        #input()

        #save windowz
        windows.append(window)
    
    return windows



#########################################################
# BiLSTM - Segmenetation Model
#########################################################

#NOTE, THIS ONLY GETS THE PREDICTED BREAK POINTS FROM A PREDICTION
#WITH THE NEWLINE TOKEN AT CENTER OF WINDOW
def get_predicted_break_points(code_windows, model):
    start = 0
    code  =[]
    break_points = []
    print(len(code_windows))
    for window_i in range(len(code_windows)):
        #get window, which has our tokens
        window = code_windows[window_i]
        window_predictions = get_window_predictions(window,model)
        #mid = math.ceil(len(window)/2)
        mid = int(len(window)/2) #actually we need to round down...
        mid_token = tokenizer.decode(int(window[mid]))
        mid_pred = window_predictions[mid]
        if mid_token[-7:]=='NEWLINE':
            if round(mid_pred) == 1:
                break_points.append(window_i)
                
                
        code.append(mid_token)
        start+=1
    return code, break_points
    
#code_windows = segments['0']['x']
#code, breaks = get_predicted_break_points(code_windows,LSTM_LM_net_trained)
#print(breaks)


#########################################################
# BiLSTM - Segmenetation Model
#########################################################
#gets all predictions from one window
def get_window_predictions(window, model):
    preds, h = model(torch.tensor([window]))
    preds = torch.flatten(torch.sigmoid(preds))
    preds = preds.detach().numpy()
    return preds


#########################################################
# BiLSTM - Segmenetation Model
#########################################################

#from code segmentation file
def insert_comments(code, break_spots, comment='\n'+'###'*8+'\n',at_begining=True):
    #if there is a a comment at begining of snippet
    if at_begining:
        #adds a notation to add a 0
        #at beigning of break spots too
        break_spots.insert(0,0)
    
    #go through breaks backwards
    #so as not to mess up break 
    #spots as we would if we went forward
    for b in break_spots[::-1]:
        code.insert(b,comment)
    return code


#########################################################
# BiLSTM - Segmenetation Model
#########################################################

def segment(code=None):
    LSTM_LM_net_trained = initalize_model()
    if code==None:
        code = get_GCJ_code()
    
    
    new_code = code.replace(' ',' SPACE')
    #newline
    new_code = new_code.replace('\n',' NEWLINE')
    #tab
    new_code = new_code.replace('\t',' TAB')

    tokens = tokenizer.encode(new_code,t=str)


    wd=20 #window diameter
    X_windows = centered_sliding_window(tokens,wd,encode=True)
    code, breaks = get_predicted_break_points(X_windows,LSTM_LM_net_trained)

    #from code segmentation file
    comments_added = insert_comments(code,breaks)
    comments_added_decoded = tokenizer.decode(comments_added)
    comments_added_token_string = ''.join(comments_added_decoded)
    comments_added_token_string = comments_added_token_string.replace('SPACE',' ')
    comments_added_token_string = comments_added_token_string.replace('NEWLINE','\n')
    comments_added_token_string = comments_added_token_string.replace('TAB','\t')
    return comments_added_token_string

if __name__ == "__main__":
    print(segment())

















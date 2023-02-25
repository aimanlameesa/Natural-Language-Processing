import torch
from torch import nn
from torchtext.datasets import SST2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

###################### Setting tokeniner, vocab, and text pipeline ######################
tokenizer = get_tokenizer('spacy', language='en_core_web_md')

train = SST2(split='train')

def yield_tokens(data_iter): 
    for text, _  in data_iter:
        yield tokenizer(text)
        
vocab = build_vocab_from_iterator(yield_tokens(train), specials=['<unk>', '<pad>',
                                                                 '<bos>', '<eos>'])

vocab.set_default_index(vocab["<unk>"]) 

pad_idx = vocab['<pad>']

text_pipeline  = lambda x: vocab(tokenizer(x))

################################# Creating model #################################
class LSTM(nn.Module):
    
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, 
                 dropout):
        super().__init__()
        #input dim = how many vocab you have
        #emb dim = 300 --> we use fasttext
        #padding_idx tells this lookup table to ignore, and just randomize....
        #<unk>, <bos>, <eos>
        self.embedding_layer = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm            = nn.LSTM(emb_dim,
                                       hid_dim,
                                       num_layers = num_layers,
                                       bidirectional = bidirectional,
                                       dropout = dropout,  #dropout is applied between layers....
                                       batch_first=True)
        
        self.fc              = nn.Linear(hid_dim * 2, output_dim)
        
    def forward(self, x, lengths):
        #x: [batch size, seq len]
        
        embedded_x = self.embedding_layer(x)
        #x: [batch size, seq len, emb dim]
        
        #pack this embedded_x in such a way that RNN knows to ignore padding....
        #without batch_first = True; things will become [seq len, batch size, emb dim]
        pack_embedded = nn.utils.rnn.pack_padded_sequence(embedded_x, lengths.to('cpu'),
                                                          enforce_sorted=False,
                                                          batch_first = True
                                                          )
        
        #packed_outputs is basically all hidden states
        #h is the last hidden state
        #c is the last cell state
        packed_outputs, (h, _) = self.lstm(pack_embedded)
        
        #h: [num_layers * num_directions, batch_size, hidden dim]
        
        #it happens that because packed_outputs is all hidden states....some hidden states near the end is
        #hidden state for padding, pytorch guys help you
        #by using this pad_packed_sequence, then all the hidden states will only be not padding....
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first = True)
        #output: [batch size, seq len, direction * hidden sim]
        
        #last hidden state - concat last forward and backward states
        last_hidden_state = torch.cat((h[-1, :, :], h[-2, :, :]), dim = 1)
        #last_hidden_state: [batch_size, hidden_dim * 2]
        
        #for sentiment analysis.....what should i sent to my linear layer...
        return self.fc(last_hidden_state)  #[batch_size, output_dim]==> [batch_size, 4]

################################## Instance of model ##################################
input_dim  = 13844
hid_dim    = 256
emb_dim    = 300
output_dim = 2 # two classes

#for biLSTM
num_layers = 2
bidirectional = True
dropout = 0.5

model = LSTM(input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout)

################################## Loading trained model ##################################
path = f'{model.__class__.__name__}.pt'
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

# Making sure model uses cpu
if next(model.parameters()).is_cuda:
    model = model.cpu()

def _predict(text, text_length):
    with torch.no_grad():
        output = model(text, text_length).squeeze(1)
        predicted = torch.max(output.data, 1)[1]
        return predicted

def predict_sentiment(title):
    text = torch.tensor(text_pipeline(title)) # Tokenize and numericalize text
    text = text.reshape(1, -1)
    text_length = torch.tensor([text.size(1)]).to(dtype=torch.int64)
    senti_pred = _predict(text, text_length) # Predict sentiment
    
    return int(senti_pred[0])

print("Success")
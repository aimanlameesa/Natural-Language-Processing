import torch, torchdata, torchtext
from torch import nn
import torch.nn.functional as F

import random, math, time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
        # attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]
        
        return x, attention
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        # x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim]
        
        return x
    
class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, logProb, length):
        self.prevNode = previousNode  # where does it come from
        self.wordid   = wordId  #  numericalized integer of the word
        self.logp     = logProb  # the log probability
        self.len      = length  # the current length; first word starts at 1

    def eval(self, alpha=0.7):
        # the score will be simply the log probability penaltized by the length 
        # we are adding some small number to avoid division error
        # read https://arxiv.org/abs/1808.10006 to understand how alpha is selected
        return self.logp / float(self.len + 1e-6) ** (alpha)
    
    # this is the function for comparing between two beamsearchnodes, whether which one is better
    # it is called when you called "put"
    def __lt__(self, other):
        return self.len < other.len

    def __gt__(self, other):
        return self.len > other.len
    

from queue import PriorityQueue
import operator

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, 
                 pf_dim, dropout, device, pad_idx, max_length = 100):
                
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    
        self.pad_idx = pad_idx
    
    def make_mask(self, x):
        
        # x = [batch size, len]
        
        pad_mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # pad_mask = [batch size, 1, 1, len]
        
        x_len = x.shape[1]
        
        sub_mask = torch.tril(torch.ones((x_len, x_len), device = self.device)).bool()
        # sub_mask = [len, len]
            
        mask = pad_mask & sub_mask
        # mask = [batch size, 1, len, len]
        
        return mask 
    
    def forward(self, x):
        
        # x = [batch size, len]
                
        batch_size = x.shape[0]
        x_len    = x.shape[1]
        
        # getting mask here since we remove seq2seq class
        mask   = self.make_mask(x)
        # mask = [batch size, 1, len, len]

        pos = torch.arange(0, x_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)          
            
        x = self.dropout((self.tok_embedding(x) * self.scale) + self.pos_embedding(pos))
        # x = [batch size, len, hid dim]
        
        for layer in self.layers:
            x, attention = layer(x, mask)
        
        # x = [batch size, len, hid dim]
        # attention = [batch size, n heads, len, len]
        
        output = self.fc_out(x)
        # output = [batch size, len, output dim]
            
        return output, attention

    def beam_decode(self, src_tensor, method='beam-search'):
        
        # src_tensor = [batch size, src len]
        src_len = src_tensor.shape[1]
        
        # how many parallel searches
        beam_width = 3
        
        # how many sentence do you want to generate
        topk = 1  
        
        # final generated sentence
        decoded_batch = []
                                        
        # starting with the start of the sentence token
        decoder_input = torch.LongTensor([SOS_IDX]).to(device)

        # number of sentences to generate
        endnodes = []  # holding the nodes of EOS, so we can backtrack
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(None, decoder_input, 0, 1)
        nodes = PriorityQueue()  # this is a min-heap

        # starting the queue
        nodes.put((-node.eval(), node))  # we need to put - because PriorityQueue is a min-heap
        qsize = 1

        # starting beam search
        while True:
            # giving up when decoding takes too long
            if qsize > 100: break
            
            # print(f"{nodes.queue=}")

            # fetching the best node
            # the score is log p divides by the length scaled by some constants
            score, n       = nodes.get()
            decoder_input  = n.wordid

            # getting all the previous nodes to construct a complete decoder input
            # because Transformer decoder expects the whole sentence
            prevNode = n.prevNode
            while prevNode != None:
                prev_word = torch.LongTensor([prevNode.wordid]).to(device)
                # print(f"{prev_word=}")
                decoder_input = torch.cat((decoder_input, prev_word))
                prevNode = prevNode.prevNode

            inv_idx       = torch.arange(decoder_input.size(0)-1, -1, -1).long()
            decoder_input = decoder_input[inv_idx]

            # wordid is simply the numercalized integer of the word
            current_len    = n.len

            decoder_input  = decoder_input.unsqueeze(0)
            # decoder_input: batch_size, src_len

            if n.wordid.item() == EOS_IDX and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            # decoder_input = SOS_IDX
            # mask = [1, src len]
            decoder_input = F.pad(decoder_input, pad=(0, src_len), mode='constant', value=PAD_IDX)
            # padding because our decoder expects a whole sentence, not one token by token....

#             print(f"{current_len=}")
#             print(f"{decoder_input=}")

            prediction, _ = self.forward(decoder_input)
            # prediction   = [batch size, src len, output dim]

            prediction = prediction[:, current_len, :] # getting only the next word, but ignoring the padding
            # prediction   = [batch size, output dim]

            # so basically prediction is probabilities across all possible vocab
            # we gonna retrieve k top probabilities (which is defined by beam_width) and their indexes
            # we recall that beam_width defines how many parallel searches we want
            log_prob, indexes = torch.topk(prediction, beam_width)
            # log_prob      = (1, beam width)
            # indexes       = (1, beam width)
            
            # print(f"{log_prob.shape}")
            # print(f"{indexes.shape}")

            nextnodes = []  # the next possible node we can move to

            # we only select beam_width amount of nextnodes
            for top in range(beam_width):
                pred_t = indexes[0, top].reshape(-1)  # reshaping because wordid is assume to be []; see when we define SOS
                log_p  = log_prob[0, top].item()

                # decoder previous node, current node, prob, length
                node = BeamSearchNode(n, pred_t, n.logp + log_p, n.len + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # putting them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increasing qsize
            qsize += len(nextnodes) - 1


        ### Once everything is finished, we choose nbest paths and back trace them.

        ## in case it does not finish, we simply get couple of nodes with highest probability
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        # looking from the end and go back....
        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back tracing by looking at the previous nodes.....
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]  # reversing it....
            utterances.append(utterance) # appending to the list of sentences....

        decoded_batch.append(utterances)

        return decoded_batch  # (batch size, length)
    
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)        
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        
        # x = [batch size, len, hid dim]
        # mask = [batch size, 1, len, len]
        
        # multi attention, skip and then norm
        _x, attention = self.self_attention(x, x, x, mask)
        x = self.self_attn_layer_norm(x + self.dropout(_x))
        # x = [batch size, len, hid dim]
        # attention = [batch size, n heads, len, len]
    
        # positionwise feedforward
        _x = self.positionwise_feedforward(x)
        x = self.ff_layer_norm(x + self.dropout(_x))
        # x = [batch size, len, hid dim]
        
        return x, attention
import torch
import pickle
from model import Decoder
from torchtext.data.utils import get_tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

# tokenizing
tokenizer = get_tokenizer('spacy', language='en_core_web_md')

# defining path
vocab_path = "C:/Users/aiman/Downloads/Code Generator (using Transformer)/vocab.pkl"
with open(vocab_path, 'rb') as handle:
    vocab = pickle.load(handle)
print(len(vocab))

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, _ = model(src)
            
            # prediction: [batch size, seq len, vocab size]
            # prediction[:, -1]: [batch size, vocab size] # probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input
            
            #####################################################################
            # only pure sampling is done....
            # comparing with top-k, top-p, and beam search can be done here
            #####################################################################

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

def predict(prompt, temperature=0.5):
    max_seq_len = 10
    seed = 0
                # superdiverse       more diverse
    # temperatures = [0.5, 0.7, 0.75, 0.8, 1.0] 
    # sample from this distribution higher probability will get more change
    # for temperature in temperatures:
    #     generation = generate(prompt, max_seq_len, temperature, model, tokenizer, 
    #                         vocab, device, seed)
    #     print(str(temperature)+'\n'+' '.join(generation)+'\n')
    generation = generation = generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed)
    return ' '.join(generation)

vocab_size = len(vocab)
hid_dim    = 256                
dec_layers = 3               
dec_heads  = 8
dec_pf_dim = 512
dec_dropout = 0.1     
lr = 1e-3                     
model = Decoder(vocab_size, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, device, PAD_IDX).to(device)
save_path = "C:/Users/aiman/Downloads/Code Generator (using Transformer)/best-val-tr_lm.pt"
model.load_state_dict(torch.load((save_path), map_location=device))
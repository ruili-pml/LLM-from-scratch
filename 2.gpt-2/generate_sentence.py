from model import GPT
import torch
import tiktoken
import torch.nn.functional as F

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')

### prefix tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype = torch.long) # [input_len, ]
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # [num_return_sequences, input_len]
x = tokens.to('cuda')

### generate text
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# generate sentences in token
while x.size(1) < max_length:
    with torch.no_grad():
        # get the prob over vocab
        logits = model(x) # [num_return_sequences, T, vocab_size]
        logits = logits[:, -1, :] # take last logits, [num_return_sequences, vocab_size]
        probs = F.softmax(logits, dim = -1)
        
        # get the k possible tokens with the largest probablity
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # [num_return_sequences, 50]
        # sample one token for each sentence
        ix = torch.multinomial(topk_probs, 1) # [num_return_sequences, 1]
        # get the sampled token idx
        xcol = torch.gather(topk_indices, -1, ix) # [num_return_sequences, 1]
        # append to the sequence
        x = torch.cat([x, xcol], dim = 1)

# print out sampled sentences
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
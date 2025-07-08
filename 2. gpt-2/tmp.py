# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# let's look at the first 1000 characters
print(text[:300])

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create a mapping from characters to integers
char_to_int = {char : i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}

def encode(input_string):
    return [char_to_int[char] for char in input_string]

def decode(input_list):
    decoded_chars = [int_to_char[idx] for idx in input_list]
    return "".join(decoded_chars)

print(encode("hii there"))
print(decode(encode("hii there")))


# tokenise input
import torch
data = torch.tensor(encode(text), dtype = torch.long)

n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n: ]

# set up context length
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]

print(f"input sequence {x}")

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"for input {context}, target is {target}")
    
# batch

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    
    data = train_data if split =='train' else val_data
    
    # draw a number from [0, 1, ..., len(data) - block_size -1] batch_size times
    data_start_idx = torch.randint(len(data) - block_size, (batch_size, )) 
    
    x = torch.stack([data[i:i+block_size] for i in data_start_idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in data_start_idx])
    
    return x, y

xb, yb = get_batch('train')

print(xb)
print(yb)

# self-attention in spelled-out format
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
    
from model import GPT, GPTConfig
import torch
import tiktoken
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
################## overfit a batch ##################

config = GPTConfig()
model = GPT(config)

B, T = 4, 32

enc = tiktoken.get_encoding('gpt2')
with open("shakespeare.txt", "r") as f:
    text = f.read()

text = text[:1000]
tokens = enc.encode(text)

buffer = torch.tensor(tokens[:B * T + 1])
x = buffer[:-1].view(B, T)
y = buffer[1:].view(B, T)

x = x.to(device)
y = y.to(device)
model.to(device)

"""
## simple training loop
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
for i in range(50):
    optimizer.zero_grad()
    
    logits, loss = model(x, y)
    loss.backward()    
    optimizer.step()
    
    print(f"step {i}, loss {loss.item()}")
"""

## optimised training loop
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50 

def get_lr(cur_step):
    # 1) linear warmup for warmup_iters steps
    if cur_step < warmup_steps:
        return max_lr * (cur_step + 1) / warmup_steps
    # 2) if cur_step > lr_decay_iters, return min learning rate
    if cur_step > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (cur_step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


weight_decay = 1e-3
lr = 3e-4
# weight decay only 2d vector
optimizer = model.configure_optimizers(weight_decay, lr, device)

# complie
#model = torch.compile(model)

grad_accum_steps = 32

for cur_step in range(max_steps):
    optimizer.zero_grad()
    
    loss_accum = 0.0
    # gradeint accumulation
    for micro_step in range(grad_accum_steps):
        # mixed precision
        with torch.autocast(device_type = device, dtype = torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # CE average over the number of batch
        loss_accum += loss.detach()
        loss.backward()    
    # gradient clip
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # learning rate schedule
    lr = get_lr(cur_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    print(f"step {cur_step:4d} | loss {loss_accum.item():.6f} | norm: {norm:.4f}")

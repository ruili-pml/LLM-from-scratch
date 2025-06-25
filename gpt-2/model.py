from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
# --------------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class SingleHeadCausalAttention(nn.Module):
    def __init__(self, config, keep_bias_term = True):
        super().__init__()
        
        self.dim = int(config.n_embd / config.n_head)
        
        self.q_proj = nn.Linear(config.n_embd, self.dim, bias = keep_bias_term)
        self.v_proj = nn.Linear(config.n_embd, self.dim, bias = keep_bias_term)
        self.k_proj = nn.Linear(config.n_embd, self.dim, bias = keep_bias_term)
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))
    def forward(self, x):
        
        B, T, C = x.size()
        
        q = self.q_proj(x) # [B, T, hs]
        k = self.k_proj(x) # [B, T, hs]
        v = self.v_proj(x) # [B, T, hs]
        
        att =  q @ k.transpose(1, 2) * (1.0 / math.sqrt(self.dim))
            # [B, T, T] i-th row, the weights for token i
        att = att.masked_fill(self.bias[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1) # softmax for each row
        
        z = att @ v # [B, T, hs]
        
        return z

class MultiHeadCausalAttention(nn.Module):
    
    def __init__(self, config, keep_bias_term = True):
        super().__init__()
        
        self.multi_head_attention = nn.ModuleList([SingleHeadCausalAttention(config,keep_bias_term) for _ in range(config.n_head)])
        
        self.multi_head_projection = nn.Linear(config.n_embd, config.n_embd, bias = keep_bias_term)
        
    def forward(self, x):
        
        concatenaed_multi_head = torch.cat([h(x) for h in self.multi_head_attention], dim = -1)
        projected_multi_head = self.multi_head_projection(concatenaed_multi_head)
        
        return projected_multi_head


class CausalSelfAttention(nn.Module):
    
    def __init__(self, config, keep_bias_term = True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias = keep_bias_term)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = keep_bias_term)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        
        B, T, C = x.size()
        
        qkv = self.c_attn(x) # [B, T, 3 * C]
        q, k, v = qkv.split(self.n_embd, dim = 2)  # [B, T, C]
                 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
            # [B, T, C] -> [B, T, nh, hs] -> [B, nh, T, hs]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
            # [B, T, C] -> [B, T, nh, hs] -> [B, nh, T, hs]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
            # [B, T, C] -> [B, T, nh, hs] -> [B, nh, T, hs]
        
        att = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(k.size(-1)))
            # [B, nh, T, hs] @ [B, nh, hs, T] -> [B, nh, T, T]
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        
        y = att @ v # [B, nh, T, T] @ [B, nh, T, hs] -> [B, nh, T, hs]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.c_proj(y)
        
        return y

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approxiamte = 'tanh') # to be consistent with original GPT
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
    
    def forward(self, x):
        
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        
        return x

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x
        

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpt = nn.Embedding(config.context_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),   
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
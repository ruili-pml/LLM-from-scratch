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
        self.gelu = nn.GELU(approximate = 'tanh') # to be consistent with original GPT
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
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),   
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        
        # weight tying scheme
        self.transformer.wte.weight = self.lm_head.weight  # about 30% params for 124M model

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, target = None):
        
        # idx is the token idx, of shape [B, T]. target is [B ,T] as well
        B, T = idx.size()
        assert T <= self.config.block_size
        
        # get the token embedding
        tok_emb = self.transformer.wte(idx) # [B, T, n_embd]
        # get the position embedding
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device) # [T, ]
        pos_emb = self.transformer.wpe(pos) # [T, n_embd]
        # input embedding
        x = tok_emb + pos_emb
        
        # forward
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x) # [B, T, vocab_size]
        
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   target.view(-1))
            
        return logits, loss
        
    @classmethod
    def from_pretrained(cls, model_type):
        
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained {model_type}")
        
        config_args = {
            'gpt2':         dict(n_layer = 12, n_head = 12, n_embd = 768),
            'gpt2-medium':  dict(n_layer = 24, n_head = 16, n_embd = 1024),
            'gpt2-large':   dict(n_layer = 36, n_head = 20, n_embd = 1280),
            'gpt2-xl':      dict(n_layer = 48, n_head = 25, n_embd = 1600),
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the mask buffer
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


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
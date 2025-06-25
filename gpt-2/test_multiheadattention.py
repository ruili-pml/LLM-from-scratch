from model import SingleHeadCausalAttention, MultiHeadCausalAttention, CausalSelfAttention, GPTConfig
import torch

slow_multi_head_attention = MultiHeadCausalAttention(GPTConfig, keep_bias_term = False)
fast_multi_head_attention = CausalSelfAttention(GPTConfig, keep_bias_term = False)


# --------------------------------------------------------------------------
# Layout of the “big” QKV weight matrix expected by each implementation
#
#   ┌───────────── slow path (“hand-rolled” heads) ────────────────┐
#   │      Head 1     |      Head 2     |     …    |     Head H    │
#   │    Q1  K1  V1   |    Q2  K2  V2   |     …    |   QH  KH  VH  │
#   └──────────────────────────────────────────────────────────────┘

#
#   ┌──────── fast path (CausalSelfAttention.c_attn) ─────────┐
#   │       ALL Q       │        ALL K       │       ALL V    │
#   │    Q1 Q2 …  QH    |    K1 K2  …  KH    |  V1 V2  … VH   │     
#   └─────────────────────────────────────────────────────────┘
# --------------------------------------------------------------------------

all_W_q = torch.cat([single_head.q_proj.weight.data for single_head in slow_multi_head_attention.multi_head_attention], dim=0)
all_W_k = torch.cat([single_head.k_proj.weight.data for single_head in slow_multi_head_attention.multi_head_attention], dim=0)
all_W_v = torch.cat([single_head.v_proj.weight.data for single_head in slow_multi_head_attention.multi_head_attention], dim=0)

concatenate_multihead_qkv = torch.cat([all_W_q, all_W_k, all_W_v], dim = 0)

fast_multi_head_attention.c_attn.weight.data = concatenate_multihead_qkv
fast_multi_head_attention.c_proj.weight.data = slow_multi_head_attention.multi_head_projection.weight.data


B, T, C = 2, 5, GPTConfig.n_embd
torch.manual_seed(42)
test_input = torch.rand((B, T, C))

slow_output = slow_multi_head_attention(test_input)
fast_output = fast_multi_head_attention(test_input)

torch.testing.assert_close(slow_output, fast_output)
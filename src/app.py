import torch

from src.attention.multiHeadAttention import MultiHeadAttention
from src.attention.multiHeadAttentionStacked import MultiHeadAttentionWrapper
from src.attention.selfAttention import SelfAttention_v1, SelfAttention_v2
from src.attention.simpleAttention import SimpleAttention
from src.attention.casualAttention import  CasualAttention

input = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# # calculate simple attention weights
# attn = SimpleAttention(input)
# attn_weights = attn.get_attention_weights()
# print(attn_weights)

# # calculate self attention weights
# torch.manual_seed(124)
# selfAttn = SelfAttention_v1(input.shape[1], 2)
# print(selfAttn(input))
#
# # calculate self attention weights
# torch.manual_seed(124)
# selfAttnV2 = SelfAttention_v2(input.shape[1], 2)
# print(selfAttnV2(input))


#################################################
# Casual Attention
#################################################

batch = torch.stack((input, input), dim=0)
print(batch.shape)
#
# torch.manual_seed(123)
# context_length = batch.shape[1]
# ca = CasualAttention(input.shape[1], 2, context_length, 0.0)
# context_vecs = ca(batch)
# print("context_vecs.shape:", context_vecs.shape)

#################################################
# Multi- Head Attention Stacked
#################################################

# torch.manual_seed(123)
# context_length = batch.shape[1]
# d_in, d_out = 3, 2
#
# mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
# context_vecs = mha(batch)
#
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)

#################################################
# Multi- Head Attention
#################################################

torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2

mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)










import torch

from src.attention.selfAttention import SelfAttention_v1, SelfAttention_v2
from src.attention.simpleAttention import SimpleAttention

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

# calculate self attention weights
torch.manual_seed(124)
selfAttn = SelfAttention_v1(input.shape[1], 2)
print(selfAttn(input))

# calculate self attention weights
torch.manual_seed(124)
selfAttnV2 = SelfAttention_v2(input.shape[1], 2)
print(selfAttnV2(input))
# Instead of maintaining two separate classes, MultiHeadAttentionWrapper and CausalAttention,
# we can combine these concepts into a single MultiHeadAttention class.
# This class is more efficient than multiHeadAttentionStacked as we only need one matrix multiplication to compute the keys
# In the other class we needed to repeat this matrix multiplication, which is computationally one of the most expensive steps,
# for each attention head.

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # Rather than processing all the dimensions of a token through all the heads, we divide the dimensions of a token
        # Each head processes all tokens, but only a slice of each token's features/dimensions (head_dim).
        # This lets different heads learn specialized views over the full sequence. head_dim contains the # of dims read by a head
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # This is the final linear projection that comes after all attention heads have done their job
        # and their outputs have been concatenated.
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Project the input sequence into queries, keys, and values using shared linear layers.
        # These layers produce a [batch, seq_len, d_out] tensor for each type.
        # At this point, the outputs are still shared across all heads and not yet separated.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape keys, values and queries from [batch, seq_len, d_out] to [batch, seq_len, num_heads, head_dim],
        # so each head gets its own chunk of features per token.
        # Reshaping the Q, K, and V tensors so that each attention head can read and process its own slice of the data independently.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Rearrange dimensions from [batch, seq_len, num_heads, head_dim] to
        # [batch, num_heads, seq_len, head_dim] so that each head has its own
        # contiguous view of the entire sequence. This allows parallel attention
        # computation across heads using efficient batched matrix operations.
        # See example in multiHead scratchpad
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #Compute raw attention scores via dot product between queries and keys.
        # We use keys.transpose(2, 3) to swap the last two dimensions:
        # from [batch, num_heads, seq_len, head_dim] to [batch, num_heads, head_dim, seq_len],
        # so the inner dimensions align correctly for batched matrix multiplication.
        attn_scores = queries @ keys.transpose(2, 3)

        # Convert the causal mask to boolean and trim it to the actual sequence length.
        # This ensures we only mask future tokens for the current input size, not the full context window.
        # True = mask (set to -inf later), False = keep.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute the context vector for each head by applying the attention weights to the values.
        # Shape: [batch, num_heads, seq_len, head_dim] → then transpose to
        # [batch, seq_len, num_heads, head_dim] to prepare for merging heads.
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Flatten the multiple heads into a single vector per token.
        # Shape goes from [batch, seq_len, num_heads, head_dim] to [batch, seq_len, d_out],
        # where d_out = num_heads * head_dim. `.contiguous()` ensures safe reshaping after transpose.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 11

        return context_vec
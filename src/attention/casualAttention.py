import torch
import torch.nn as nn

# Notes: GPUs are much faster for tensor operations, especially large matrix multiplications used in attention, transformers,
# and neural networks in general. Therefore, deep learning models are usually trained on GPU's
# We move our model and data to GPU with:
#       model.to('cuda')
#       input = input.to('cuda')
# If any part of our model (parameters, inputs, masks, etc.) is still on CPU, we get a device mismatch error.
# That's why we are registering our mask as a buffer

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # added dropout layer

        # Register the causal attention mask as a buffer. This mask prevents attending to future positions.
        # Using `register_buffer` ensures the mask is part of the model’s state (i.e., saved with state_dict),
        # but it is not a trainable parameter. Most importantly, it will automatically move to the correct
        # device (CPU or GPU) when we call `model.to(device)`, avoiding device mismatch errors during training
        # or inference. Without this, we'd have to manually move the tensor and could easily forget.
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
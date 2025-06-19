# Note: The way simple embedding  works is that the same token ID always gets mapped to the same vector representation,
# regardless of where the token ID is positioned in the input sequence.
# Positional embeddings give transformers a sense of sequence, enabling them to understand word order.
import torch

from src.dataloader.dataloader import GPTDataLoader
from src.utils.utils import get_raw_text

raw_text = get_raw_text()

max_length = 4
dataloader = GPTDataLoader(
    raw_text,
    batch_size=8, # Each batch yielded by the dataloader will contain exactly 1 sequence (or sample) of length max_length tokens.
    max_length=max_length,
    stride=max_length, # dictates the number of positions the inputs shift across batches, emulating a sliding window approach
    shuffle=False
)

# vocab size of BPE tokenizer
vocab_size = 50257
output_dim = 3
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) #  Creates an embedding layer that maps each token ID to a 256-dimensional vector.

data_iter = iter(dataloader)
inputs, targets = next(data_iter) # returns next batch of input and target token ID's
print("Token IDs:\n", inputs) # this token is of size 8 * 4 (8 batches of max length 4)
print("Inputs shape:", inputs.shape)  # dimensions of input - 8 * 4

token_embeddings = token_embedding_layer(inputs) # returns embedding for each token - which is now of 256 dimension
print("Token Embeddings Shape: ", token_embeddings.shape)


# Now we add absolute embedding layer - to add positional data to each token, to let transformer know about recurring sequences of same token
# first we create a lookup table of random values of shape context length = 4 * dimensions = 3
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# creates a 1D tensor of sequential integers starting from 0 up to (but not including) context_length.
# In this case it will create tensor([0, 1, 2, 3]), as each batch is of max length 4,
# so if in a batch all tokens are same, they will have a differnt positional value
pos_ids = torch.arange(context_length)
print("pos ids: ", pos_ids)

# lookup the vectors for these positions in the pos_embeddings tensor
pos_embeddings = pos_embedding_layer(pos_ids)
print("Positional Embeddings Shape: ", pos_embeddings.shape)

# We generate input embeddings by adding token embeddings in pos_embeddings
input_embeddings = token_embeddings + pos_embeddings
print("Final Input Embeddings: ", input_embeddings)

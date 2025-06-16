import torch

from src.dataloader.dataloader import GPTDataLoader
from src.utils.utils import get_raw_text

raw_text = get_raw_text()
dataloader = GPTDataLoader(
    raw_text,
    batch_size=1, # Each batch yielded by the dataloader will contain exactly 1 sequence (or sample) of length max_length tokens.
    max_length=4,
    stride=1, # dictates the number of positions the inputs shift across batches, emulating a sliding window approach
    shuffle=False
)

iterator = iter(dataloader)
first_batch = next(iterator)

print("First Batch: ", first_batch)

vocab_size = len(dataloader)
output_dimensions = 3
print("vocab_size: ", vocab_size)

# generate embedding layer
torch.manual_seed(777)
embedding_layer = torch.nn.Embedding(vocab_size, output_dimensions)

# print embeddings for first batch - inputs
first_input_set = first_batch[0]
print(embedding_layer(first_input_set))

print("-----------------------------")
print("New Data Loader")
# Note: The way above embedding layer works is that the same token ID always gets mapped to the same vector representation,
# regardless of where the token ID is positioned in the inpyut sequence

max_length = 4
new_dataloader = GPTDataLoader(
    raw_text,
    batch_size=8, # Each batch yielded by the dataloader will contain exactly 1 sequence (or sample) of length max_length tokens.
    max_length=max_length,
    stride=max_length, # dictates the number of positions the inputs shift across batches, emulating a sliding window approach
    shuffle=False
)

# vocab size of BPE tokenizer
vocab_size_new = 50257
output_dim_new = 256
token_embedding_layer = torch.nn.Embedding(vocab_size_new, output_dim_new) #  Creates an embedding layer that maps each token ID to a 256-dimensional vector.

data_iter = iter(new_dataloader)
inputs, targets = next(data_iter) # returns next batch of input and target token ID's
print("Token IDs:\n", inputs) # this token is of size 8 * 4 (8 batches of max length 4)
print("Inputs shape:", inputs.shape)  # dimensions of input - 8 * 4

token_embeddings = token_embedding_layer(inputs) # returns embedding for each token - which is now of 256 dimension
print(token_embeddings.shape)

# Now we add absolute embedding layer - to add positional data to each token, to let transformer know about recurring sequences of same token
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim_new)
# creates a 1D tensor of sequential integers starting from 0 up to (but not including) context_length.
# In this case it will create tensor([0, 1, 2, 3]), as each batch is of max length 4, so if in a batch all tokens are same, they will have a differnt positional value
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Positional Embeddings Shape: ", pos_embeddings.shape)


# We generate input embeddings by adding token embeddings in pos_embeddings
input_embeddings = token_embeddings + pos_embeddings



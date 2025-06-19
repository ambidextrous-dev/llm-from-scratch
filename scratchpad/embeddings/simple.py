# generate embeddings from input
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
print("Tokenized Input/Target Data (First Batch): ", first_batch)

vocab_size = len(dataloader)
output_dimensions = 3
print("vocab_size of tiktoken: ", vocab_size)


torch.manual_seed(777) # setting manual seed for reproducibility
# creates a learnable lookup table, which is conceptually a matrix with shape - [vocab_size, output_dimensions]
# Each token ID will be mapped to a corresponding vector
embedding_layer = torch.nn.Embedding(vocab_size, output_dimensions)


# lookup embeddings for first batch - inputs
first_input_set = first_batch[0]
print("Embeddings for input set: ", embedding_layer(first_input_set))

first_target_set = first_batch[1]
print("Embeddings for target set: ", embedding_layer(first_target_set))

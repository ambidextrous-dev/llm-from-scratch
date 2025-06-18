import torch

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dimensions = 3

torch.manual_seed(101)
embedding_layer = torch.nn.Embedding(vocab_size, output_dimensions)
print(embedding_layer.weight)


print("embedding for one token")
print(embedding_layer(torch.tensor([3]))) # fetch embedding for this particular token

print("embedding for multiple tokens")
print(embedding_layer(torch.tensor([2, 3, 5, 1])))
import torch

# assume we have embeddings for an input text
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# lets calculate context vector for one of the token - x2
query = inputs[1]
# create attention scores for all the tokens in the input in reference to x2
# we create an empty tensor and then initialize it with elements matching # of tokens, as we need to compute a relative attention score to every other token
attn_scores_2 = torch.empty(inputs.shape[0])

# to calculate attention scores we will compute a dot product between the reference token and all other tokens
# dot product => example:  a = [a1, a2, a3] , b = [b1, b2, b3] ; dot product = a1*b1 + a2*b2 + a3*b3
# the dot product is a measure of similarity because it quantifies how closely two vectors are aligned
# a higher dot product indicates a greater degree of alignment or similarity between the vectors
for index, token_embeddings in enumerate(inputs):
  attn_scores_2[index] = torch.dot(query, token_embeddings)

print("Attn Scores: ", attn_scores_2)

# now that we have attention scores, lets normalize them. Normalization means, the sum of all the attention scores should be 1
# useful for interpretation and maintaining training stability in an LLM
# Manual Normalization:

# temp_attn_scores_2_norm = attn_scores_2 / attn_scores_2.sum()
# print("Normalized Attn Scores: ", temp_attn_scores_2_norm)
# print("Sum of Normalized Tensor: ", temp_attn_scores_2_norm.sum())

# Rather than doing this manually, we can also use softmax to normalize our scores
# Softmax turns a list of raw scores (logits) into probabilities that:
#       - Are all non-negative
#       - Sum to 1
#       - Reflect the relative strength of each input score

attn_scores_2_normalized = torch.softmax(attn_scores_2, dim=0)
print("Softmax Normalized Attention weights:", attn_scores_2_normalized)

# Next step is to calculate context vector - multiply each embedding tokens with its respective attention weight and then sum this up
# weighted sum of all input vectors, obtained by multiplying each input vector by its corresponding attention weight

contex_vec_2 = torch.empty(query.shape)

for index, embed_token in enumerate(inputs):
 contex_vec_2 += embed_token * attn_scores_2_normalized[index]

# This is second context vector as the attention weights used were calculated based on the second input
print(contex_vec_2)


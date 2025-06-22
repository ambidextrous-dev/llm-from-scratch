# self attention with trainable weights
# we will introduce three matrices - query - Wq, key - Wk, and value - Wv
# We then use these matrices to project input tokens into query, key, value vectors which will be further used to calculate attention weights
# The query vector q(2) is obtained via matrix multiplication between the input x(2) and the weight matrix Wq.
# Similarly, we obtain the key and value vectors via matrix multiplication involving the weight matrices Wk and Wv.
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


# let's calculate attention weights with trainable weights for second input = journey
x_2 = inputs[1]
dim_in = x_2.shape[0]
#  in GPT-like models, the input and output dimensions are usually the same, but to better follow the computation,
#  we’ll use different input (d_in=3) and output (d_out=2) dimensions here
dim_out = 2

torch.manual_seed(123)
# torch.rand(d_in, d_out) creates three matrices of shape [d_in, d_out] filled with random values between 0 and 1.
# torch.nn.Parameter(...) wraps the tensor so that it's registered as a model parameter
# We set requires_grad=False to reduce clutter in the outputs, this indicates that this parameter will not be updated during backpropagation.
# But if we were to use the weight matrices for model training, we would set requires_grad=True to update these matrices during model training.
W_query = torch.nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=False)

# generate query, key and value vectors
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

#The output for the query results in a two-dimensional vector since we set the number of columns of the corresponding weight matrix, via dim_out, to 2
print(query_2)

# we can even do for all input tokens at once
# Even though our temporary goal is only to compute the one context vector, z(2), we still require the key and value vectors for
# all input elements as they are involved in computing the attention weights with respect to the query q(2)
query = inputs @ W_query
keys = inputs @ W_key
values = inputs @ W_value

print(query)
print("query shape: ", query.shape)
print("query[2] shape: ", query[1].shape)

# the attention score mechanism is similar to what we used in the simple attention mechanism
# But, instead of direct input values we use dot product between query and keys

# how much token 2 attends to itself
attn_score_2_2 = query[1].dot(keys[1])
print("attn score of 2 respect to 2: ", attn_score_2_2)

# calculate attn_scores of 2nd token with respect to all the tokens
# we need a matrix of dimensions (6 * 1) ==> query[1] is of shape = (2,1), key shape = (6 ,2)
# but the order of matters - we want query to dot with each key. Each value shows how much token 2 (journey) attends to token i.
# [q₂ · k₁, q₂ · k₂, q₂ · k₃, q₂ · k₄, q₂ · k₅, q₂ · k₆]
# "Token 2 compares its query with every token's key."

attn_scores_2 = query[1] @ keys.T
print("attn_scores_2: ", attn_scores_2)

# Now we will scale down our scores.
# The raw dot products between queries and keys can get very large in magnitude, especially when the dimensionality of the vectors (d_k) is high.
# This causes:
#    1. Very sharp softmax outputs (almost one-hot), leading to vanishing gradients, almost reaching zero
#    2. Unstable training
# So, we scale the scores by dividing the scores by square root of the embedding dimension of the keys
# The scaling by the square root of the embedding dimension is the reason why this self-attention mechanism is also called scaled-dot product attention.

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("attn weights w.r.t second token: ", attn_weights_2)

context_vec_2 = attn_weights_2 @ values
print("contxt_vector_2:", context_vec_2)

# Each token (word/vector) in the input acts like:
#      1) A query looking for related information in the sentence.
#      2) A key describing what information each token holds.
#      3) A value representing the actual content each token offers.
#
# So: Query = "What am I looking for?"    Key = "What do I offer?"      Value = "What do I contain?"
# Each word sends out a query, compares it to all keys in the sequence (even its own),
# and uses that to figure out how much of each value it should pull in to compute its final representation

#Even though all three (Q, K, V) are derived from the same input, they’re projected using different trainable matrices
# so the model can learn different roles:
# Q (Query): What to attend to.
# K (Key): How important each word is in response to a query.
# V (Value): The actual data used to build the output (contextualized representation).


import torch

class SimpleAttention:
    def __init__(self, input):
        self.input = input

    def get_attention_weights(self):
        # attn_scores will be a square n*n matrix where each i,j contains dot product of token's i and j
        attn_scores = torch.empty(self.input.shape[0], self.input.shape[0])

        # for i, x_i in enumerate(input):
        #     for j, x_j in enumerate(input):
        #         attn_scores[i,j] = torch.dot(x_i, x_j)

        # why transpose here?
        # cause it gives us the resulting matrix of 6 * 6 which we need - (6, 3) @ (3, 6) → (6, 6)
        attn_scores = self.input @ self.input.T # matrix multiplication , much faster than nested for loop

        # normalize the scores
        # By setting dim=-1, we are instructing the softmax function to apply the normalization along the last dimension of the attn_scores tensor.
        # If attn_scores is a two-dimensional tensor (for example, with a shape of [rows, columns]), it will normalize across the columns so that
        # the values in each row (summing over the column dimension) sum up to 1

        attn_scores_normalized = torch.softmax(attn_scores, dim=-1)

        # matrix multiplication = (6 * 6) @ (6, 3) = (6, 3) - same dimension which we need
        context_vectors = attn_scores_normalized @ self.input

        return context_vectors

# Instead of computing a single attention operation, multi-head attention computes multiple attention operations in parallel
#  — each with its own learned projections of the input.
# These multiple "heads" each focus on different parts of the input or different types of relationships,
# and their outputs are concatenated and projected into the final result.
# Multi-head attention = Multiple independent attention layers + final projection.

# Why Multi-Head?
# 1. Capture Different Types of Information
#    Each attention head can attend to different positions in the sequence. For example:
#    - One head might focus on syntax.
#    - Another might look at long-term dependencies.
#    - Another might focus on punctuation or structure.
# This diversity is richer than what a single attention layer could learn.
# 2. Improve Representational Power
#    - Instead of learning just one attention mapping, you let the model learn several, making the representation more expressive and robust.
#    - It's like having multiple "experts" analyzing the sentence from different angles.
# 3. Helps Parallelization
#    - Each head can be computed independently and in parallel, making it efficient on GPUs.

#######################################################################
# Tensor example in multiHead

# lets assume our original tensor is below:
# [  # sequence length = 2
#         [1.0, 2.0, 3.0, 4.0],   # token 0
#         [5.0, 6.0, 7.0, 8.0]    # token 1
#     ]

# we first reshape it as per our number of heads
#     [[1., 2.],   # head 0, token 0
#      [3., 4.]],  # head 1, token 0
#
#     [[5., 6.],   # head 0, token 1
#      [7., 8.]]   # head 1, token 1
# ]

# then we transpose to make it look like
# tensor([[
#     [[1., 2.],  # token 0 for head 0
#      [5., 6.]], # token 1 for head 0
#
#     [[3., 4.],  # token 0 for head 1
#      [7., 8.]]  # token 1 for head 1
# ]])

# So each head now has a full sequence of its own feature slice → ready for attention!
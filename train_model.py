# Some notes about alphago's RL policy network

# The strong policy network is a 13-layer convolutional network.
# All of these layers produce 19 × 19 filters;
# you consistently keep the original board size across the whole network.
# For this to work, you need to pad the inputs accordingly
# The first convolutional layer has a kernel size of 5,
# and all following layers work with a kernel size of 3.
# The last layer uses softmax activations and has one output filter,
# and the first 12 layers use ReLU activations and have 192 output filters each.

# There are 8 planes for liberties because the number of liberties is split across 8 binary planes

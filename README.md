# parsnet
TensorFlow implementation of the constraints necessary for [Parseval networks](https://arxiv.org/abs/1704.08847).

Parseval networks constrain the weight matrices of neural networks to be tight frames, so that the Lipschitz contant of the entire network is <= 1. Thus, the network will be stabel to adversarial noise.
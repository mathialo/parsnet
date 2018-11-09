from tensorflow.python.keras.constraints import Constraint
from tensorflow.python.ops import math_ops


class TightFrame(Constraint):
    """
    Parseval (tight) frame contstraint, as introduced in https://arxiv.org/abs/1704.08847

    Constraints the weight matrix to be a tight frame, so that the Lipschitz
    constant of the layer is <= 1. This increases the robustness of the network
    to adversarial noise.

    Args:
        beta (float):   Retraction parameter.

    Returns:
        Weight matrix after applying regularizer.
    """


    def __init__(self, beta):
        self.beta = beta


    def __call__(self, w):
        return (1 + self.beta) * w - self.beta * math_ops.matmul(
            w,
            math_ops.matmul(w, w, transpose_a=True))


    def get_config(self):
        return {'beta': self.beta}


# Alias
tight_frame = TightFrame

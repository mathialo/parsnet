from tensorflow.python.keras.constraints import Constraint
from tensorflow.python.ops import math_ops, array_ops


class TightFrame(Constraint):
    """
    Parseval (tight) frame contstraint, as introduced in https://arxiv.org/abs/1704.08847

    Constraints the weight matrix to be a tight frame, so that the Lipschitz
    constant of the layer is <= 1. This increases the robustness of the network
    to adversarial noise.

    Args:
        scale (float):   Retraction parameter.

    Returns:
        Weight matrix after applying regularizer.
    """


    def __init__(self, scale, num_passes):
        self.scale = scale

        if num_passes < 1:
            raise ValueError("Number of passes cannot be non-positive! (got {})".format(num_passes))
        self.num_passes = num_passes


    def __call__(self, w):
        transpose_channels = (len(w.shape) == 4)

        # Move channels to the front in order to make the dimensions correct for matmul
        if transpose_channels:
            w_reordered = array_ops.transpose(w, perm=[1, 2, 3, 0])
        else:
            w_reordered = array_ops.transpose(w)


        # Perform the projection
        for i in range(self.num_passes):
            result_reordered = (1 + self.scale) * w_reordered - self.scale * math_ops.matmul(
                w_reordered,
                math_ops.matmul(w_reordered, w_reordered, transpose_a=True))

        # Move channels to the back again
        if transpose_channels:
            return array_ops.transpose(result_reordered, perm=[3, 0, 1, 2])
        else:
            return array_ops.transpose(result_reordered)


    def get_config(self):
        return {'scale': self.scale, 'num_passes': self.num_passes}


# Alias
tight_frame = TightFrame

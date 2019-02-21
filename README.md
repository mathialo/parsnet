# parsnet
TensorFlow implementation of the constraints necessary for [Parseval networks](https://arxiv.org/abs/1704.08847).

Parseval networks constrain the weight matrices of neural networks to be tight frames, so that the Lipschitz constant of the entire network is <= 1. Thus, the network will be stable to adversarial noise.


## Example use
Using the `parsnet` package is very easy. Simply import `parsnet` and use the [`kernel_constraint`](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d#arguments) keyword argument on your layers of choice:
```
import tensorflow as tf
import parsnet


img_size = (32, 32, 3)
batch_size = 512
train_epochs = 100
retraction_par = 0.001
num_passes = 1

input_layer = tf.placeholder(tf.float32, shape=(batch_size, *img_size))

layer1 = tf.layers.conv2d(
    inputs=input_layer,
    kernel_size=(5, 5),
    filters=64,
    strides=(1, 1),
    padding="SAME",
    activation=tf.nn.relu,
    kernel_initializer=tf.initializers.orthogonal(),
    name="convlayer1",

    # Applying Parseval constraint:
    kernel_constraint=parsnet.constraints.tight_frame(beta, num_passes)
)

...

```
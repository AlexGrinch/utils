import tensorflow as tf
import tensorflow.contrib.layers as layers

from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.layers import base as base_layer

class CustomRNNCell(LayerRNNCell):
    """ Wrapper around LayerRNNCell to define custom rnn cell.
    
    Args:
        num_hidden: int, the number of units in RNN cell.
        num_outputs: int, the number of units in RNN outputs.
        activation_fn: Nonlinearity to use. Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
            
    This example implements standard RNN cell with:
    h_{t+1} = tanh(Ax_{t+1} + Bh_{t} + c)
    o_{t} = tanh(Dh_{t} + e)
    """       
    
    def __init__(self, num_hidden, num_outputs, activation_fn=None, reuse=None):
        super(CustomRNNCell, self).__init__(_reuse=reuse)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self._activation = activation_fn or tf.nn.tanh

    @property
    def state_size(self):
        return self.num_hidden

    @property
    def output_size(self):
        return self.num_outputs

    def build(self, inputs_shape):
        
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)
        self.num_inputs = inputs_shape[1].value
        
        # Cell (hidden state) parameters
        self.A = self.add_variable("input_to_hidden", shape=[self.num_inputs, self.num_hidden],
                                   initializer=tf.random_normal_initializer())
        self.B = self.add_variable("hidden_to_hidden", shape=[self.num_hidden, self.num_hidden],
                                   initializer=tf.random_normal_initializer())
        self.c = self.add_variable("next_hidden_bias", shape=[self.num_hidden],
                                   initializer=tf.random_normal_initializer())
        
        # Output parameters
        self.D = self.add_variable("hidden_to_output", shape=[self.num_hidden, self.num_outputs],
                                   initializer=tf.random_normal_initializer())
        self.e = self.add_variable("output_bias", shape=[self.num_outputs],
                                   initializer=tf.random_normal_initializer())
        
        self.built = True

    def call(self, inputs, state):
        
        Axt1 = tf.einsum('ij,bi->bj', self.A, inputs)           # Ax_{t+1}
        Bht = tf.einsum('ij,bi->bj', self.B, state)             # Bh_{t}                   
        next_state = self._activation(Axt1 + Bht + self.c)      # h_{t+1}
        
        Dht1 = tf.einsum('ij,bi->bj', self.D, next_state)       # Dh_{t+1}
        output = self._activation(Dht1 + self.e)                # o_{t+1}       
        
        return output, next_state

def custom_rnn_layer(input_tensor, num_hidden, num_outputs, activation_fn=None, reuse=False):
    
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(input_tensor, axis=1)
    
    # Define a custom rnn cell with tensorflow
    rnn_cell = CustomRNNCell(num_hidden, num_outputs, activation_fn=activation_fn, reuse=reuse)
    
    # Initialize zero hidden state
    initial_state = tf.zeros(shape=[tf.shape(input_tensor)[0], num_hidden], dtype=tf.float32)
    
    # Get rnn cell outputs and hidden states
    outputs, states = tf.nn.static_rnn(rnn_cell, x, initial_state=initial_state, dtype=tf.float32)
    
    return outputs[-1]
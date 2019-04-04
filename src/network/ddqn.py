import tensorflow as tf

from .memory import Memory

from . import DeepQNetwork

class DuelingDeepQNetwork(DeepQNetwork):

    def __init__(self, **kwargs):
        super().__init__(input_size=kwargs['input_size'], action_size=kwargs['action_size'], num_episodes=kwargs['num_episodes'], state_stack_size=kwargs['state_stack_size'])


    def build(self, name):
        with tf.device(self.device):
            with tf.variable_scope(name):
                self.inputs_ = tf.placeholder(tf.float32, [None, *self.input_size], name="inputs")

                self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

                self.target_Q = tf.placeholder(tf.float32, [None], name="target")

                self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                              filters=128,
                                              kernel_size=[3, 3],
                                              strides=[1, 1],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              bias_initializer=tf.contrib.layers.xavier_initializer(),
                                              use_bias=True,
                                              name="conv1",
                                              activation=tf.nn.elu
                                              )

                self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

                self.mp1 = tf.layers.max_pooling2d(inputs=self.conv1_out, strides=[2,2], pool_size=2, name="mp1")

                self.conv2 = tf.layers.conv2d(inputs=self.mp1,
                                              filters=256,
                                              kernel_size=[2, 2],
                                              strides=[2, 2],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              bias_initializer=tf.contrib.layers.xavier_initializer(),
                                              use_bias=True,
                                              name="conv2",
                                              activation=tf.nn.elu
                                              )

                self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

                self.mp2 = tf.layers.max_pooling2d(inputs=self.conv2_out, strides=[2,2], pool_size=2, name="mp2")

                self.conv3 = tf.layers.conv2d(inputs=self.mp2,
                                              filters=256,
                                              kernel_size=[2, 2],
                                              strides=[1, 1],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              bias_initializer=tf.contrib.layers.xavier_initializer(),
                                              use_bias=True,
                                              name="conv3",
                                              activation=tf.nn.elu
                                              )

                self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

                self.mp3 = tf.layers.max_pooling2d(inputs=self.conv3_out, strides=[2,2], pool_size=2, name="mp3")

                self.flatten = tf.contrib.layers.flatten(self.mp3)

                self.value_fc = tf.layers.dense(inputs=self.flatten,
                                                units=256,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="value_fc")

                self.value = tf.layers.dense(inputs=self.value_fc,
                                             units=1,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="value")

                self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                    units=256,
                                                    activation=tf.nn.elu,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name="advantage_fc")

                self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                                 units=self.action_size,
                                                 activation=None,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 name="advantages")

                # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
                self.output = self.value + tf.subtract(self.advantage,
                                                       tf.reduce_mean(self.advantage, axis=1, keepdims=True))

                self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), 1)

                self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

                # Reference https://arxiv.org/pdf/1511.06581.pdf

                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            return self.output
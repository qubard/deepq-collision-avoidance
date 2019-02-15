import tensorflow as tf
import numpy as np

from .memory import Memory
from flenv.src.env import Environment


class DeepQNetwork():

    def __init__(self, input_size, action_size, num_episodes, \
                 explore_prob=0.1 ,gamma=0.9, learning_rate=0.01, max_memory_size=100000, \
                 max_steps=10000, batch_size=1000, name='DeepQNetwork'):
        self.input_size = input_size # tuple representing size of input
        self.action_size = action_size # of available actions
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate

        self.explore_prob = explore_prob

        self.gamma = gamma

        self._reset_env()

        self.memory = Memory(max_memory_size)

        self.sess = tf.Session()

        self.network = self.build(name)

        self.saver = tf.train.Saver()

    def build(self, name):# Build the network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *self.input_size], name="inputs")

            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=tf.nn.activation.tanh)

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            return self.output


    def initialize_memory(self):
        while not self.memory.full:
            action, action_vec, explore_prob = self.generate_action(self.explore_prob)

            state = self.env.state

            next_state, reward = self.env.step(action)

            self.memory.add([state, action_vec, reward, next_state, self.env.done])

            if self.env.done:
                self._reset_env()

    # See https://arxiv.org/pdf/1312.5602.pdf
    def generate_action(self, probability):
        explore_prob = np.random.rand()

        actions = np.zeros(self.action_size)
        if explore_prob < probability:
            # take a random action (generate a one-hot vector for it)
            action = np.random.randint(0, self.action_size + 1)
        else:
            # Estimate based on the current state using the q value network
            action = np.argmax(self.sess.run(self.model, feed_dict={self.inputs_: self.env.state}))

        actions[action] = 1
        return action, actions, explore_prob

    def _reset_env(self):
        self.env = Environment(render=False, max_projectiles=100, seed=0, scale=5, fov_size=200)

    def train(self):
        self.initialize_memory()

        for episode in range(self.num_episodes):
            self._reset_env()

            total_reward = 0

            # Stack the frames

            for _ in range(self.max_steps):
                action, action_vec, explore_prob = self.generate_action(self.explore_prob)

                state = self.env.state

                next_state, reward = self.env.step(action)

                total_reward += reward

                self.memory.add([state, action_vec, reward, next_state, self.env.done])

                batch_sample = self.memory.sample(self.batch_size)
                batch_states = np.array([batch[3] for batch in batch_sample])
                batch_actions = np.array([batch[1] for batch in batch_sample])
                batch_next_states = np.array([batch[3] for batch in batch_sample])

                next_qs = self.sess.run(self.model, feed_dict={self.inputs_: batch_next_states})

                target_q = np.array(self.batch_size)

                # Sample a mini-batch and update the network's parameters
                for i in range(self.batch_size):
                    sample = batch_sample[i]

                    if self.env.done:
                        target_q.append(sample[2]) # add reward r_j
                    else:
                        target_q.append(sample[2] + self.gamma * np.max(next_qs[i]))

                # Run and compute loss against target_q given action
                _, loss = self.sess.run(self.model, feed_dict={
                    self.inputs_: batch_states,
                    self.target_Q: target_q,
                    self.actions_: batch_actions
                })

                if self.env.done:
                    break

                print('Episode: {}'.format(episode), 'Loss: {}'.format(loss), 'Total reward: {}'.format(total_reward))

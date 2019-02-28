import tensorflow as tf
import numpy as np

from .memory import Memory
from flenv.src.env import Environment


class DeepQNetwork():

    def __init__(self, input_size, action_size, num_episodes, \
                 gamma=0.9, learning_rate=0.001, max_memory_size=153600, \
                 max_steps=1500, batch_size=512, memory_frame_rate=1, name='DeepQNetwork',
                 checkpoint_dir="checkpoints/", prev_frame_size=4):

        self.memory_frame_rate = memory_frame_rate
        self.input_size = input_size # tuple representing size of input
        self.action_size = action_size # of available actions
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate

        self.prev_frame_size = prev_frame_size

        self.explore_start = 1.0
        self.explore_stop = 0.01

        self.decay_rate = 0.001

        self.checkpoint_dir = checkpoint_dir

        assert(batch_size <= max_memory_size)

        self.gamma = gamma

        self._reset_env()

        self.memory = Memory(max_memory_size, reward_key=2)

        self.sess = tf.Session()

        self.model = self.build(name)

        self.saver = tf.train.Saver()

        self.reward = tf.Variable(0.0)

        self._initialize_tensorboard()

        # Initialize the previous frame stack
        from collections import deque
        self.state_stack = deque(maxlen=self.prev_frame_size)

    def _initialize_tensorboard(self):
        self.writer = tf.summary.FileWriter("tensorboard", graph=self.sess.graph)

        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Total reward", self.reward)

        self.write_op = tf.summary.merge_all()

    def build(self, name):# Build the network
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
                                          filters=512,
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

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=tf.nn.tanh)

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), 1)

            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self.sess.run(tf.global_variables_initializer())

            return self.output

    def _stacked_state(self):
        return np.transpose(np.stack(self.state_stack))

    def _initialize_memory(self):
        step = 0
        while not self.memory.full(0.1):
            action, action_vec = self.generate_action(step)

            state = self.env.get_raster()

            self.env.clear_raster() # very important! clear the raster

            next_state, reward = self.env.step(action)

            if step % self.memory_frame_rate == 0:
                self.state_stack.append(state)

                if len(self.state_stack) == self.prev_frame_size:
                    # Stack the current state and the next state
                    stacked_state = self._stacked_state()
                    self.state_stack.append(next_state)
                    stacked_next_state = self._stacked_state()

                    experience = [stacked_state, action_vec, reward, stacked_next_state, self.env.done]
                    self.memory.add(experience)

            if self.env.done:
                self._reset_env()
                self.state_stack.clear()
                step = 0
            else:
                step += 1

        print("Finished initializing memory")

    def get_action_for_env(self):
        stacked_state = self._stacked_state()
        return np.argmax(self.sess.run(self.model, feed_dict={\
                self.inputs_: np.reshape(stacked_state, [1, *self.input_size])}
            ))

    # See https://arxiv.org/pdf/1312.5602.pdf
    def generate_action(self, decay_step):
        exp_exp_tradeoff = np.random.rand()
        explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * decay_step)

        actions = np.zeros(self.action_size)

        if exp_exp_tradeoff < explore_probability or len(self.state_stack) < self.prev_frame_size:
            # take a random action (generate a one-hot vector for it)
            action = np.random.randint(0, self.action_size)
        else:
            # Estimate based on the current state using the q value network
            action = self.get_action_for_env()

        actions[action] = 1
        return action, actions

    def _reset_env(self):
        self.env = Environment(render=False, max_projectiles=100, scale=5, fov_size=int(self.input_size[1] / 2))

    def restore_checkpoint(self, checkpoint):
        self.restore_last_checkpoint(checkpoint)

    def restore_last_checkpoint(self, checkpoint=None):
        import os
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if checkpoint:
                ckpt_name = "model%s.ckpt" % checkpoint
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def train(self):
        self._initialize_memory()

        self.state_stack.clear()

        for episode in range(self.num_episodes):
            self._reset_env()

            self.sess.run(self.reward.assign(0))

            for step in range(self.max_steps):
                action, action_vec = self.generate_action(step)

                state = self.env.get_raster()

                self.env.clear_raster()

                next_state, reward = self.env.step(action)

                self.sess.run(self.reward.assign_add(reward))

                if step % self.memory_frame_rate == 0:

                    # Stack the frames
                    self.state_stack.append(state)

                    if len(self.state_stack) == self.prev_frame_size:
                        # Stack the current state and the next state
                        stacked_state = self._stacked_state()
                        self.state_stack.append(next_state)
                        stacked_next_state = self._stacked_state()

                        experience = [stacked_state, action_vec, reward, stacked_next_state, self.env.done]
                        self.memory.add(experience)

                batch_sample = self.memory.sample(self.batch_size)
                batch_states = np.array([batch[0] for batch in batch_sample])
                batch_actions = np.array([batch[1] for batch in batch_sample])
                batch_next_states = np.array([batch[3] for batch in batch_sample])

                next_qs = self.sess.run(self.model, feed_dict={self.inputs_: batch_next_states})

                target_q = []

                # Sample a mini-batch and update the network's parameters
                for i in range(self.batch_size):
                    sample = batch_sample[i]

                    if self.env.done:
                        target_q.append(sample[2]) # add reward r_j
                    else:
                        target_q.append(sample[2] + self.gamma * np.max(next_qs[i]))

                # Run and compute loss against target_q given action
                _, loss, summary, output = self.sess.run([self.optimizer, self.loss, self.write_op, self.output], feed_dict={
                    self.inputs_: batch_states,
                    self.target_Q: np.array(target_q),
                    self.actions_: batch_actions
                })

                self.writer.add_summary(summary, episode)
                self.writer.flush()

                if self.env.done:
                    break

                print('Episode: {}'.format(episode), 'Loss: {}'.format(loss), 'Total reward: {}'.format(self.sess.run(self.reward)))

            if episode % 5 == 0:
                save_path = self.saver.save(self.sess, "checkpoints/model%s.ckpt" % episode)
                print("Model saved in path: %s" % save_path)
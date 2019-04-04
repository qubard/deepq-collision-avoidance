from src.network.dqn import DeepQNetwork

dqn = DeepQNetwork(input_size=[100, 100, 4], max_steps=1000, \
                   state_stack_size=4, action_size=9, num_episodes=10000, memory_frame_rate=3)

n_simulations = 50

import numpy as np

for checkpoint in range(6, 41, 5):
    restored = dqn.restore_last_checkpoint(checkpoint)
    avg_reward = 0.0
    rewards = np.array([])
    for _ in range(0, n_simulations):
        frames, total_reward = dqn.simulate()
        rewards = np.append(rewards, total_reward)
    print('Checkpoint: {}'.format(checkpoint),
          'Avg Total Reward: {}'.format(np.mean(rewards)),
          'StdDev: {}'.format(np.std(rewards)))

#import imageio

#imageio.mimwrite('animation.gif', frames, duration=25/len(frames))
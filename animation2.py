from src.network.dqn import DeepQNetwork


dqn = DeepQNetwork(input_size=[100, 100, 4], batch_size=60, learning_rate=0.0001, max_memory_size=5500, state_stack_size=4, action_size=9, \
                              checkpoint_dir='workingmodel', num_episodes=10000, memory_frame_rate=3, device='gpu')

dqn.restore_last_checkpoint(15)
frames, reward = dqn.simulate()

import imageio

imageio.mimwrite('animation.gif', frames, duration=25/len(frames))

print(dqn.env.total_reward)

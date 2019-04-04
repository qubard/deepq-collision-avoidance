from src.network.dqn import DeepQNetwork

dqn = DeepQNetwork(input_size=[100, 100, 4], max_steps=1000, \
                   state_stack_size=4, action_size=9, num_episodes=10000, memory_frame_rate=3)
dqn.restore_last_checkpoint(69)
frames, reward = dqn.simulate()

import imageio

imageio.mimwrite('animation.gif', frames, duration=25/len(frames))
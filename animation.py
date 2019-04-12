from src.network.ddqn import DuelingDeepQNetwork


ddqn = DuelingDeepQNetwork(input_size=[80, 80, 4], batch_size=60, learning_rate=0.0003, max_memory_size=5500, state_stack_size=4, action_size=9, \
                              num_episodes=10000, memory_frame_rate=3, device='gpu')

ddqn.restore_last_checkpoint(23)
frames, reward = ddqn.simulate()

import imageio

imageio.mimwrite('animation.gif', frames, duration=25/len(frames))
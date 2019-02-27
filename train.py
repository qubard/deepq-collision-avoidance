from src.network.dqn import DeepQNetwork

network = DeepQNetwork(input_size=[200, 200, 1], action_size=9, num_episodes=1500, memory_frame_rate=3)
network.train()
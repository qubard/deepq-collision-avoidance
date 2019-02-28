from src.network.dqn import DeepQNetwork

network = DeepQNetwork(input_size=[50, 50, 4], action_size=8, num_episodes=10000, memory_frame_rate=3, device='gpu')
network.train()